"""Shared immutable candidate lifecycle for pipeline improvement policies."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.agents.pipeline_improvement import ImprovementProposal
from backend.evaluation.constrained_schemas import (
    CandidateSourceSpec,
    ConstrainedEvaluationConfig,
)
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    repo_path,
    source_bundle_hash,
    verify_frozen_file,
)
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationConfigV3,
    ConstrainedEvaluationRunV3,
    EvaluationFailureV3,
    V3EvaluationSummary,
)
from backend.evaluation.objective_v3_workflow import run_evaluation_v3_file
from backend.evaluation.constrained import run_constrained_evaluation
from backend.llm import LLMClient
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    OperationalRepetition,
    ProposalObjective,
    SearchMeasurementReport,
)


PROTECTED_SEARCH_PATHS = {
    "manifest.json",
    "pipeline_contract.json",
    "requirements.txt",
    "run_pipeline.py",
}
EDITABLE_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}


@dataclass(frozen=True)
class ReboundSuites:
    primary: Path
    alternate: Path | None


@dataclass(frozen=True)
class CandidateEvaluationResult:
    summary: V3EvaluationSummary | None
    report_path: Path
    duration_seconds: float
    failure: str | None = None
    proposal_feedback: Mapping[str, Any] | None = None


def copy_root_candidate(
    *,
    base_suite: Path,
    repository_root: Path,
    destination: Path,
) -> list[str]:
    """Copy only suite-declared source and frozen runner metadata into one node."""

    payload = _json_object(base_suite)
    source_spec = CandidateSourceSpec.model_validate(payload["candidate_source"])
    if source_bundle_hash(source_spec, repository_root) != source_spec.sha256:
        raise ValueError("Root candidate source no longer matches the frozen suite.")
    source_cwd = repo_path(source_spec.cwd, repository_root)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    copied: set[str] = set()
    for raw in source_spec.paths:
        source = repo_path(raw, repository_root)
        relative = source.relative_to(source_cwd)
        _copy_confined(source, destination / relative)
        copied.add(relative.as_posix())
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Base suite has no execution mapping.")
    for key in ("pipeline_contract", "manifest"):
        spec = execution.get(key)
        if not isinstance(spec, dict):
            raise ValueError(f"Base suite has no frozen {key}.")
        source = verify_frozen_file(
            _frozen_file_model(spec), repository_root
        )
        relative = source.relative_to(source_cwd)
        if relative.as_posix() not in copied:
            _copy_confined(source, destination / relative)
            copied.add(relative.as_posix())
    _validate_no_symlinks(destination)
    return _relative_source_paths(payload, destination, repository_root)


def copy_child_candidate(parent: Path, destination: Path) -> None:
    """Create a complete source snapshot for one child without runtime artifacts."""

    parent = parent.resolve()
    destination = destination.resolve()
    _validate_no_symlinks(parent)
    shutil.copytree(parent, destination, symlinks=False)
    _validate_no_symlinks(destination)


def editable_candidate_paths(source_paths: list[str]) -> list[str]:
    """Return candidate-owned text files eligible for experimental edits."""

    result = []
    for relative in source_paths:
        path = Path(relative)
        if relative in PROTECTED_SEARCH_PATHS or path.parts[0] == "tests":
            continue
        if path.suffix.lower() in EDITABLE_SUFFIXES:
            result.append(path.as_posix())
    return sorted(result)


def apply_proposal(
    *,
    candidate_dir: Path,
    parent_dir: Path,
    proposal: ImprovementProposal,
    editable_paths: list[str],
    patch_path: Path,
) -> None:
    """Apply exact replacements, then enforce syntax and protected-file invariants."""

    candidate_dir = candidate_dir.resolve()
    parent_dir = parent_dir.resolve()
    allowed = set(editable_paths)
    originals: dict[Path, str] = {}
    updates: dict[Path, str] = {}
    relative_paths: dict[Path, str] = {}
    for edit in proposal.edits:
        if edit.relative_path not in allowed:
            raise ValueError(f"Path is not editable: {edit.relative_path}")
        target = _confined(candidate_dir, edit.relative_path)
        if target not in originals:
            originals[target] = target.read_text(encoding="utf-8")
            updates[target] = originals[target]
            relative_paths[target] = edit.relative_path
        content = updates[target]
        old_text = _normalized_old_text(content, edit.old_text)
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ValueError(
                f"Expected exactly one old_text match in {edit.relative_path}; "
                f"found {occurrences}."
            )
        revised = content.replace(old_text, edit.new_text, 1)
        if target.suffix == ".py":
            ast.parse(revised, filename=str(target))
        updates[target] = revised

    patches: list[str] = []
    for target, revised in updates.items():
        content = originals[target]
        relative = relative_paths[target]
        patches.extend(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                revised.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    if not patches:
        raise ValueError("Proposal produced no source diff.")
    for target, revised in updates.items():
        target.write_text(revised, encoding="utf-8")
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(patch_path, "".join(patches))
    _validate_no_symlinks(candidate_dir)
    _validate_protected_files(parent_dir, candidate_dir)
    for path in candidate_dir.rglob("*.py"):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _normalized_old_text(content: str, old_text: str) -> str:
    if content.count(old_text) == 1:
        return old_text
    if old_text.rstrip("\n") == content.rstrip("\n"):
        return content
    return old_text


def rebind_evaluation_suites(
    *,
    base_suite: Path,
    candidate_dir: Path,
    candidate_id: str,
    output_dir: Path,
    repository_root: Path,
    require_extensibility: bool,
) -> ReboundSuites:
    """Bind frozen evaluator inputs to a relocated immutable candidate snapshot."""

    repository_root = repository_root.resolve()
    candidate_dir = candidate_dir.resolve()
    _require_within(candidate_dir, repository_root, "Candidate directory")
    primary_payload = _json_object(base_suite)
    if primary_payload.get("schema_version") != (
        "evaluation_constrained.regular_grid_zarr.v3"
    ):
        raise ValueError("Self-improvement requires a regular-grid objective-v3 suite.")
    alternate_payload: dict[str, Any] | None = None
    probe = primary_payload.get("extensibility_probe")
    if isinstance(probe, dict) and probe.get("enabled"):
        suite = probe.get("suite")
        if not isinstance(suite, dict):
            raise ValueError("Extensibility probe has no frozen suite.")
        alternate_path = verify_frozen_file(
            _frozen_file_model(suite), repository_root
        )
        alternate_payload = _json_object(alternate_path)
    elif require_extensibility:
        raise ValueError("Search protocol requires an extensibility probe.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    alternate_output: Path | None = None
    if alternate_payload is not None:
        rebound_alternate = _rebind_common(
            alternate_payload,
            candidate_dir=candidate_dir,
            candidate_id=candidate_id,
            repository_root=repository_root,
        )
        alternate_config = ConstrainedEvaluationConfig.model_validate(
            rebound_alternate
        )
        alternate_output = output_dir / "alternate_suite.json"
        _write_new(alternate_output, alternate_config.model_dump_json(indent=2) + "\n")
        primary_payload["extensibility_probe"] = {
            "enabled": True,
            "required": bool(probe.get("required", True)),
            "suite": {
                "path": _display(alternate_output, repository_root),
                "sha256": canonical_json_file_hash(alternate_output),
            },
        }

    rebound_primary = _rebind_common(
        primary_payload,
        candidate_dir=candidate_dir,
        candidate_id=candidate_id,
        repository_root=repository_root,
    )
    primary_config = ConstrainedEvaluationConfigV3.model_validate(rebound_primary)
    primary_output = output_dir / "primary_suite.json"
    _write_new(primary_output, primary_config.model_dump_json(indent=2) + "\n")
    return ReboundSuites(primary=primary_output, alternate=alternate_output)


class FrozenSuiteCandidateEvaluator:
    """Evaluate every policy candidate through the same frozen objective-v3 suite."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_dir: Path,
        base_suite: Path,
        llm_client: LLMClient | None,
        llm_timeout_seconds: int,
        require_extensibility: bool,
        operational_repetitions: int = 3,
        search_run_id: str,
        required_objectives: list[ProposalObjective] | None = None,
    ) -> None:
        self._repository_root = repository_root.resolve()
        self._dataset_dir = dataset_dir.resolve()
        self._base_suite = base_suite.resolve()
        self._llm_client = llm_client
        self._llm_timeout_seconds = llm_timeout_seconds
        self._require_extensibility = require_extensibility
        if not 1 <= operational_repetitions <= 9:
            raise ValueError("operational_repetitions must be between one and nine.")
        self._operational_repetitions = operational_repetitions
        self._search_run_id = search_run_id
        self._required_objectives = required_objectives or [
            "materialization_seconds",
            "consumer_samples_per_second",
            "output_bytes",
            "q_engineering",
        ]

    def evaluate(
        self,
        *,
        node_id: str,
        candidate_dir: Path,
        node_dir: Path,
    ) -> CandidateEvaluationResult:
        suites = rebind_evaluation_suites(
            base_suite=self._base_suite,
            candidate_dir=candidate_dir,
            candidate_id=node_id,
            output_dir=node_dir / "evaluation_inputs",
            repository_root=self._repository_root,
            require_extensibility=self._require_extensibility,
        )
        evaluation_dir = (
            self._dataset_dir
            / "runs"
            / "evaluations"
            / "self_improvement"
            / self._search_run_id
            / node_id
        )
        started = time.perf_counter()
        result, full_report_path = run_evaluation_v3_file(
            suites.primary,
            repository_root=self._repository_root,
            output_dir=evaluation_dir,
            llm_client=self._llm_client,
            llm_timeout_seconds=self._llm_timeout_seconds,
        )
        duration = time.perf_counter() - started
        if isinstance(result, EvaluationFailureV3):
            return CandidateEvaluationResult(
                summary=None,
                report_path=full_report_path,
                duration_seconds=duration,
                failure=f"{result.error_type}: {result.error}",
            )
        if not isinstance(result, ConstrainedEvaluationRunV3):
            return CandidateEvaluationResult(
                summary=None,
                report_path=full_report_path,
                duration_seconds=duration,
                failure=f"Unexpected evaluator result: {type(result).__name__}",
            )
        available = set(result.summary.operational_objectives) | set(
            result.summary.engineering_objective
        )
        if not result.summary.feasible or not set(self._required_objectives).issubset(
            available
        ):
            return CandidateEvaluationResult(
                summary=result.summary,
                report_path=full_report_path,
                duration_seconds=duration,
                proposal_feedback=_proposal_feedback(result),
            )

        primary_config = ConstrainedEvaluationConfigV3.model_validate_json(
            suites.primary.read_text(encoding="utf-8")
        )
        v2_config = primary_config.as_v2()
        operational_suite = node_dir / "evaluation_inputs" / "operational_suite.json"
        _write_new(operational_suite, v2_config.model_dump_json(indent=2) + "\n")
        repetitions = [
            OperationalRepetition(
                repetition=1,
                evaluation_report=_display(full_report_path, self._repository_root),
                objectives=result.summary.operational_objectives,
            )
        ]
        for repetition in range(2, self._operational_repetitions + 1):
            repeat_dir = evaluation_dir / f"operational_repeat_{repetition:02d}"
            repeat_dir.mkdir(parents=False, exist_ok=False)
            repeat = run_constrained_evaluation(
                v2_config,
                repository_root=self._repository_root,
                config_file=operational_suite,
                run_dir=repeat_dir,
                execution_policy=primary_config.execution_policy,
            )
            repeat_path = repeat_dir / "evaluation.json"
            _write_new(repeat_path, repeat.model_dump_json(indent=2) + "\n")
            if not repeat.summary.feasible:
                return CandidateEvaluationResult(
                    summary=None,
                    report_path=repeat_path,
                    duration_seconds=time.perf_counter() - started,
                    failure=(
                        f"Operational repetition {repetition} failed a hard gate."
                    ),
                )
            repetitions.append(
                OperationalRepetition(
                    repetition=repetition,
                    evaluation_report=_display(repeat_path, self._repository_root),
                    objectives=repeat.summary.objectives,
                )
            )
        objective = _median_objective(repetitions, result.summary)
        summary_payload = result.summary.model_dump(mode="json")
        operational = {
            "materialization_seconds": objective.materialization_seconds,
            "consumer_samples_per_second": objective.consumer_samples_per_second,
            "output_bytes": objective.output_bytes,
        }
        summary_payload["diagnostic_operational_metrics"] = operational
        summary_payload["operational_objectives"] = operational
        summary_payload["objective_groups"] = None
        aggregate_summary = V3EvaluationSummary.model_validate(summary_payload)
        measurement = SearchMeasurementReport(
            node_id=node_id,
            full_evaluation_report=_display(full_report_path, self._repository_root),
            operational_repetitions=repetitions,
            objective=objective,
        )
        measurement_path = node_dir / "search_measurement.json"
        _write_new(measurement_path, measurement.model_dump_json(indent=2) + "\n")
        return CandidateEvaluationResult(
            summary=aggregate_summary,
            report_path=measurement_path,
            duration_seconds=time.perf_counter() - started,
            proposal_feedback=_proposal_feedback(result, repetitions),
        )


def candidate_source_paths(
    base_suite: Path,
    candidate_dir: Path,
    repository_root: Path,
) -> list[str]:
    """Map the base suite's declared source bundle to a candidate snapshot."""

    return _relative_source_paths(
        _json_object(base_suite), candidate_dir.resolve(), repository_root.resolve()
    )


def base_suite_candidate_dir(base_suite: Path, repository_root: Path) -> Path:
    """Return the source cwd pinned by a frozen base suite."""

    payload = _json_object(base_suite)
    source = CandidateSourceSpec.model_validate(payload["candidate_source"])
    return repo_path(source.cwd, repository_root)


def base_suite_search_invariants(base_suite: Path) -> dict[str, Any]:
    """Expose evaluator-visible interfaces that proposals must preserve."""

    payload = _json_object(base_suite)
    grid = payload.get("grid")
    execution = payload.get("execution")
    return {
        "candidate_id": payload.get("candidate_id"),
        "candidate_grid": grid.get("candidate_grid") if isinstance(grid, dict) else None,
        "candidate_channel_mappings": (
            [
                {
                    "field_id": channel.get("field_id"),
                    "candidate": channel.get("candidate"),
                }
                for channel in grid.get("channels", [])
                if isinstance(channel, dict)
            ]
            if isinstance(grid, dict)
            else []
        ),
        "output_policy": payload.get("output_policy"),
        "command_template": (
            execution.get("command") if isinstance(execution, dict) else None
        ),
        "protected_source_paths": sorted(PROTECTED_SEARCH_PATHS),
        "rule": (
            "These names, paths, mappings, policies, identities, and command semantics "
            "are frozen evaluator interfaces. Preserve them exactly."
        ),
    }


def _rebind_common(
    payload: Mapping[str, Any],
    *,
    candidate_dir: Path,
    candidate_id: str,
    repository_root: Path,
) -> dict[str, Any]:
    rebound = json.loads(json.dumps(payload))
    source_paths = _relative_source_paths(rebound, candidate_dir, repository_root)
    source_spec = CandidateSourceSpec(
        cwd=_display(candidate_dir, repository_root),
        paths=[_display(candidate_dir / relative, repository_root) for relative in source_paths],
        sha256="0" * 64,
    )
    source_spec.sha256 = source_bundle_hash(source_spec, repository_root)
    # The protected manifest and receipts retain the generated pipeline identity.
    # The search-node identity belongs in suite/run lineage, not candidate claims.
    rebound["suite_id"] = f"{payload['suite_id']}__{candidate_id}"
    rebound["candidate_source"] = source_spec.model_dump(mode="json")
    execution = rebound.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Suite execution mapping is missing.")
    for key, filename in (
        ("pipeline_contract", "pipeline_contract.json"),
        ("manifest", "manifest.json"),
    ):
        path = candidate_dir / filename
        execution[key] = {
            "path": _display(path, repository_root),
            "sha256": canonical_json_file_hash(path),
        }
    return rebound


def _median_objective(
    repetitions: list[OperationalRepetition],
    summary: V3EvaluationSummary,
) -> ObjectiveVector:
    values = [item.objectives for item in repetitions]
    return ObjectiveVector(
        materialization_seconds=float(
            statistics.median(item["materialization_seconds"] for item in values)
        ),
        consumer_samples_per_second=float(
            statistics.median(
                item["consumer_samples_per_second"] for item in values
            )
        ),
        output_bytes=int(statistics.median(item["output_bytes"] for item in values)),
        q_engineering=(
            float(summary.engineering_objective["q_engineering"])
            if "q_engineering" in summary.engineering_objective
            else None
        ),
    )


def _proposal_feedback(
    result: ConstrainedEvaluationRunV3,
    repetitions: list[OperationalRepetition] | None = None,
) -> dict[str, Any]:
    return {
        "deterministic_feedback": [
            item.model_dump(mode="json") for item in result.deterministic.summary.feedback
        ],
        "operational_evidence": _operational_evidence(result, repetitions or []),
        "engineering_components": [
            component.model_dump(mode="json")
            for component in result.engineering_quality.components
        ],
        "extensibility_probe": {
            "status": result.extensibility_probe.status.value,
            "feedback_code": result.extensibility_probe.feedback_code,
            "error": result.extensibility_probe.error,
        },
    }


def _operational_evidence(
    result: ConstrainedEvaluationRunV3,
    repetitions: list[OperationalRepetition],
) -> dict[str, Any]:
    diagnostics = result.deterministic.summary.diagnostics
    storage = _mapping(diagnostics.get("output_storage"))
    workload = _mapping(diagnostics.get("consumer_workload"))
    timings = _mapping(diagnostics.get("timings"))
    resources = _mapping(diagnostics.get("resources"))
    cache = _mapping(diagnostics.get("cache_evidence"))
    objective_repetitions = [
        {
            "repetition": item.repetition,
            **dict(item.objectives),
        }
        for item in repetitions
    ]
    variation: dict[str, Any] = {}
    for name in (
        "materialization_seconds",
        "consumer_samples_per_second",
        "output_bytes",
    ):
        values = [float(item.objectives[name]) for item in repetitions]
        if not values:
            continue
        median = float(statistics.median(values))
        variation[name] = {
            "minimum": min(values),
            "median": median,
            "maximum": max(values),
            "relative_span": (
                (max(values) - min(values)) / abs(median) if median else 0.0
            ),
        }
    observations = _operational_observations(storage, workload, timings, cache)
    return {
        "objective_repetitions": objective_repetitions,
        "variation": variation,
        "output_storage": _pick(
            storage,
            "zarr_format",
            "consolidated_metadata",
            "output_bytes",
            "chunk_bytes",
            "metadata_bytes",
            "object_count",
            "chunk_object_count",
            "metadata_object_count",
            "dataset_open_latency_seconds",
        ),
        "consumer_workload": _pick(
            workload,
            "kind",
            "access_pattern",
            "effective_cache_mode",
            "repetitions",
            "sample_shape",
            "dtype",
            "samples",
            "durations_seconds",
            "samples_per_second",
            "median_samples_per_second",
            "min_samples_per_second",
            "max_samples_per_second",
        ),
        "cache_evidence": cache,
        "timings": timings,
        "resources": resources,
        "actionable_observations": observations,
    }


def _operational_observations(
    storage: Mapping[str, Any],
    workload: Mapping[str, Any],
    timings: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> list[str]:
    observations: list[str] = []
    output_bytes = storage.get("output_bytes")
    metadata_bytes = storage.get("metadata_bytes")
    if isinstance(output_bytes, int) and isinstance(metadata_bytes, int) and output_bytes:
        share = metadata_bytes / output_bytes
        observations.append(
            f"Zarr metadata is {share:.1%} of measured output bytes."
        )
    samples = workload.get("samples")
    minimum = workload.get("min_samples_per_second")
    maximum = workload.get("max_samples_per_second")
    if isinstance(samples, int) and isinstance(minimum, (int, float)) and isinstance(
        maximum, (int, float)
    ):
        observations.append(
            f"Consumer timing uses {samples} samples and spans "
            f"{float(minimum):.1f}-{float(maximum):.1f} samples/s within one run."
        )
    initial_cache = _mapping(cache.get("initial"))
    if initial_cache.get("misses") == 0:
        observations.append(
            "The measured materialization path is cache-hit only; provider download "
            "changes cannot improve this objective."
        )
    initial_timing = _mapping(timings.get("initial"))
    execution = initial_timing.get("execution_seconds")
    total = initial_timing.get("total_seconds")
    if isinstance(execution, (int, float)) and isinstance(total, (int, float)) and total:
        observations.append(
            f"Candidate execution is {float(execution) / float(total):.1%} of initial "
            "end-to-end materialization time."
        )
    return observations


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick(values: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: values[key] for key in keys if key in values}


def _relative_source_paths(
    payload: Mapping[str, Any],
    candidate_dir: Path,
    repository_root: Path,
) -> list[str]:
    source_spec = CandidateSourceSpec.model_validate(payload["candidate_source"])
    source_cwd = repo_path(source_spec.cwd, repository_root)
    relative = [
        repo_path(raw, repository_root).relative_to(source_cwd).as_posix()
        for raw in source_spec.paths
    ]
    missing = [item for item in relative if not (candidate_dir / item).exists()]
    if missing:
        raise FileNotFoundError(f"Candidate snapshot is missing source paths: {missing}")
    return relative


def _validate_protected_files(parent_dir: Path, candidate_dir: Path) -> None:
    for relative in sorted(PROTECTED_SEARCH_PATHS):
        parent = parent_dir / relative
        child = candidate_dir / relative
        if not parent.exists() and not child.exists():
            continue
        if not parent.is_file() or not child.is_file():
            raise FileNotFoundError(f"Protected candidate file missing: {relative}")
        if _sha256_file(parent) != _sha256_file(child):
            raise ValueError(f"Proposal modified protected path: {relative}")


def _copy_confined(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def _confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require_within(path, root, "Candidate edit")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_no_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"Candidate source snapshots may not contain symlinks: {root}")


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes {root}: {path}") from error


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _frozen_file_model(value: Mapping[str, Any]):
    from backend.evaluation.constrained_schemas import FrozenFile

    return FrozenFile.model_validate(value)


def _display(path: Path, repository_root: Path) -> str:
    return str(path.resolve().relative_to(repository_root.resolve()))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
