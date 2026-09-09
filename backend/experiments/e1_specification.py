"""E1 specification-reliability and evaluation-coverage experiment."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from backend.access_probes import characterize_frozen_dataset
from backend.agents.contract_drafting.schemas import (
    DatasetContract,
    PipelineDownstreamUse,
    PipelineObjectivePreference,
    PipelineOptimizationRequirements,
    PipelineRequirements,
)
from backend.agents.evaluation_planning import EvaluationPlanningAgent
from backend.agents.intention_space import (
    ConversationTurn,
    IntentionSpaceAgent,
    create_intention_space_artifacts,
)
from backend.contracts import file_hash, read_contract_lock
from backend.evaluation.check_library import (
    EvaluationProfile,
    compile_evaluation_check_plan,
)
from backend.evaluation.evidence import logical_field_id
from backend.evaluation.planning import create_evaluation_planning_artifacts
from backend.llm import LLMClient, aggregate_llm_usage


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class E1DatasetConfig(_StrictModel):
    dataset_slug: str
    display_name: str
    role: Literal["development", "held_out"]
    inventory_path: str
    candidate_path: str | None = None
    access_context_path: str | None = None
    source_contract_lock_path: str
    source_fixture_manifest_path: str
    data_class: Literal["regular_rectilinear_grid", "station_time_series"]
    output_format: Literal["zarr", "parquet"]
    output_format_version: Literal[1, 3]
    workload_kind: Literal["full_field_tensor", "filtered_station_scan"]


class E1Config(_StrictModel):
    schema_version: Literal["e1_specification_coverage_config.v1"] = (
        "e1_specification_coverage_config.v1"
    )
    experiment_id: str
    model: str
    repetitions: int = Field(ge=1, le=10)
    intention_prompt: str = "default"
    evaluation_prompt: str = "default"
    datasets: list[E1DatasetConfig] = Field(min_length=1)


class E1Reference(_StrictModel):
    schema_version: Literal["e1_reference.v1"] = "e1_reference.v1"
    dataset_slug: str
    role: Literal["development", "held_out"]
    inventory_sha256: str
    fixture_manifest_sha256: str
    expected_credential_references: list[str]
    expected_contract: DatasetContract
    expected_target: dict[str, Any]
    accepted_oracle_strategies: list[str]
    expected_field_ids: list[str]
    expected_check_ids: list[str]


class E1RunResult(_StrictModel):
    schema_version: Literal["e1_run_result.v1"] = "e1_run_result.v1"
    experiment_id: str
    dataset_slug: str
    display_name: str
    role: Literal["development", "held_out"]
    repetition: int
    status: Literal["success", "failed"]
    started_at: str
    completed_at: str
    access_probe_valid: bool = False
    inventory_valid: bool = False
    specification_schema_valid: bool = False
    contract_lock_valid: bool = False
    field_selector_exact: bool = False
    scope_exact: bool = False
    objective_priority_exact: bool = False
    workload_exact: bool = False
    credential_policy_exact: bool = False
    intention_structured_consistent: bool = False
    target_exact: bool = False
    logical_mapping_valid: bool = False
    oracle_strategy_valid: bool = False
    mandatory_check_recall: float = 0.0
    unauthorized_check_count: int = 0
    unnecessary_proposal_count: int = 0
    unresolved_gap_count: int = 0
    planning_complete: bool = False
    executable_suite_ready: bool = False
    run_success: bool = False
    intention_latency_seconds: float | None = None
    planning_latency_seconds: float | None = None
    intention_usage: dict[str, Any] | None = None
    planning_usage: dict[str, Any] | None = None
    resolved_checks_by_category: dict[str, int] = Field(default_factory=dict)
    failure_type: str | None = None
    failure_message: str | None = None


_REQUIREMENTS_GRID = PipelineRequirements(
    downstream_use=PipelineDownstreamUse(
        kind="ml_training",
        workload_kind="full_field_tensor",
        description="Repeated shuffled reads of complete multi-channel spatial fields.",
    ),
    optimization=PipelineOptimizationRequirements(
        ordered_objectives=[
            PipelineObjectivePreference(
                objective="consumer_samples_per_second",
                direction="maximize",
                priority=1,
            ),
            PipelineObjectivePreference(
                objective="output_bytes",
                direction="minimize",
                priority=2,
            ),
        ],
        descriptive_measurements=["materialization_seconds", "q_engineering"],
    ),
)
_REQUIREMENTS_STATION = _REQUIREMENTS_GRID.model_copy(
    update={
        "downstream_use": PipelineDownstreamUse(
            kind="ml_training",
            workload_kind="filtered_station_scan",
            description="Repeated filtered scans over selected station-field observations.",
        )
    }
)


def load_e1_config(path: Path) -> E1Config:
    return E1Config.model_validate_json(path.read_text(encoding="utf-8"))


def prepare_e1(*, config_path: Path, repository_root: Path) -> Path:
    """Create researcher-authored references and freeze all experiment inputs."""

    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_e1_config(config_path)
    experiment_dir = config_path.parent
    references_dir = experiment_dir / "references"
    freeze_path = experiment_dir / "freeze_manifest.json"
    if references_dir.exists() or freeze_path.exists():
        raise FileExistsError("E1 references or freeze manifest already exist")
    references_dir.mkdir(parents=True)

    for dataset in config.datasets:
        inventory_path = _repo_path(dataset.inventory_path, root)
        fixture_path = _repo_path(dataset.source_fixture_manifest_path, root)
        source_lock = read_contract_lock(
            _repo_path(dataset.source_contract_lock_path, root)
        )
        requirements = _requirements(dataset)
        contract = source_lock.contract.model_copy(
            update={
                "schema_version": "dataset_contract.v2",
                "pipeline_requirements": requirements,
                "human_confirmed": True,
            }
        )
        profile = _profile(dataset, contract)
        check_plan = compile_evaluation_check_plan(
            profile,
            engineering_enabled=True,
            extensibility_probe_enabled=False,
        )
        reference = E1Reference(
            dataset_slug=dataset.dataset_slug,
            role=dataset.role,
            inventory_sha256=file_hash(inventory_path),
            fixture_manifest_sha256=file_hash(fixture_path),
            expected_credential_references=_credential_references(
                fixture_path=fixture_path,
                access_context_path=(
                    _repo_path(dataset.access_context_path, root)
                    if dataset.access_context_path
                    else None
                ),
            ),
            expected_contract=contract,
            expected_target={
                "data_class": dataset.data_class,
                "output_format": dataset.output_format,
                "output_format_version": dataset.output_format_version,
                "workload_kind": dataset.workload_kind,
                "engineering_profile": "general_pipeline",
            },
            accepted_oracle_strategies=["independent_trusted_reference"],
            expected_field_ids=_field_ids(contract),
            expected_check_ids=[check.check_id for check in check_plan.checks],
        )
        output = references_dir / dataset.dataset_slug / "expected.json"
        output.parent.mkdir()
        output.write_text(reference.model_dump_json(indent=2) + "\n", encoding="utf-8")

    protocol = experiment_dir / "PROTOCOL.md"
    protocol.write_text(_protocol_text(config), encoding="utf-8")
    frozen_paths = _freeze_paths(config_path=config_path, config=config, root=root)
    freeze = {
        "schema_version": "experiment_freeze_manifest.v1",
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "repetitions": config.repetitions,
        "files": [
            {"path": _display(path, root), "sha256": file_hash(path)}
            for path in sorted(frozen_paths)
        ],
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_journal(experiment_dir, "prepared", {"freeze_manifest": str(freeze_path)})
    return freeze_path


def run_e1_phase(
    *,
    config_path: Path,
    repository_root: Path,
    phase: Literal["development", "held_out"],
) -> list[Path]:
    """Run first-attempt specification and planning calls for one frozen phase."""

    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_e1_config(config_path)
    experiment_dir = config_path.parent
    verify_e1_freeze(experiment_dir / "freeze_manifest.json", root)
    llm = LLMClient(model=config.model, timeout_seconds=180)
    intention_agent = IntentionSpaceAgent(llm, prompt_name=config.intention_prompt)
    planning_agent = EvaluationPlanningAgent(llm, prompt_name=config.evaluation_prompt)
    outputs: list[Path] = []
    for dataset in config.datasets:
        if dataset.role != phase:
            continue
        reference = E1Reference.model_validate_json(
            (experiment_dir / "references" / dataset.dataset_slug / "expected.json").read_text(
                encoding="utf-8"
            )
        )
        for repetition in range(1, config.repetitions + 1):
            run_dir = experiment_dir / "runs" / dataset.dataset_slug / f"r{repetition}"
            if run_dir.exists():
                raise FileExistsError(f"E1 run already exists: {run_dir}")
            run_dir.mkdir(parents=True)
            result = _run_one(
                config=config,
                dataset=dataset,
                reference=reference,
                repetition=repetition,
                run_dir=run_dir,
                root=root,
                intention_agent=intention_agent,
                planning_agent=planning_agent,
            )
            result_path = run_dir / "result.json"
            result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
            outputs.append(result_path)
            _append_journal(
                experiment_dir,
                "run_completed",
                {
                    "dataset_slug": dataset.dataset_slug,
                    "role": dataset.role,
                    "repetition": repetition,
                    "status": result.status,
                    "run_success": result.run_success,
                },
            )
    return outputs


def _run_one(
    *,
    config: E1Config,
    dataset: E1DatasetConfig,
    reference: E1Reference,
    repetition: int,
    run_dir: Path,
    root: Path,
    intention_agent: IntentionSpaceAgent,
    planning_agent: EvaluationPlanningAgent,
) -> E1RunResult:
    started = datetime.now(UTC).isoformat()
    base = {
        "experiment_id": config.experiment_id,
        "dataset_slug": dataset.dataset_slug,
        "display_name": dataset.display_name,
        "role": dataset.role,
        "repetition": repetition,
        "started_at": started,
    }
    try:
        inventory_path = _repo_path(dataset.inventory_path, root)
        fixture_path = _repo_path(dataset.source_fixture_manifest_path, root)
        access_context = (
            _repo_path(dataset.access_context_path, root)
            if dataset.access_context_path
            else None
        )
        metadata, probe, characterisation = characterize_frozen_dataset(
            inventory_path=inventory_path,
            fixture_manifest_path=fixture_path,
            candidate_path=(
                _repo_path(dataset.candidate_path, root)
                if dataset.candidate_path
                else None
            ),
            access_context_path=access_context,
            output_dir=run_dir / "characterisation",
        )
        interaction = _interaction(reference, dataset)
        intention_run, intention_artifacts = create_intention_space_artifacts(
            agent=intention_agent,
            run_id=f"{dataset.dataset_slug}-r{repetition}",
            inventory_path=characterisation.inventory,
            interaction=interaction,
            supported_capabilities={
                "evaluation_targets": [reference.expected_target],
                "credential_references": metadata.credential_references,
                "required_output_format": dataset.output_format,
                "provider_network_during_evaluation": False,
            },
            repository_root=root,
            output_dir=run_dir / "specification",
            clarification_resolver=lambda questions, active: _clarification_answers(
                questions=questions,
                reference=reference,
                dataset=dataset,
            ),
            max_clarification_rounds=1,
        )
        planning_run, _ = create_evaluation_planning_artifacts(
            agent=planning_agent,
            plan_id=f"{dataset.dataset_slug}-r{repetition}-plan",
            dataset_dir=characterisation.output_dir,
            contract_lock_path=intention_artifacts.contract_lock,
            repository_root=root,
            output_dir=run_dir / "planning",
            engineering_enabled=True,
            extensibility_probe_enabled=False,
            planning_context={
                "source_mode": "immutable_local_fixture_replay",
                "provider_network_enabled": False,
                "oracle_requirement": "independent_trusted_reference",
                "target": reference.expected_target,
            },
        )
        contract = intention_run.draft.contract
        observed_ids = _field_ids(contract)
        planned_ids = {check.check_id for check in planning_run.check_plan.checks}
        expected_ids = set(reference.expected_check_ids)
        mandatory_recall = (
            len(planned_ids & expected_ids) / len(expected_ids) if expected_ids else 1.0
        )
        checks_by_category = _checks_by_category(planning_run.check_plan.checks)
        field_exact = observed_ids == reference.expected_field_ids
        scope_exact = contract.scope == reference.expected_contract.scope
        objective_exact = (
            contract.pipeline_requirements is not None
            and reference.expected_contract.pipeline_requirements is not None
            and contract.pipeline_requirements.optimization
            == reference.expected_contract.pipeline_requirements.optimization
            and planning_run.objective_policy
            == reference.expected_contract.pipeline_requirements.optimization
        )
        workload_exact = (
            contract.pipeline_requirements is not None
            and contract.pipeline_requirements.downstream_use.workload_kind
            == dataset.workload_kind
            and planning_run.profile.workload_kind == dataset.workload_kind
        )
        credentials_exact = (
            sorted(item.name for item in intention_run.draft.credential_references)
            == reference.expected_credential_references
        )
        target_exact = (
            planning_run.llm_draft.target.model_dump(mode="json")
            == reference.expected_target
        )
        mapping_valid = _mapping_valid(planning_run.llm_draft, dataset.data_class)
        oracle_valid = (
            planning_run.llm_draft.oracle.strategy
            in reference.accepted_oracle_strategies
        )
        no_proposals = not planning_run.quarantined_check_proposals
        run_success = all(
            (
                probe.ok,
                field_exact,
                scope_exact,
                objective_exact,
                workload_exact,
                credentials_exact,
                target_exact,
                mapping_valid,
                oracle_valid,
                math.isclose(mandatory_recall, 1.0),
                not (planned_ids - expected_ids),
                no_proposals,
                planning_run.planning_complete,
            )
        )
        return E1RunResult(
            **base,
            completed_at=datetime.now(UTC).isoformat(),
            status="success" if run_success else "failed",
            access_probe_valid=probe.ok,
            inventory_valid=(
                file_hash(characterisation.inventory) == reference.inventory_sha256
            ),
            specification_schema_valid=True,
            contract_lock_valid=True,
            field_selector_exact=field_exact,
            scope_exact=scope_exact,
            objective_priority_exact=objective_exact,
            workload_exact=workload_exact,
            credential_policy_exact=credentials_exact,
            intention_structured_consistent=True,
            target_exact=target_exact,
            logical_mapping_valid=mapping_valid,
            oracle_strategy_valid=oracle_valid,
            mandatory_check_recall=mandatory_recall,
            unauthorized_check_count=len(planned_ids - expected_ids),
            unnecessary_proposal_count=len(planning_run.quarantined_check_proposals),
            unresolved_gap_count=len(planning_run.quarantined_check_proposals),
            planning_complete=planning_run.planning_complete,
            executable_suite_ready=planning_run.suite_ready,
            run_success=run_success,
            intention_latency_seconds=intention_run.llm_latency_seconds,
            planning_latency_seconds=planning_run.llm_latency_seconds,
            intention_usage=(
                intention_run.llm_usage.model_dump(mode="json")
                if intention_run.llm_usage
                else None
            ),
            planning_usage=(
                planning_run.llm_usage.model_dump(mode="json")
                if planning_run.llm_usage
                else None
            ),
            resolved_checks_by_category=checks_by_category,
        )
    except Exception as error:
        (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return E1RunResult(
            **base,
            completed_at=datetime.now(UTC).isoformat(),
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error)[:2_000],
        )


def summarize_e1(*, config_path: Path, repository_root: Path) -> dict[str, Any]:
    """Derive CSV, summary, reports, and heatmap from immutable run results."""

    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = load_e1_config(config_path)
    experiment_dir = config_path.parent
    results: list[E1RunResult] = []
    for dataset in config.datasets:
        for repetition in range(1, config.repetitions + 1):
            path = experiment_dir / "runs" / dataset.dataset_slug / f"r{repetition}/result.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            results.append(E1RunResult.model_validate_json(path.read_text(encoding="utf-8")))
    _write_results_csv(experiment_dir / "results.csv", results)

    usage_items = []
    for result in results:
        for payload in (result.intention_usage, result.planning_usage):
            if payload:
                from backend.llm import LLMUsageSummary

                usage_items.append(LLMUsageSummary.model_validate(payload))
    usage = aggregate_llm_usage(usage_items)
    dataset_summaries = []
    for dataset in config.datasets:
        rows = [item for item in results if item.dataset_slug == dataset.dataset_slug]
        dataset_summaries.append(
            {
                "dataset_slug": dataset.dataset_slug,
                "display_name": dataset.display_name,
                "role": dataset.role,
                "successful_runs": sum(item.run_success for item in rows),
                "total_runs": len(rows),
                "mean_mandatory_check_recall": statistics.fmean(
                    item.mandatory_check_recall for item in rows
                ),
                "planning_complete_runs": sum(item.planning_complete for item in rows),
                "unresolved_gap_runs": sum(item.unresolved_gap_count > 0 for item in rows),
                "failures": [
                    {
                        "repetition": item.repetition,
                        "type": item.failure_type,
                        "message": item.failure_message,
                    }
                    for item in rows
                    if not item.run_success
                ],
            }
        )
    summary = {
        "schema_version": "e1_summary.v1",
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "runs": len(results),
        "successful_runs": sum(item.run_success for item in results),
        "development_success_rate": _success_rate(results, "development"),
        "held_out_success_rate": _success_rate(results, "held_out"),
        "datasets": dataset_summaries,
        "llm_usage": usage.model_dump(mode="json") if usage else None,
        "executable_suite_ready_note": (
            "Always false before a generated candidate, physical mappings, and frozen oracle are bound. "
            "E1 evaluates planning_complete instead."
        ),
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix = _coverage_matrix(config, results)
    (experiment_dir / "coverage_matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _draw_heatmap(experiment_dir, matrix)
    (experiment_dir / "REPORT.md").write_text(
        _report_text(config, summary, matrix), encoding="utf-8"
    )
    _append_journal(experiment_dir, "summarized", {"successful_runs": summary["successful_runs"]})
    return summary


def verify_e1_freeze(path: Path, repository_root: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = repository_root.resolve()
    changed = []
    for item in payload["files"]:
        file_path = _repo_path(item["path"], root)
        if not file_path.is_file() or file_hash(file_path) != item["sha256"]:
            changed.append(item["path"])
    if changed:
        raise ValueError(f"Frozen E1 inputs changed: {changed}")


def _requirements(dataset: E1DatasetConfig) -> PipelineRequirements:
    return _REQUIREMENTS_GRID if dataset.data_class == "regular_rectilinear_grid" else _REQUIREMENTS_STATION


def _profile(dataset: E1DatasetConfig, contract: DatasetContract) -> EvaluationProfile:
    family = contract.dataset_family or "scientific-data"
    return EvaluationProfile(
        profile_id=f"{dataset.dataset_slug}-{dataset.output_format}-v{dataset.output_format_version}",
        data_class=dataset.data_class,
        output_format=dataset.output_format,
        output_format_version=dataset.output_format_version,
        workload_kind=dataset.workload_kind,
        engineering_profile="general_pipeline",
        dataset_tags=[
            f"dataset:{_tag(dataset.dataset_slug)}",
            "domain:earth-science",
            f"family:{_tag(family)}",
        ],
    )


def _interaction(reference: E1Reference, dataset: E1DatasetConfig) -> list[ConversationTurn]:
    contract = reference.expected_contract
    selected = {
        "dataset_slug": contract.dataset_slug,
        "title": contract.title,
        "source_url": str(contract.source_url),
        "provider": contract.provider,
        "dataset_family": contract.dataset_family,
        "fields": [field.model_dump(mode="json") for field in contract.fields],
        "scope": contract.scope,
        "advanced_options": contract.advanced_options,
        "credential_references": reference.expected_credential_references,
        "required_output": {
            "format": dataset.output_format,
            "version": dataset.output_format_version,
        },
    }
    return [
        ConversationTurn(
            role="user",
            content=(
                f"Prepare an ML-ready pipeline for {dataset.display_name}. "
                "Use exactly this selected data and scope:\n"
                + json.dumps(selected, indent=2, sort_keys=True)
            ),
        ),
        ConversationTurn(
            role="system",
            content=(
                "Confirm the downstream access pattern and the relative importance "
                "of runtime, storage, and engineering objectives."
            ),
        ),
        ConversationTurn(
            role="user",
            content=(
                "The data will be read repeatedly during ML training. Correctness is "
                "mandatory. First maximise consumer throughput; second minimise native "
                "output size. Still report materialisation time and engineering quality "
                "for analysis, but do not use them to override those priorities."
            ),
        ),
    ]


def _clarification_answers(
    *,
    questions: list[str],
    reference: E1Reference,
    dataset: E1DatasetConfig,
) -> list[ConversationTurn]:
    """Answer only from the frozen task instead of allowing model invention."""

    contract = reference.expected_contract
    answer = {
        "questions_being_answered": questions,
        "authoritative_selected_fields": [
            field.model_dump(mode="json") for field in contract.fields
        ],
        "authoritative_scope": contract.scope,
        "authoritative_advanced_options": contract.advanced_options,
        "credential_references": reference.expected_credential_references,
        "target": reference.expected_target,
        "instructions": (
            "The frozen task is complete. Preserve native values and units. Do not add "
            "derived scope aliases, aggregation, unit conversion, feature engineering, "
            "or physical layout requirements. Credential references are sufficient; "
            "credential values are not requested or available."
        ),
    }
    return [
        ConversationTurn(
            role="system",
            content="Frozen-task clarification:\n" + json.dumps(answer, indent=2, sort_keys=True),
        )
    ]


def _field_ids(contract: DatasetContract) -> list[str]:
    return sorted(
        logical_field_id(
            field.name,
            [selector.model_dump(mode="json") for selector in field.selectors],
        )
        for field in contract.fields
    )


def _mapping_valid(draft: Any, data_class: str) -> bool:
    if data_class == "regular_rectilinear_grid":
        return {axis.role for axis in draft.axes} == {"sample", "y", "x"} and draft.station_mapping is None
    return not draft.axes and draft.station_mapping is not None


def _checks_by_category(checks: list[Any]) -> dict[str, int]:
    labels = {
        "core": "Core",
        "data_class": "Data Structure",
        "output_format": "Output Format",
        "workload": "Consumer Workload",
        "objective": "Objectives",
        "engineering": "Engineering Quality",
        "robustness": "Robustness",
    }
    result: dict[str, int] = {}
    for check in checks:
        label = labels[check.layer]
        result[label] = result.get(label, 0) + 1
    result["Dataset Mapping/Oracle"] = 1
    return result


def _credential_references(*, fixture_path: Path, access_context_path: Path | None) -> list[str]:
    names: set[str] = set()
    manifest = json.loads(fixture_path.read_text(encoding="utf-8"))
    for entry in manifest.get("entries", []):
        value = entry.get("credential_environment_variable")
        if isinstance(value, str) and value:
            names.add(value)
    if access_context_path and access_context_path.is_file():
        context = json.loads(access_context_path.read_text(encoding="utf-8"))
        for item in context.get("credential_env_vars", []):
            value = item.get("env_var")
            if isinstance(value, str) and value:
                names.add(value)
    return sorted(names)


def _freeze_paths(*, config_path: Path, config: E1Config, root: Path) -> set[Path]:
    paths = {
        config_path,
        root / f"backend/agents/intention_space/prompts/{config.intention_prompt}_system.md",
        root / f"backend/agents/intention_space/prompts/{config.intention_prompt}_user.md",
        root / f"backend/agents/evaluation_planning/prompts/{config.evaluation_prompt}_system.md",
        root / f"backend/agents/evaluation_planning/prompts/{config.evaluation_prompt}_user.md",
        root / "backend/agents/intention_space/schemas.py",
        root / "backend/agents/intention_space/agent.py",
        root / "backend/agents/intention_space/workflow.py",
        root / "backend/agents/evaluation_planning/schemas.py",
        root / "backend/agents/evaluation_planning/agent.py",
        root / "backend/evaluation/planning.py",
        root / "backend/evaluation/check_library.py",
        root / "backend/evaluation/check_promotion.py",
        root / "backend/experiments/e1_specification.py",
    }
    experiment_dir = config_path.parent
    paths.update((experiment_dir / "references").glob("*/expected.json"))
    for dataset in config.datasets:
        for value in (
            dataset.inventory_path,
            dataset.source_contract_lock_path,
            dataset.source_fixture_manifest_path,
        ):
            paths.add(_repo_path(value, root))
        if dataset.candidate_path:
            paths.add(_repo_path(dataset.candidate_path, root))
        if dataset.access_context_path:
            paths.add(_repo_path(dataset.access_context_path, root))
    return {path.resolve() for path in paths}


def _coverage_matrix(config: E1Config, results: list[E1RunResult]) -> dict[str, Any]:
    columns = [
        "Core",
        "Data Structure",
        "Output Format",
        "Consumer Workload",
        "Objectives",
        "Engineering Quality",
        "Dataset Mapping/Oracle",
    ]
    rows = []
    for dataset in config.datasets:
        observed = [item for item in results if item.dataset_slug == dataset.dataset_slug]
        all_success = all(item.run_success for item in observed)
        representative = next((item for item in observed if item.resolved_checks_by_category), None)
        cells = {}
        for column in columns:
            count = representative.resolved_checks_by_category.get(column, 0) if representative else 0
            cells[column] = {
                "state": "resolved" if all_success and count > 0 else "invalid",
                "resolved_check_count": count,
            }
        rows.append(
            {
                "dataset_slug": dataset.dataset_slug,
                "display_name": dataset.display_name,
                "role": dataset.role,
                "cells": cells,
            }
        )
    return {"schema_version": "evaluation_coverage_matrix.v1", "columns": columns, "rows": rows}


def _draw_heatmap(experiment_dir: Path, matrix: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    states = {"invalid": 0, "gap": 1, "not_applicable": 2, "resolved": 3}
    colors = ["#c9473d", "#e7a339", "#d6d9dd", "#2f7d61"]
    rows = matrix["rows"]
    columns = matrix["columns"]
    values = [[states[row["cells"][column]["state"]] for column in columns] for row in rows]
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.imshow(values, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(range(len(columns)), labels=columns, rotation=28, ha="right")
    axis.set_yticks(range(len(rows)), labels=[row["display_name"] for row in rows])
    for y, row in enumerate(rows):
        for x, column in enumerate(columns):
            count = row["cells"][column]["resolved_check_count"]
            axis.text(x, y, str(count), ha="center", va="center", color="white" if values[y][x] in {0, 3} else "black", fontsize=9)
    axis.set_title("Evaluation coverage across frozen Earth science tasks")
    axis.set_xlabel("Evaluation check category")
    axis.set_ylabel("Dataset")
    axis.legend(
        handles=[
            Patch(color=colors[3], label="Resolved and validated"),
            Patch(color=colors[2], label="Not applicable"),
            Patch(color=colors[1], label="Unresolved gap"),
            Patch(color=colors[0], label="Invalid or failed"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    figures = experiment_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "evaluation_coverage_heatmap.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / "evaluation_coverage_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_results_csv(path: Path, results: list[E1RunResult]) -> None:
    fields = [
        "dataset_slug", "display_name", "role", "repetition", "status", "run_success",
        "field_selector_exact", "scope_exact", "objective_priority_exact", "workload_exact",
        "credential_policy_exact", "target_exact", "logical_mapping_valid", "oracle_strategy_valid",
        "mandatory_check_recall", "unauthorized_check_count", "unnecessary_proposal_count",
        "planning_complete", "intention_latency_seconds", "planning_latency_seconds",
        "failure_type", "failure_message",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            payload = result.model_dump(mode="json")
            writer.writerow({field: payload.get(field) for field in fields})


def _success_rate(results: list[E1RunResult], role: str) -> float:
    selected = [item for item in results if item.role == role]
    return sum(item.run_success for item in selected) / len(selected) if selected else 0.0


def _protocol_text(config: E1Config) -> str:
    return f"""# {config.experiment_id}

- Model: `{config.model}`
- Repetitions: {config.repetitions} per dataset
- Development datasets: ERA5 and OpenAQ
- Held-out datasets: CAMS EAC4, OISST, GHCN-Daily, and USGS Daily Values
- Provider network: disabled; immutable local fixtures only
- LLM retries: no semantic retries; shared transport behavior only
- Ordered objectives: maximize consumer throughput, then minimize output bytes
- Descriptive measurements: materialization time and engineering quality

References and implementation were frozen before final development and held-out calls. Held-out outputs are append-only and must not be used to revise this experiment identifier.
"""


def _report_text(config: E1Config, summary: Mapping[str, Any], matrix: Mapping[str, Any]) -> str:
    lines = [
        "# E1 Specification Reliability and Evaluation Coverage",
        "",
        f"Model: `{config.model}`. Runs: {summary['runs']}. Successful first-attempt runs: {summary['successful_runs']}.",
        "",
        "| Dataset | Role | Successful runs | Mandatory-check recall |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in summary["datasets"]:
        lines.append(
            f"| {item['display_name']} | {item['role']} | {item['successful_runs']}/{item['total_runs']} | {item['mean_mandatory_check_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The planning compiler, not the LLM, selects mandatory checks. `planning_complete` indicates that the pre-generation semantic plan is complete. `executable_suite_ready` remains false until a generated candidate, physical mappings, and evaluator-owned oracle are bound.",
            "",
            "![Evaluation coverage](figures/evaluation_coverage_heatmap.png)",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_journal(experiment_dir: Path, event: str, details: Mapping[str, Any]) -> None:
    record = {"at": datetime.now(UTC).isoformat(), "event": event, **details}
    with (experiment_dir / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _repo_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())
