"""Immutable file workflow and typed failure persistence for v3 evaluation."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.env import load_env
from backend.evaluation.atomic_report import write_new_atomic
from backend.evaluation.objective_v3 import (
    V3EvaluationPhaseError,
    run_constrained_evaluation_v3,
)
from backend.evaluation.markdown_report import render_evaluation_report
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationConfigV3,
    ConstrainedEvaluationRunV3,
    EvaluationFailureV3,
    TerraioExtensionEvaluationConfigV3,
    TerraioExtensionEvaluationRunV3,
)
from backend.evaluation.terraio_extension_evaluation import (
    run_terraio_extension_evaluation_v3,
)
from backend.llm import LLMClient


V3FileResult = (
    ConstrainedEvaluationRunV3 | TerraioExtensionEvaluationRunV3 | EvaluationFailureV3
)


def run_constrained_evaluation_v3_file(
    config_path: Path,
    *,
    repository_root: Path,
    output_dir: Path | None = None,
    llm_client: LLMClient | None = None,
    llm_timeout_seconds: int = 60,
) -> tuple[V3FileResult, Path]:
    """Backward-compatible named entrypoint for the regular-grid v3 suite."""

    return run_evaluation_v3_file(
        config_path,
        repository_root=repository_root,
        output_dir=output_dir,
        llm_client=llm_client,
        llm_timeout_seconds=llm_timeout_seconds,
    )


def run_evaluation_v3_file(
    config_path: Path,
    *,
    repository_root: Path,
    output_dir: Path | None = None,
    llm_client: LLMClient | None = None,
    llm_timeout_seconds: int = 60,
) -> tuple[V3FileResult, Path]:
    """Persist a typed terminal report for success, setup error, or partial run."""

    if llm_timeout_seconds < 1:
        raise ValueError("llm_timeout_seconds must be positive")

    started = time.perf_counter()
    root = repository_root.resolve()
    resolved_config = _resolve(config_path, root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default_parent = (
        resolved_config.parent.parent / "runs" / "evaluations"
        if resolved_config.parent.name == "benchmarks"
        else resolved_config.parent / "runs"
    )
    final = (
        _resolve(output_dir, root)
        if output_dir is not None
        else default_parent
        / f"{resolved_config.stem}_{timestamp}_{uuid.uuid4().hex[:8]}"
    )
    if final.exists():
        raise FileExistsError(f"Evaluation output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=False, exist_ok=False)
    phase = "setup"
    partial: dict[str, object] | None = None
    try:
        raw = json.loads(resolved_config.read_text(encoding="utf-8"))
        schema_version = raw.get("schema_version")
        if schema_version == "evaluation_constrained.regular_grid_zarr.v3":
            config = ConstrainedEvaluationConfigV3.model_validate(raw)
        elif schema_version == "evaluation_constrained.terraio_extension.v3":
            config = TerraioExtensionEvaluationConfigV3.model_validate(raw)
        else:
            raise ValueError(f"Unsupported v3 evaluation schema: {schema_version!r}")
        load_env()
        write_new_atomic(
            final / "suite.json",
            json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
        )
        client = llm_client
        if config.engineering_quality.enabled and client is None:
            try:
                client = LLMClient(
                    model=config.engineering_quality.model,
                    timeout_seconds=llm_timeout_seconds,
                )
            except RuntimeError:
                client = None
        phase = "deterministic"
        if isinstance(config, ConstrainedEvaluationConfigV3):
            result = run_constrained_evaluation_v3(
                config,
                repository_root=root,
                config_file=resolved_config,
                run_dir=final,
                llm_client=client,
            )
        else:
            result = run_terraio_extension_evaluation_v3(
                config,
                repository_root=root,
                config_file=resolved_config,
                run_dir=final,
                llm_client=client,
            )
        phase = "validation"
        partial = result.model_dump(mode="json")
        result_path = final / "evaluation.json"
        write_new_atomic(result_path, result.model_dump_json(indent=2) + "\n")
        write_new_atomic(final / "report.md", render_evaluation_report(result))
        return result, result_path
    except Exception as error:
        original = (
            error.original if isinstance(error, V3EvaluationPhaseError) else error
        )
        if isinstance(error, V3EvaluationPhaseError):
            phase = error.phase
            partial = error.partial
        status = "timeout" if isinstance(original, TimeoutError) else "error"
        failure = EvaluationFailureV3(
            run_id=final.name,
            phase=phase,  # type: ignore[arg-type]
            status=status,
            config_file=_display(resolved_config, root),
            error_type=type(original).__name__,
            error=str(original),
            partial_report=partial,
            duration_seconds=time.perf_counter() - started,
        )
        failure_path = final / "evaluation_failure.json"
        write_new_atomic(failure_path, failure.model_dump_json(indent=2) + "\n")
        write_new_atomic(final / "report.md", render_evaluation_report(failure))
        return failure, failure_path


def _resolve(path: Path, repository_root: Path) -> Path:
    expanded = path.expanduser()
    resolved = (
        expanded.resolve()
        if expanded.is_absolute()
        else (repository_root / expanded).resolve()
    )
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"Evaluation path escapes the repository: {resolved}"
        ) from error
    return resolved


def _display(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())
