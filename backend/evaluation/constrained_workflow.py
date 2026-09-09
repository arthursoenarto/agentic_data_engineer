"""File workflow for immutable directories and atomic evaluation reports."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.evaluation.constrained import run_constrained_evaluation
from backend.evaluation.constrained_schemas import (
    ConstrainedEvaluationConfig,
    ConstrainedEvaluationRun,
)
from backend.evaluation.markdown_report import render_evaluation_report


def run_constrained_evaluation_file(
    config_path: Path,
    *,
    repository_root: Path,
    output_dir: Path | None = None,
) -> tuple[ConstrainedEvaluationRun, Path]:
    """Load a trusted suite and write into one exclusively created run directory."""

    root = repository_root.resolve()
    resolved_config = _resolve(config_path, root)
    config = ConstrainedEvaluationConfig.model_validate_json(
        resolved_config.read_text(encoding="utf-8")
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default_parent = (
        resolved_config.parent.parent / "runs" / "evaluations"
        if resolved_config.parent.name == "benchmarks"
        else resolved_config.parent / "runs"
    )
    final = (
        _resolve(output_dir, root)
        if output_dir is not None
        else default_parent / f"{config.suite_id}_{timestamp}_{uuid.uuid4().hex[:8]}"
    )
    if final.exists():
        raise FileExistsError(f"Evaluation output already exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    final.mkdir(parents=False, exist_ok=False)
    run = run_constrained_evaluation(
        config,
        repository_root=root,
        config_file=resolved_config,
        run_dir=final,
    )
    _write_new(final / "evaluation.json", run.model_dump_json(indent=2) + "\n")
    _write_new(final / "report.md", render_evaluation_report(run))
    _write_new(
        final / "suite.json",
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n",
    )
    return run, final / "evaluation.json"


def _resolve(path: Path, repository_root: Path) -> Path:
    expanded = path.expanduser()
    resolved = (
        expanded.resolve()
        if expanded.is_absolute()
        else (repository_root / expanded).resolve()
    )
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Evaluation path escapes the repository: {resolved}"
        ) from error
    return resolved


def _write_new(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
