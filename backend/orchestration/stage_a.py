"""One-command Stage A generation-to-evaluation workflow."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import venv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.agents.etl_pipeline import (
    PipelineGenerationAgent,
    PipelineConditionName,
    PipelinePolicy,
    GenerationPromptOverride,
    accept_family_pipeline,
    execute_family_pipeline,
    resolve_pipeline_command,
)
from backend.agents.pipeline_repair.schemas import PipelineRepairLog
from backend.contracts import load_dataset_contract
from backend.env import ENV_FILE
from backend.evaluation.objective_v3_schemas import (
    ConstrainedEvaluationRunV3,
    EvaluationFailureV3,
)
from backend.evaluation.objective_v3_workflow import run_evaluation_v3_file
from backend.evaluation.station_parquet import run_station_parquet_evaluation
from backend.evaluation.station_parquet_schemas import (
    StationParquetEvaluationConfig,
    StationParquetEvaluationRun,
)
from backend.llm import LLMClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REQUIREMENTS = REPOSITORY_ROOT / "backend" / "requirements-benchmark.txt"
FRAMEWORK_LINEAGE_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
PROTECTED_FAMILY_PATHS = {
    "manifest.json",
    "pipeline_contract.json",
    "run_pipeline.py",
}


@dataclass(frozen=True)
class _RuntimePreparation:
    python: Path
    freeze_path: Path | None
    requirements_sha256: str
    dependency_source: Literal["candidate_plus_benchmark", "benchmark_fallback"]
    repair_log: PipelineRepairLog
    repair_log_path: Path


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageAWorkflowConfig(_StrictModel):
    """Versioned inputs for one immutable generation-to-evaluation trial."""

    schema_version: Literal["stage_a_workflow_config.v1"] = "stage_a_workflow_config.v1"
    workflow_id: str
    dataset_dir: Path
    seed_contract: Path
    condition: PipelineConditionName
    pipeline_id: str
    prompt_lock: Path | None = None
    generation_prompt_override: GenerationPromptOverride | None = None
    inventory: Path | None = None
    candidate_cache: Path | None = None
    alternate_contract: Path | None = None
    evaluation_profile: Path | None = None
    evaluation_plan: Path | None = None
    pipeline_policy: PipelinePolicy | None = None
    primary_suite_builder: Path
    primary_suite_spec: Path | None = None
    primary_suite_dir: Path
    extensibility_suite_builder: Path | None = None
    linked_suite_dir: Path | None = None
    alternate_suite_dir: Path | None = None
    alternate_inputs_dir: Path | None = None
    evaluation_output_dir: Path
    max_repair_attempts: int = Field(default=3, ge=1, le=3)
    command_timeout_seconds: int = Field(default=3600, ge=1)
    llm_timeout_seconds: int = Field(default=600, ge=1)
    repair_prompt_name: str = Field(default="default", pattern=r"^[a-z0-9_]+$")
    completion_requirement: Literal["complete_objective", "hard_gates"] = (
        "complete_objective"
    )
    env_file: Path = ENV_FILE

    @model_validator(mode="after")
    def evaluation_inputs_are_unambiguous(self) -> "StageAWorkflowConfig":
        if self.evaluation_profile is not None and self.evaluation_plan is not None:
            raise ValueError("Stage A accepts one evaluation profile or plan, not both")
        if self.prompt_lock is not None and self.generation_prompt_override is not None:
            raise ValueError("Stage A accepts a prompt lock or prompt override, not both")
        return self


class StageAPhaseRecord(_StrictModel):
    """One durable phase outcome in a Stage A workflow."""

    name: str
    status: Literal["passed", "failed", "blocked", "skipped", "diagnostic_failed"]
    artifact: str | None = None
    detail: str | None = None


class StageAWorkflowReport(_StrictModel):
    """Compact immutable evidence for a full Stage A attempt."""

    schema_version: Literal["stage_a_workflow_run.v2"] = "stage_a_workflow_run.v2"
    workflow_id: str
    pipeline_id: str
    condition: PipelineConditionName
    generation_prompt_id: str | None = None
    started_at: str
    completed_at: str | None = None
    status: Literal["running", "passed", "failed", "blocked"] = "running"
    phases: list[StageAPhaseRecord] = Field(default_factory=list)
    repair_attempts_used: int = Field(default=0, ge=0, le=3)
    repair_logs: list[str] = Field(default_factory=list)
    acceptance_report: str | None = None
    evaluation_report: str | None = None
    objective_vector_complete: bool = False
    optimization_ready: bool = False
    error: str | None = None
    runtime_python: str | None = None
    runtime_requirements_sha256: str | None = None
    runtime_freeze: str | None = None
    runtime_dependency_source: Literal[
        "candidate_plus_benchmark", "candidate_requirements", "benchmark_fallback"
    ] | None = None
    workflow_config_snapshot: str
    workflow_config_sha256: str
    framework_source_sha256: str
    framework_source_unchanged: bool | None = None
    input_files: dict[str, str] = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    outcome_sha256: dict[str, str] = Field(default_factory=dict)
    lineage_unchanged: bool | None = None


def run_stage_a_workflow(
    config: StageAWorkflowConfig,
    *,
    generation_agent: PipelineGenerationAgent,
) -> tuple[StageAWorkflowReport, Path]:
    """Run generation, bounded repair, acceptance, transfer, and evaluation."""

    dataset_dir = config.dataset_dir.resolve()
    inventory = (config.inventory or dataset_dir / "dataset_inventory.json").resolve()
    cache = (
        config.candidate_cache
        or dataset_dir / "data" / "cache" / "stage_a" / config.pipeline_id
    ).resolve()
    run_dir = dataset_dir / "runs" / "stage_a" / config.workflow_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "stage_a_run.json"
    config_snapshot = run_dir / "workflow_config.json"
    config_content = json.dumps(
        config.model_dump(mode="json"), indent=2, sort_keys=True
    ) + "\n"
    config_snapshot.write_text(config_content, encoding="utf-8")
    input_files = _lineage_input_files(config, inventory=inventory)
    report = StageAWorkflowReport(
        workflow_id=config.workflow_id,
        pipeline_id=config.pipeline_id,
        condition=config.condition,
        generation_prompt_id=(
            config.generation_prompt_override.prompt_id
            if config.generation_prompt_override is not None
            else None
        ),
        started_at=_now(),
        workflow_config_snapshot=_display(config_snapshot),
        workflow_config_sha256=_sha256_file(config_snapshot),
        framework_source_sha256=_framework_source_sha256(),
        input_files={name: _display(path) for name, path in input_files.items()},
        input_sha256={name: _sha256_file(path) for name, path in input_files.items()},
    )
    _write_report(report_path, report)

    try:
        inventory_payload = json.loads(inventory.read_text(encoding="utf-8"))
        from backend.agents.dataset_inventory.schemas import DatasetInventory

        inventory_model = DatasetInventory.model_validate(inventory_payload)
        prompt_lock = None
        if config.prompt_lock is not None:
            from backend.agents.etl_pipeline import load_and_validate_pipeline_prompt_lock

            prompt_lock = load_and_validate_pipeline_prompt_lock(
                config.prompt_lock.resolve()
            )
        build = generation_agent.generate(
            condition_name=config.condition,
            seed_contract=load_dataset_contract(config.seed_contract.resolve()),
            inventory=inventory_model,
            dataset_dir=dataset_dir,
            pipeline_id=config.pipeline_id,
            prompt_lock=prompt_lock,
            prompt_override=config.generation_prompt_override,
            policy=config.pipeline_policy
            or PipelinePolicy(
                provider=inventory_model.provider,
                dataset_id=inventory_model.dataset_id,
                publication_format="zarr",
                data_model="xarray_dataset",
            ),
        )
        _phase(report, "generation", "passed", artifact=_display(build.pipeline_dir))
        report.outcome_sha256["pipeline_manifest"] = _sha256_file(
            build.pipeline_dir / "manifest.json"
        )
        _write_report(report_path, report)

        remaining = config.max_repair_attempts
        runtime = _prepare_runtime(
            build.pipeline_dir,
            workflow_id=config.workflow_id,
            run_dir=run_dir,
            timeout_seconds=config.command_timeout_seconds,
            pipeline_agent=generation_agent,
            max_repair_attempts=remaining,
            repair_prompt_name=config.repair_prompt_name,
            env_path=config.env_file,
        )
        runtime_repairs = len(runtime.repair_log.attempts)
        report.repair_attempts_used += runtime_repairs
        report.repair_logs.append(_display(runtime.repair_log_path))
        remaining -= runtime_repairs
        report.runtime_python = _display(runtime.python)
        report.runtime_requirements_sha256 = runtime.requirements_sha256
        report.runtime_dependency_source = runtime.dependency_source
        if runtime.freeze_path is None:
            final_execution = runtime.repair_log.initial_execution
            if runtime.repair_log.attempts:
                final_attempt = runtime.repair_log.attempts[-1]
                if final_attempt.execution_after is not None:
                    final_execution = final_attempt.execution_after
            status = (
                "blocked"
                if runtime.repair_log.final_status == "non_candidate_failure"
                else "failed"
            )
            _phase(
                report,
                "runtime",
                status,
                artifact=_display(runtime.repair_log_path),
                detail=final_execution.failure_code,
            )
            return _finish(report_path, report, status)
        runtime_python = runtime.python
        report.runtime_freeze = _display(runtime.freeze_path)
        _phase(
            report,
            "runtime",
            "passed",
            artifact=_display(runtime.repair_log_path),
            detail=(
                f"requirements_sha256={runtime.requirements_sha256}; "
                f"repairs={runtime_repairs}; freeze={report.runtime_freeze}"
            ),
        )
        _write_report(report_path, report)

        has_generated_tests = _has_generated_tests(build.pipeline_dir)
        if has_generated_tests:
            _phase(
                report,
                "generated_tests",
                "skipped",
                detail=(
                    "Deferred to generation acceptance as a non-blocking diagnostic; "
                    "candidate-authored tests do not consume the repair budget."
                ),
            )
        else:
            _phase(
                report,
                "generated_tests",
                "skipped",
                detail="Candidate supplied no pytest-discoverable tests.",
            )

        cache.mkdir(parents=True, exist_ok=True)
        preparation_root = run_dir / "primary_preparation"
        command = resolve_pipeline_command(
            build.pipeline_contract,
            contract_lock=config.seed_contract.resolve(),
            inventory=inventory,
            cache_dir=cache,
            output_dir=preparation_root / "output",
            run_receipt=preparation_root / "pipeline_run.json",
            python_executable=runtime_python,
        )
        live_log, live_log_path = generation_agent.repair(
            pipeline_dir=build.pipeline_dir,
            command=command,
            prompt_name=config.repair_prompt_name,
            max_attempts=remaining,
            timeout_seconds=config.command_timeout_seconds,
            env_path=config.env_file,
            protected_paths=PROTECTED_FAMILY_PATHS,
            repair_root=run_dir / "repairs" / "primary_execution",
        )
        used = len(live_log.attempts)
        report.repair_attempts_used += used
        report.repair_logs.append(_display(live_log_path))
        if live_log.final_status not in {"already_succeeded", "repaired"}:
            status = (
                "blocked"
                if live_log.final_status == "non_candidate_failure"
                else "failed"
            )
            _phase(
                report,
                "primary_preparation",
                status,
                artifact=_display(live_log_path),
                detail=live_log.initial_execution.failure_code,
            )
            return _finish(report_path, report, status)
        _phase(
            report,
            "primary_preparation",
            "passed",
            artifact=_display(live_log_path),
        )
        _write_report(report_path, report)

        origin = "repaired" if report.repair_attempts_used else "generated"
        acceptance, acceptance_path = accept_family_pipeline(
            dataset_dir=dataset_dir,
            pipeline_id=config.pipeline_id,
            contract_lock_path=config.seed_contract.resolve(),
            prepared_cache_dir=cache,
            inventory_path=inventory,
            candidate_origin=origin,
            repair_reference=(
                _display(report_path) if origin == "repaired" else None
            ),
            timeout_seconds=config.command_timeout_seconds,
            env_path=config.env_file,
            run_generated_tests=has_generated_tests,
            python_executable=runtime_python,
        )
        report.acceptance_report = _display(acceptance_path)
        if acceptance.final_status != "passed":
            _phase(
                report,
                "acceptance",
                "failed",
                artifact=report.acceptance_report,
            )
            return _finish(report_path, report, "failed")
        _phase(
            report,
            "acceptance",
            "passed",
            artifact=report.acceptance_report,
        )
        _write_report(report_path, report)

        primary_suite = _build_primary_suite(
            config,
            config.pipeline_id,
            run_dir,
            python_executable=runtime_python,
        )
        evaluation_suite = primary_suite
        _phase(report, "primary_suite", "passed", artifact=_display(primary_suite))

        if config.alternate_contract is not None:
            alternate = None
            try:
                alternate = execute_family_pipeline(
                    dataset_dir=dataset_dir,
                    pipeline_id=config.pipeline_id,
                    contract_lock_path=config.alternate_contract.resolve(),
                    inventory_path=inventory,
                    cache_dir=cache,
                    timeout_seconds=config.command_timeout_seconds,
                    env_path=config.env_file,
                    repair_run_reference=(
                        _display(report_path) if origin == "repaired" else None
                    ),
                    python_executable=runtime_python,
                )
            except Exception as error:  # noqa: BLE001 - retain primary-vector eligibility.
                _phase(
                    report,
                    "alternate_contract",
                    "diagnostic_failed",
                    detail=(
                        "Non-blocking extensibility execution error: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            if alternate is not None and not alternate.succeeded:
                _phase(
                    report,
                    "alternate_contract",
                    "diagnostic_failed",
                    artifact=_display(alternate.paths.receipt),
                    detail=(
                        "The unchanged pipeline did not execute the alternate contract; "
                        "the primary suite remains eligible for F(p)."
                    ),
                )
            elif alternate is not None:
                _phase(
                    report,
                    "alternate_contract",
                    "passed",
                    artifact=_display(alternate.paths.receipt),
                )
                try:
                    evaluation_suite = _build_extensibility_suite(
                        config,
                        alternate.paths.receipt,
                        run_dir,
                    )
                except Exception as error:  # noqa: BLE001 - primary evaluation remains valid.
                    _phase(
                        report,
                        "extensibility_suite",
                        "diagnostic_failed",
                        detail=(
                            "Non-blocking extensibility-suite error: "
                            f"{type(error).__name__}: {error}"
                        ),
                    )
                    evaluation_suite = primary_suite
                else:
                    _phase(
                        report,
                        "extensibility_suite",
                        "passed",
                        artifact=_display(evaluation_suite),
                    )
            _write_report(report_path, report)

        evaluation, evaluation_path = _run_stage_a_evaluation(
            evaluation_suite,
            output_dir=config.evaluation_output_dir.resolve(),
            llm_timeout_seconds=config.llm_timeout_seconds,
        )
        report.evaluation_report = _display(evaluation_path)
        if isinstance(evaluation, EvaluationFailureV3):
            _phase(
                report,
                "evaluation",
                "failed",
                artifact=report.evaluation_report,
                detail=evaluation.error,
            )
            return _finish(report_path, report, "failed")
        hard_gates_passed = False
        if isinstance(evaluation, ConstrainedEvaluationRunV3):
            report.objective_vector_complete = evaluation.summary.objective_vector_complete
            report.optimization_ready = evaluation.summary.optimization_ready
            hard_gates_passed = evaluation.summary.feasible
        elif isinstance(evaluation, StationParquetEvaluationRun):
            report.objective_vector_complete = evaluation.optimization_ready
            report.optimization_ready = evaluation.optimization_ready
            hard_gates_passed = evaluation.feasible
        else:
            raise TypeError("Stage A received an unsupported evaluation result.")
        complete = (
            hard_gates_passed
            if config.completion_requirement == "hard_gates"
            else report.objective_vector_complete
        )
        status: Literal["passed", "failed"] = (
            "passed" if complete else "failed"
        )
        _phase(
            report,
            "evaluation",
            status,
            artifact=report.evaluation_report,
            detail=(
                "All required hard gates passed."
                if complete and config.completion_requirement == "hard_gates"
                else (
                    "Complete F(p)."
                    if report.objective_vector_complete
                    else "Required evaluation state is incomplete."
                )
            ),
        )
        return _finish(report_path, report, status)
    except Exception as error:  # noqa: BLE001 - persist terminal workflow evidence.
        report.error = f"{type(error).__name__}: {error}"
        _phase(report, "workflow", "failed", detail=report.error)
        return _finish(report_path, report, "failed")


def _build_primary_suite(
    config: StageAWorkflowConfig,
    pipeline_id: str,
    run_dir: Path,
    *,
    python_executable: Path,
) -> Path:
    command = [
        sys.executable,
        str(config.primary_suite_builder.resolve()),
        "--accepted-candidate",
        pipeline_id,
        "--output-dir",
        str(config.primary_suite_dir.resolve()),
        "--python-executable",
        str(python_executable.absolute()),
    ]
    if config.primary_suite_spec is not None:
        command.extend(["--spec", str(config.primary_suite_spec.resolve())])
    if config.evaluation_profile is not None:
        command.extend(
            ["--evaluation-profile", str(config.evaluation_profile.resolve())]
        )
    if config.evaluation_plan is not None:
        command.extend(["--evaluation-plan", str(config.evaluation_plan.resolve())])
    _run_builder(
        command,
        log_dir=run_dir / "suite_builder",
    )
    path = config.primary_suite_dir.resolve() / f"{pipeline_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Primary suite builder did not create {path}")
    return path


def _run_stage_a_evaluation(
    suite: Path,
    *,
    output_dir: Path,
    llm_timeout_seconds: int,
) -> tuple[
    ConstrainedEvaluationRunV3 | StationParquetEvaluationRun | EvaluationFailureV3,
    Path,
]:
    """Dispatch one accepted suite without weakening either evaluator boundary."""

    payload = json.loads(suite.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "evaluation_constrained.station_parquet.v1":
        return run_evaluation_v3_file(
            suite,
            repository_root=REPOSITORY_ROOT,
            output_dir=output_dir,
            llm_timeout_seconds=llm_timeout_seconds,
        )

    station_config = StationParquetEvaluationConfig.model_validate(payload)
    client = None
    if station_config.engineering_quality.enabled:
        client = LLMClient(
            model=station_config.engineering_quality.model,
            timeout_seconds=llm_timeout_seconds,
        )
    return run_station_parquet_evaluation(
        station_config,
        config_path=suite,
        output_dir=output_dir,
        repository_root=REPOSITORY_ROOT,
        client=client,
    )


def _build_extensibility_suite(
    config: StageAWorkflowConfig,
    receipt: Path,
    run_dir: Path,
) -> Path:
    if any(
        value is None
        for value in (
            config.extensibility_suite_builder,
            config.linked_suite_dir,
            config.alternate_suite_dir,
            config.alternate_inputs_dir,
        )
    ):
        raise ValueError(
            "Alternate contract requires all extensibility builder output directories."
        )
    assert config.extensibility_suite_builder is not None
    assert config.linked_suite_dir is not None
    assert config.alternate_suite_dir is not None
    assert config.alternate_inputs_dir is not None
    _run_builder(
        [
            sys.executable,
            str(config.extensibility_suite_builder.resolve()),
            "--run-receipt",
            str(receipt.resolve()),
            "--primary-suite-dir",
            str(config.primary_suite_dir.resolve()),
            "--linked-suite-dir",
            str(config.linked_suite_dir.resolve()),
            "--alternate-suite-dir",
            str(config.alternate_suite_dir.resolve()),
            "--alternate-inputs-dir",
            str(config.alternate_inputs_dir.resolve()),
        ],
        log_dir=run_dir / "extensibility_suite_builder",
    )
    path = config.linked_suite_dir.resolve() / f"{config.pipeline_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Extensibility suite builder did not create {path}")
    return path


def _run_builder(command: list[str], *, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (log_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Suite builder exited with {completed.returncode}; see {log_dir}."
        )


def _prepare_runtime(
    pipeline_dir: Path,
    *,
    workflow_id: str,
    run_dir: Path,
    timeout_seconds: int,
    pipeline_agent: PipelineGenerationAgent,
    max_repair_attempts: int,
    repair_prompt_name: str,
    env_path: Path,
) -> _RuntimePreparation:
    """Create an isolated runtime and repair candidate-owned install failures."""

    if not BENCHMARK_REQUIREMENTS.is_file():
        raise FileNotFoundError(
            f"Benchmark requirements do not exist: {BENCHMARK_REQUIREMENTS}"
        )
    candidate_requirements = pipeline_dir / "requirements.txt"
    requirement_files = [BENCHMARK_REQUIREMENTS]
    dependency_source: Literal[
        "candidate_plus_benchmark", "benchmark_fallback"
    ] = "benchmark_fallback"
    if candidate_requirements.is_file():
        candidate_content = candidate_requirements.read_text(encoding="utf-8")
        _validate_requirements(candidate_content)
        requirement_files.append(candidate_requirements)
        dependency_source = "candidate_plus_benchmark"
    runtime = REPOSITORY_ROOT / "tmp" / "venvs" / "stage_a" / workflow_id
    if runtime.exists():
        raise FileExistsError(f"Stage A runtime already exists: {runtime}")
    runtime.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=False).create(runtime)
    python = runtime / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Virtual environment has no interpreter: {python}")

    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    for requirements in requirement_files:
        install_command.extend(["-r", str(requirements.resolve())])
    repair_log, repair_log_path = pipeline_agent.repair(
        pipeline_dir=pipeline_dir,
        command=install_command,
        prompt_name=repair_prompt_name,
        max_attempts=max_repair_attempts,
        timeout_seconds=timeout_seconds,
        env_path=env_path,
        protected_paths=PROTECTED_FAMILY_PATHS,
        repair_root=run_dir / "repairs" / "runtime_install",
    )
    if candidate_requirements.is_file():
        _validate_requirements(candidate_requirements.read_text(encoding="utf-8"))
    hash_input = "".join(
        f"{path.relative_to(REPOSITORY_ROOT)}\n{path.read_text(encoding='utf-8')}\n"
        for path in requirement_files
    )
    requirements_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    if repair_log.final_status not in {"already_succeeded", "repaired"}:
        return _RuntimePreparation(
            python=python,
            freeze_path=None,
            requirements_sha256=requirements_hash,
            dependency_source=dependency_source,
            repair_log=repair_log,
            repair_log_path=repair_log_path,
        )

    logs = run_dir / "runtime"
    logs.mkdir(parents=True, exist_ok=False)
    frozen = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        cwd=pipeline_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if frozen.returncode != 0:
        raise RuntimeError(f"Runtime freeze exited with {frozen.returncode}.")
    freeze_path = logs / "pip_freeze.txt"
    freeze_path.write_text(frozen.stdout, encoding="utf-8")
    return _RuntimePreparation(
        python=python,
        freeze_path=freeze_path,
        requirements_sha256=requirements_hash,
        dependency_source=dependency_source,
        repair_log=repair_log,
        repair_log_path=repair_log_path,
    )


def _has_generated_tests(pipeline_dir: Path) -> bool:
    """Match pytest's default Python test-file discovery patterns."""

    return any(
        path.is_file()
        for pattern in ("test_*.py", "*_test.py")
        for path in pipeline_dir.rglob(pattern)
    )


def _validate_requirements(content: str) -> None:
    """Reject installer directives and non-index dependency URLs."""

    for line_number, raw in enumerate(content.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError(
                f"requirements.txt line {line_number} contains an installer directive."
            )
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(
                f"requirements.txt line {line_number} is invalid: {error}"
            ) from error
        if requirement.url is not None:
            raise ValueError(
                f"requirements.txt line {line_number} uses a direct URL."
            )


def _phase(
    report: StageAWorkflowReport,
    name: str,
    status: Literal["passed", "failed", "blocked", "skipped", "diagnostic_failed"],
    *,
    artifact: str | None = None,
    detail: str | None = None,
) -> None:
    report.phases.append(
        StageAPhaseRecord(
            name=name,
            status=status,
            artifact=artifact,
            detail=detail,
        )
    )


def _finish(
    report_path: Path,
    report: StageAWorkflowReport,
    status: Literal["passed", "failed", "blocked"],
) -> tuple[StageAWorkflowReport, Path]:
    lineage_ok = _verify_lineage(report)
    report.lineage_unchanged = lineage_ok
    report.framework_source_unchanged = (
        _framework_source_sha256() == report.framework_source_sha256
    )
    _record_outcome_hashes(report)
    _phase(
        report,
        "lineage",
        "passed" if lineage_ok else "failed",
        detail=(
            "Workflow configuration, framework source, and declared inputs are unchanged."
            if lineage_ok
            else "Workflow configuration, framework source, or a declared input changed during execution."
        ),
    )
    if not lineage_ok:
        status = "failed"
        report.error = report.error or "Stage A lineage changed during execution."
    report.status = status
    report.completed_at = _now()
    _write_report(report_path, report)
    return report, report_path


def _write_report(path: Path, report: StageAWorkflowReport) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _display(path: Path) -> str:
    try:
        return path.absolute().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.absolute())


def _lineage_input_files(
    config: StageAWorkflowConfig,
    *,
    inventory: Path,
) -> dict[str, Path]:
    files = {
        "seed_contract": config.seed_contract.resolve(),
        "inventory": inventory.resolve(),
        "primary_suite_builder": config.primary_suite_builder.resolve(),
        "benchmark_requirements": BENCHMARK_REQUIREMENTS.resolve(),
    }
    optional = {
        "prompt_lock": config.prompt_lock,
        "alternate_contract": config.alternate_contract,
        "evaluation_profile": config.evaluation_profile,
        "evaluation_plan": config.evaluation_plan,
        "primary_suite_spec": config.primary_suite_spec,
        "extensibility_suite_builder": config.extensibility_suite_builder,
    }
    if config.generation_prompt_override is not None:
        optional.update(
            {
                "generation_system_prompt": config.generation_prompt_override.system_prompt,
                "generation_user_prompt": config.generation_prompt_override.user_prompt,
            }
        )
    files.update(
        {
            name: path.resolve()
            for name, path in optional.items()
            if path is not None
        }
    )
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage A lineage inputs are missing: {missing}")
    return files


def _framework_source_sha256() -> str:
    digest = hashlib.sha256()
    backend_root = REPOSITORY_ROOT / "backend"
    files = sorted(
        path
        for path in backend_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in FRAMEWORK_LINEAGE_SUFFIXES
    )
    for path in files:
        digest.update(path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_lineage(report: StageAWorkflowReport) -> bool:
    try:
        if (
            _sha256_file(_reported_path(report.workflow_config_snapshot))
            != report.workflow_config_sha256
        ):
            return False
        if _framework_source_sha256() != report.framework_source_sha256:
            return False
        if set(report.input_files) != set(report.input_sha256):
            return False
        return all(
            _sha256_file(_reported_path(report.input_files[name])) == expected
            for name, expected in report.input_sha256.items()
        )
    except OSError:
        return False


def _record_outcome_hashes(report: StageAWorkflowReport) -> None:
    outcomes = {
        "runtime_freeze": report.runtime_freeze,
        "acceptance_report": report.acceptance_report,
        "evaluation_report": report.evaluation_report,
    }
    for name, raw_path in outcomes.items():
        if raw_path is None:
            continue
        path = _reported_path(raw_path)
        if path.is_file():
            report.outcome_sha256[name] = _sha256_file(path)


def _reported_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _now() -> str:
    return datetime.now(UTC).isoformat()
