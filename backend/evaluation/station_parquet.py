"""Independent constrained evaluator for station time-series Parquet pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.file_copy import copytree_isolated
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backend.contracts import content_hash, read_contract_lock
from backend.evaluation.command_runner import run_observed_command
from backend.evaluation.check_library import validate_emitted_check_ids
from backend.evaluation.engineering_quality_v3 import (
    run_engineering_quality_v3,
    unassessed_engineering_quality_v3,
)
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    display_path,
    frozen_path_hash,
    logical_field_id,
    pipeline_artifact_hash,
    repo_path,
    scan_secret_files,
    secret_values,
    source_bundle_hash,
    verify_frozen_file,
    verify_frozen_path,
)
from backend.evaluation.isolation import prepare_command
from backend.evaluation.station_parquet_schemas import (
    StationEvaluationCheck,
    StationParquetEvaluationConfig,
    StationParquetEvaluationRun,
    StationScenarioResult,
)
from backend.llm import LLMClient


_CONSTRAINTS = (
    "contract_correctness",
    "semantic_equivalence",
    "rerun_safety",
    "provenance_security",
)


def run_station_parquet_evaluation(
    config: StationParquetEvaluationConfig,
    *,
    config_path: Path,
    output_dir: Path,
    repository_root: Path,
    client: LLMClient | None,
) -> tuple[StationParquetEvaluationRun, Path]:
    """Run two isolated materializations, hard gates, objectives, and MERODA."""

    root = repository_root.resolve()
    run_id = "station_eval_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checks: list[StationEvaluationCheck] = []
    paths = _ground(config, root)
    before = {name: _path_hash(path) for name, path in paths.items()}
    secrets = secret_values(config.security.secret_environment_variables)

    scenarios: list[StationScenarioResult] = []
    scenario_tables: list[Any] = []
    receipts: list[dict[str, Any]] = []
    scenario_artifacts: list[Path] = []
    for scenario in ("initial", "rerun"):
        result, table, receipt, artifact = _run_scenario(
            scenario, config, paths=paths, run_dir=run_dir, root=root, secrets=secrets
        )
        scenarios.append(result)
        if table is not None and receipt is not None and artifact is not None:
            scenario_tables.append(table)
            receipts.append(receipt)
            scenario_artifacts.append(artifact)

    checks.append(
        _check(
            "contract.grounding",
            "contract_correctness",
            True,
            "Frozen contract, inventory, candidate, policy, fixture, and reference are grounded.",
            {"candidate_id": config.candidate_id},
        )
    )
    execution_ok = len(scenarios) == 2 and all(
        item.execution.exit_code == 0 for item in scenarios
    )
    checks.append(
        _check(
            "contract.initial_execution",
            "contract_correctness",
            execution_ok,
            "Both isolated local-fixture materializations succeeded.",
            {"exit_codes": [item.execution.exit_code for item in scenarios]},
        )
    )
    materialization_ok = len(scenario_tables) == 2
    validation_errors = [
        item.validation_error for item in scenarios if item.validation_error
    ]
    for check_id in ("contract.initial_materialization", "contract.parquet_integrity"):
        checks.append(
            _check(
                check_id,
                "contract_correctness",
                materialization_ok,
                "Both outputs are readable policy-compliant Parquet artifacts.",
                {"validation_errors": validation_errors},
            )
        )

    reference = None
    reference_ok = False
    reference_evidence: dict[str, Any] = {}
    try:
        _validate_parquet_file(paths["reference"], config=config)
        reference = pq.read_table(paths["reference"])
        reference_evidence = _validate_table(
            reference, config=config, lock_path=paths["contract"]
        )
        reference_ok = True
    except Exception as error:  # noqa: BLE001
        reference_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "contract.station_reference_integrity",
            "contract_correctness",
            reference_ok,
            "The independent station reference is readable and contract-complete.",
            reference_evidence,
        )
    )

    table_ok = False
    table_evidence: dict[str, Any] = {}
    if materialization_ok:
        try:
            table_evidence = _validate_table(
                scenario_tables[0], config=config, lock_path=paths["contract"]
            )
            _validate_table(scenario_tables[1], config=config, lock_path=paths["contract"])
            table_ok = True
        except Exception as error:  # noqa: BLE001 - typed evaluation evidence.
            table_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "contract.station_scope_and_keys",
            "contract_correctness",
            table_ok,
            "Canonical station table schema, scope, and keys are valid.",
            table_evidence,
        )
    )

    workload_ok = False
    workload_evidence: dict[str, Any] = {}
    if table_ok:
        try:
            workload_evidence = _measure_filtered_station_scan(
                scenario_artifacts[0], config=config
            )
            workload_ok = True
        except Exception as error:  # noqa: BLE001
            workload_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "contract.filtered_station_scan",
            "contract_correctness",
            workload_ok,
            "The declared station/field workload returns rows from Parquet.",
            workload_evidence,
        )
    )

    semantic_ok = False
    semantic_evidence: dict[str, Any] = {}
    if table_ok and reference_ok and reference is not None:
        try:
            semantic_evidence = _compare_tables(
                scenario_tables[0], reference, config=config
            )
            semantic_ok = True
        except Exception as error:  # noqa: BLE001
            semantic_evidence = {"error": f"{type(error).__name__}: {error}"}
    checks.append(
        _check(
            "semantic.station_values_and_missingness",
            "semantic_equivalence",
            semantic_ok,
            "All reference keys, values, and missingness match.",
            semantic_evidence,
        )
    )

    rerun_execution_ok = len(scenarios) == 2 and scenarios[1].execution.exit_code == 0
    checks.append(
        _check(
            "rerun.execution",
            "rerun_safety",
            rerun_execution_ok,
            "The isolated exact rerun completed successfully.",
            {"exit_code": scenarios[1].execution.exit_code if len(scenarios) == 2 else None},
        )
    )
    rerun_ok = False
    rerun_evidence: dict[str, Any] = {}
    if table_ok:
        first = _logical_fingerprint(scenario_tables[0], config.output_policy.primary_key)
        second = _logical_fingerprint(scenario_tables[1], config.output_policy.primary_key)
        rerun_ok = first == second
        rerun_evidence = {"initial": first, "rerun": second}
    for check_id in ("rerun.logical_equivalence", "rerun.station_logical_equivalence"):
        checks.append(
            _check(
                check_id,
                "rerun_safety",
                rerun_ok,
                "Exact rerun logical fingerprints match.",
                rerun_evidence,
            )
        )

    receipt_ok = False
    receipt_evidence: dict[str, Any] = {}
    if materialization_ok:
        try:
            _validate_receipts(receipts, scenarios, config=config, paths=paths)
            cache_claims = [receipt.get("cache", {}) for receipt in receipts]
            if not all(
                int(claim.get("hits", 0)) > 0
                and int(claim.get("misses", 0)) == 0
                and int(claim.get("acquired", 0)) == 0
                for claim in cache_claims
            ):
                raise ValueError("Candidate did not report local fixture-only cache hits")
            receipt_ok = True
            receipt_evidence = {"fixture_only_cache": True}
        except Exception as error:  # noqa: BLE001
            receipt_evidence = {"error": f"{type(error).__name__}: {error}"}
    for check_id in (
        "provenance.receipts",
        "provenance.receipt.initial",
        "provenance.receipt.rerun",
        "provenance.identities",
    ):
        checks.append(
            _check(
                check_id,
                "provenance_security",
                receipt_ok,
                "Receipts, linked identities, and fixture-only cache claims are valid.",
                receipt_evidence,
            )
        )

    after = {name: _path_hash(path) for name, path in paths.items()}
    inputs_unchanged = before == after
    checks.append(
        _check(
            "provenance.trusted_inputs_unchanged",
            "provenance_security",
            inputs_unchanged,
            "Frozen evaluator inputs remained byte-identical.",
            {"mismatches": sorted(name for name in before if before[name] != after[name])},
        )
    )
    checks.append(
        _check(
            "provenance.candidate_claims",
            "provenance_security",
            True,
            "Candidate manifest, interface, and public policy claims are grounded.",
            {"candidate_id": config.candidate_id},
        )
    )
    isolation = [item.execution.isolation for item in scenarios]
    isolation_ok = bool(isolation) and all(
        item is not None and item.full for item in isolation
    )
    checks.append(
        _check(
            "security.execution_isolation",
            "provenance_security",
            isolation_ok,
            "Every candidate run was network-disabled and write-confined.",
            {"scenarios": len(isolation)},
        )
    )
    findings = scan_secret_files(
        [
            item
            for base in [run_dir, *scenario_artifacts]
            for item in (base.rglob("*") if base.is_dir() else [base])
            if item.is_file()
        ],
        known_values=secrets,
        max_bytes=config.security.max_text_file_bytes,
    )
    checks.append(
        _check(
            "security.secret_scan",
            "provenance_security",
            not findings,
            "Candidate outputs, receipts, and logs contain no detected secrets.",
            {"findings": findings},
        )
    )

    validate_emitted_check_ids(config.check_plan, {check.check_id for check in checks})

    constraints = {
        name: (
            "pass"
            if all(
                check.status == "pass"
                for check in checks
                if check.constraint == name
            )
            else "fail"
        )
        for name in _CONSTRAINTS
    }
    feasible = all(status == "pass" for status in constraints.values())
    operational: dict[str, float | int] = {}
    if feasible:
        operational = _operational_objectives(
            scenarios=scenarios,
            artifact=scenario_artifacts[0],
            config=config,
        )
        engineering = run_engineering_quality_v3(
            client=client,
            settings=config.engineering_quality,
            target_name=config.candidate_id,
            review_paths=config.candidate_source.paths,
            deterministic_evidence={
                "hard_constraints": constraints,
                "operational_objectives": operational,
                "table": table_evidence,
                "semantic": semantic_evidence,
            },
            repository_root=root,
            redact_values=secrets,
        )
    else:
        engineering = unassessed_engineering_quality_v3(
            settings=config.engineering_quality,
            reason="Hard constraints failed before engineering-objective eligibility.",
            target_name=config.candidate_id,
            review_paths=config.candidate_source.paths,
            repository_root=root,
            redact_values=secrets,
        )
    objective = dict(operational)
    if feasible and engineering.engineering_assessed and engineering.q_engineering is not None:
        objective["q_engineering"] = engineering.q_engineering
    ready = feasible and len(objective) == 4
    run = StationParquetEvaluationRun(
        suite_id=config.suite_id,
        suite_version=config.suite_version,
        candidate_id=config.candidate_id,
        run_id=run_id,
        scenarios=scenarios,
        checks=checks,
        constraints=constraints,
        feasible=feasible,
        operational_objectives=operational,
        engineering_quality=engineering,
        objective_vector=objective if ready else {},
        optimization_ready=ready,
        check_plan=config.check_plan,
        provenance={
            "config": display_path(config_path, root),
            "config_sha256": canonical_json_file_hash(config_path),
            "reference_sha256": config.reference.artifact.sha256,
            "source_fixture_sha256": config.execution.frozen_cache.sha256,
        },
    )
    result_path = run_dir / "evaluation.json"
    result_path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_render_report(run), encoding="utf-8")
    return run, result_path


def _ground(config: StationParquetEvaluationConfig, root: Path) -> dict[str, Path]:
    paths = {
        "contract": verify_frozen_file(config.contract_lock, root),
        "inventory": verify_frozen_file(config.inventory, root),
        "pipeline_contract": verify_frozen_file(config.execution.pipeline_contract, root),
        "manifest": verify_frozen_file(config.execution.manifest, root),
        "cache": verify_frozen_path(config.execution.frozen_cache, root),
        "reference": verify_frozen_path(config.reference.artifact, root),
    }
    if source_bundle_hash(config.candidate_source, root) != config.candidate_source.sha256:
        raise ValueError("Candidate source bundle hash differs from suite")
    lock = read_contract_lock(paths["contract"])
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    if lock.inventory_sha256 != content_hash(inventory):
        raise ValueError("Contract lock and inventory hashes differ")
    pipeline = json.loads(paths["pipeline_contract"].read_text(encoding="utf-8"))
    if pipeline.get("implementation_interface_version") != "family_pipeline_interface.v4":
        raise ValueError("Candidate does not implement the station Parquet interface")
    if pipeline.get("policy", {}).get("parquet") != config.output_policy.model_dump(mode="json"):
        raise ValueError("Generation-visible Parquet policy differs from the suite")
    return paths


def _run_scenario(
    scenario: Literal["initial", "rerun"],
    config: StationParquetEvaluationConfig,
    *,
    paths: dict[str, Path],
    run_dir: Path,
    root: Path,
    secrets: list[str],
) -> tuple[StationScenarioResult, Any | None, dict[str, Any] | None, Path | None]:
    scenario_dir = run_dir / "materializations" / scenario
    inputs = scenario_dir / "inputs"
    writable = scenario_dir / "writable"
    inputs.mkdir(parents=True)
    writable.mkdir()
    contract = inputs / "contract.lock.json"
    inventory = inputs / "inventory.json"
    shutil.copy2(paths["contract"], contract)
    shutil.copy2(paths["inventory"], inventory)
    cache = inputs / "cache"
    copytree_isolated(paths["cache"], cache)
    output = writable / "output"
    receipt_path = writable / "pipeline_run.json"
    command = _resolve_command(
        config.execution.command,
        contract=contract,
        inventory=inventory,
        cache=cache,
        output=output,
        receipt=receipt_path,
        python_executable=config.execution.python_executable,
    )
    with tempfile.TemporaryDirectory(prefix="station-eval-candidate-") as temporary:
        candidate = _stage_candidate(config, paths=paths, root=root, destination=Path(temporary))
        prepared = prepare_command(
            command,
            cwd=candidate,
            writable_root=writable,
            policy=config.execution_policy,
        )
        observation = run_observed_command(
            command,
            cwd=candidate,
            log_dir=scenario_dir / "logs",
            label="pipeline",
            repository_root=root,
            redact_values=secrets,
            timeout_seconds=config.execution.timeout_seconds,
            environment=prepared.environment,
            execution_command=prepared.command,
            isolation=prepared.evidence,
        )
    table = None
    receipt = None
    artifact = None
    validation_error = None
    if observation.exit_code == 0 and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            layout = receipt.get("dataset_artifact", {})
            relative = layout.get("file_path")
            if not isinstance(relative, str):
                raise ValueError("Receipt does not declare a Parquet file_path")
            artifact = (output / relative).resolve()
            artifact.relative_to(output.resolve())
            _validate_parquet_file(artifact, config=config)
            table = pq.read_table(artifact)
        except Exception as error:  # noqa: BLE001 - preserve candidate evidence.
            validation_error = f"{type(error).__name__}: {error}"
            table = None
            receipt = None
            artifact = None
    return (
        StationScenarioResult(
            scenario=scenario,
            execution=observation,
            output_path=display_path(output, root),
            receipt_path=display_path(receipt_path, root),
            artifact_path=display_path(artifact, root) if artifact else None,
            validation_error=validation_error,
        ),
        table,
        receipt,
        artifact,
    )


def _validate_table(table: Any, *, config: StationParquetEvaluationConfig, lock_path: Path) -> dict[str, Any]:
    policy = config.output_policy
    if not set(policy.required_columns).issubset(table.column_names):
        raise ValueError("Required canonical columns are missing")
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("Station table is empty")
    if frame[policy.primary_key].isnull().any(axis=None):
        raise ValueError("Primary-key columns contain nulls")
    if frame.duplicated(policy.primary_key).any():
        raise ValueError("Primary key is not unique")
    timestamps = frame["timestamp"].astype("datetime64[ns, UTC]")
    lock = read_contract_lock(lock_path)
    scope = lock.contract.scope["date_range"]
    start = np.datetime64(scope["start_date"])
    end = np.datetime64(scope["end_date"]) + np.timedelta64(1, "D")
    observed = timestamps.to_numpy(dtype="datetime64[ns]")
    if observed.min() < start or observed.max() >= end:
        raise ValueError("Output includes timestamps outside the contract")
    fields = sorted(str(value) for value in frame["field_id"].unique())
    stations = sorted(str(value) for value in frame["station_id"].unique())
    if fields != sorted(config.table.expected_field_ids):
        raise ValueError(f"Field identities differ: {fields}")
    if stations != sorted(config.table.expected_station_ids):
        raise ValueError(f"Station identities differ: {stations}")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "field_ids": fields,
        "station_ids": stations,
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
    }


def _validate_parquet_file(path: Path, *, config: StationParquetEvaluationConfig) -> None:
    metadata = pq.read_metadata(path)
    if metadata.num_rows <= 0 or metadata.num_row_groups <= 0:
        raise ValueError("Parquet file has no readable rows or row groups")
    schema = metadata.schema.to_arrow_schema()
    required = config.output_policy.required_columns
    if not set(required).issubset(schema.names):
        raise ValueError("Parquet physical schema omits required columns")
    timestamp_type = schema.field("timestamp").type
    if not pa.types.is_timestamp(timestamp_type) or timestamp_type.tz != "UTC":
        raise ValueError("Parquet timestamp must be an Arrow timestamp with UTC timezone")
    station_type = schema.field("station_id").type
    if not (pa.types.is_string(station_type) or pa.types.is_large_string(station_type)):
        raise ValueError("Parquet station_id must be a string")
    field_type = schema.field("field_id").type
    if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
        raise ValueError("Parquet field_id must be a string")
    if not pa.types.is_floating(schema.field("value").type):
        raise ValueError("Parquet value must be floating point")
    for row_group in range(metadata.num_row_groups):
        group = metadata.row_group(row_group)
        for index in range(group.num_columns):
            if group.column(index).compression.upper() != config.output_policy.compression.upper():
                raise ValueError("All Parquet columns must use the declared compression")


def _compare_tables(candidate: Any, reference: Any, *, config: StationParquetEvaluationConfig) -> dict[str, Any]:
    columns = config.output_policy.required_columns
    key = config.output_policy.primary_key
    left = candidate.select(columns).to_pandas().sort_values(key).reset_index(drop=True)
    right = reference.select(columns).to_pandas().sort_values(key).reset_index(drop=True)
    for frame in (left, right):
        frame["station_id"] = frame["station_id"].astype("string")
        frame["field_id"] = frame["field_id"].astype("string")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    if len(left) != len(right):
        raise ValueError(f"Row count differs: {len(left)} != {len(right)}")
    if not left[key].equals(right[key]):
        raise ValueError("Primary-key rows differ from the independent reference")
    left_values = left["value"].to_numpy(dtype=float)
    right_values = right["value"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
        raise ValueError("Missingness differs from the independent reference")
    if not np.allclose(
        left_values,
        right_values,
        atol=config.table.value_atol,
        rtol=config.table.value_rtol,
        equal_nan=True,
    ):
        delta = np.nanmax(np.abs(left_values - right_values))
        raise ValueError(f"Values differ from reference; max_abs_error={delta}")
    return {"rows_compared": len(left), "value_mismatches": 0, "missingness_mismatches": 0}


def _validate_receipts(
    receipts: list[dict[str, Any]],
    scenarios: list[StationScenarioResult],
    *,
    config: StationParquetEvaluationConfig,
    paths: dict[str, Path],
) -> None:
    pipeline = json.loads(paths["pipeline_contract"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest_core = dict(manifest)
    manifest_core.pop("pipeline_contract_hash", None)
    lock = json.loads(paths["contract"].read_text(encoding="utf-8"))
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    for receipt, scenario in zip(receipts, scenarios, strict=True):
        expected = {
            "pipeline_id": config.candidate_id,
            "manifest_hash": content_hash(manifest_core),
            "pipeline_contract_hash": content_hash(pipeline),
            "contract_lock_hash": content_hash(lock),
            "inventory_hash": content_hash(inventory),
            "exit_code": 0,
            "final_status": "succeeded",
        }
        mismatches = [name for name, value in expected.items() if receipt.get(name) != value]
        if mismatches:
            raise ValueError(f"Receipt identities differ: {mismatches}")
        artifact = Path(str(receipt["outputs"][0]["path"]))
        if pipeline_artifact_hash(artifact) != receipt["outputs"][0]["sha256"]:
            raise ValueError("Receipt output hash differs from observed bytes")


def _operational_objectives(
    *,
    scenarios: list[StationScenarioResult],
    artifact: Path,
    config: StationParquetEvaluationConfig,
) -> dict[str, float | int]:
    durations = [item.execution.duration_seconds for item in scenarios]
    workload = _measure_filtered_station_scan(artifact, config=config)
    return {
        "materialization_seconds": float(statistics.median(durations)),
        "consumer_samples_per_second": float(workload["median_rows_per_second"]),
        "output_bytes": int(artifact.stat().st_size),
    }


def _measure_filtered_station_scan(
    artifact: Path,
    *,
    config: StationParquetEvaluationConfig,
) -> dict[str, float | int | str]:
    filters = None
    if config.workload.protocol == "filtered_station_scan.v2":
        filters = [
            ("station_id", "=", config.table.expected_station_ids[0]),
            ("field_id", "=", config.table.expected_field_ids[0]),
        ]

    def read_workload() -> Any:
        table = pq.read_table(
            artifact,
            columns=config.output_policy.required_columns,
            filters=filters,
        )
        if filters is None:
            frame = table.to_pandas()
            frame = frame[
                (frame["station_id"] == config.table.expected_station_ids[0])
                & (frame["field_id"] == config.table.expected_field_ids[0])
            ]
            return frame
        return table

    for _ in range(config.workload.warmup_passes):
        read_workload()
    throughputs: list[float] = []
    matched_rows = 0
    for _ in range(config.workload.repetitions):
        started = time.perf_counter()
        result = read_workload()
        matched_rows = len(result)
        if matched_rows <= 0:
            raise ValueError("Declared station/field workload returned no rows")
        elapsed = max(time.perf_counter() - started, 1e-12)
        throughputs.append(matched_rows / elapsed)
    return {
        "protocol": config.workload.protocol,
        "matched_rows": matched_rows,
        "median_rows_per_second": float(statistics.median(throughputs)),
        "repetitions": len(throughputs),
    }


def _logical_fingerprint(table: Any, key: list[str]) -> str:
    frame = table.to_pandas().sort_values(key).reset_index(drop=True)
    payload = frame.to_json(orient="table", date_format="iso", double_precision=15)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_candidate(config: StationParquetEvaluationConfig, *, paths: dict[str, Path], root: Path, destination: Path) -> Path:
    cwd = repo_path(config.candidate_source.cwd, root)
    candidate = destination / "candidate"
    candidate.mkdir()
    for raw in config.candidate_source.paths:
        source = repo_path(raw, root)
        relative = source.relative_to(cwd)
        target = candidate / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    for name in ("pipeline_contract", "manifest"):
        source = paths[name]
        relative = source.relative_to(cwd)
        target = candidate / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return candidate


def _resolve_command(template: list[str], *, contract: Path, inventory: Path, cache: Path, output: Path, receipt: Path, python_executable: str | None) -> list[str]:
    replacements = {
        "{contract_lock_json}": str(contract.resolve()),
        "{dataset_inventory_json}": str(inventory.resolve()),
        "{cache_dir}": str(cache.resolve()),
        "{output_dir}": str(output.resolve()),
        "{pipeline_run_json}": str(receipt.resolve()),
    }
    command = [replacements.get(token, token) for token in template]
    if command[0] == "python":
        # Preserve virtual-environment symlinks: resolving them selects the base
        # interpreter and drops the environment's installed dependencies.
        command[0] = str(Path(python_executable or os.sys.executable).absolute())
    return command


def _path_hash(path: Path) -> str:
    return frozen_path_hash(path) if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()


def _check(check_id: str, constraint: str, passed: bool, summary: str, evidence: dict[str, Any]) -> StationEvaluationCheck:
    return StationEvaluationCheck(
        check_id=check_id,
        constraint=constraint,
        status="pass" if passed else "fail",
        summary=summary,
        evidence=evidence,
    )


def _render_report(run: StationParquetEvaluationRun) -> str:
    lines = [
        "# Station Parquet Evaluation",
        "",
        f"- Candidate: `{run.candidate_id}`",
        f"- Feasible: `{str(run.feasible).lower()}`",
        f"- Optimization ready: `{str(run.optimization_ready).lower()}`",
        "",
        "## Hard Constraints",
        "",
        "| Constraint | Status |",
        "|---|---|",
        *[f"| `{name}` | **{status}** |" for name, status in run.constraints.items()],
        "",
        "## Objective Vector",
        "",
        (f"`F(p) = {json.dumps(run.objective_vector, sort_keys=True)}`" if run.objective_vector else "Not eligible."),
        "",
        "## Checks",
        "",
        "| Check | Constraint | Status |",
        "|---|---|---|",
        *[f"| `{item.check_id}` | `{item.constraint}` | **{item.status}** |" for item in run.checks],
        "",
    ]
    return "\n".join(lines)
