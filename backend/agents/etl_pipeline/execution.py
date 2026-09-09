"""Framework-owned execution orchestration for reusable generated pipelines."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from backend.agents.etl_pipeline.paths import (
    DatasetPipelinePaths,
    PipelineRunPaths,
    pipeline_run_id,
)
from backend.agents.etl_pipeline.schemas import (
    PipelineContract,
    PipelineManifest,
    PipelineRunReceipt,
    pipeline_contract_hash,
    pipeline_manifest_hash,
    stable_json_hash,
)
from backend.contracts import read_contract_lock
from backend.env import ENV_FILE, load_env


SENSITIVE_KEY_PATTERN = re.compile(
    r"(API_)?KEY|TOKEN|SECRET|PASSWORD|AUTH",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PipelineExecutionRun:
    """Paths and validated receipt for one immutable pipeline execution."""

    receipt: PipelineRunReceipt
    paths: PipelineRunPaths

    @property
    def succeeded(self) -> bool:
        return self.receipt.final_status == "succeeded"


def validate_pipeline_bundle(pipeline_dir: Path) -> tuple[PipelineManifest, PipelineContract]:
    """Validate reciprocal hashes and protected files for one pipeline version."""

    pipeline_dir = pipeline_dir.resolve()
    runner = pipeline_dir / "run_pipeline.py"
    manifest_path = pipeline_dir / "manifest.json"
    contract_path = pipeline_dir / "pipeline_contract.json"
    for path in (runner, manifest_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required framework-owned pipeline file missing: {path}")

    manifest = PipelineManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    pipeline_contract = PipelineContract.model_validate_json(
        contract_path.read_text(encoding="utf-8")
    )
    if pipeline_contract.generation_manifest_hash != pipeline_manifest_hash(manifest):
        raise ValueError("pipeline_contract.json does not match manifest.json provenance.")
    if manifest.pipeline_contract_hash != pipeline_contract_hash(pipeline_contract):
        raise ValueError("manifest.json does not match pipeline_contract.json.")
    if manifest.pipeline_id != pipeline_contract.pipeline_id:
        raise ValueError("Pipeline ID differs between manifest and pipeline contract.")
    return manifest, pipeline_contract


def resolve_pipeline_command(
    pipeline_contract: PipelineContract,
    *,
    contract_lock: Path,
    inventory: Path,
    cache_dir: Path,
    output_dir: Path,
    run_receipt: Path,
    python_executable: Path | None = None,
) -> list[str]:
    """Resolve the typed command template without invoking a shell."""

    replacements = {
        pipeline_contract.placeholders.contract: str(contract_lock),
        pipeline_contract.placeholders.inventory: str(inventory),
        pipeline_contract.placeholders.cache_dir: str(cache_dir),
        pipeline_contract.placeholders.output_dir: str(output_dir),
        pipeline_contract.placeholders.run_receipt: str(run_receipt),
    }
    command = [replacements.get(part, part) for part in pipeline_contract.command_template]
    if command and command[0] == "python":
        command[0] = str((python_executable or Path(sys.executable)).absolute())
    unresolved = [part for part in command if part.startswith("{") and part.endswith("}")]
    if unresolved:
        raise ValueError(f"Unresolved pipeline command placeholders: {unresolved}")
    return command


def execute_family_pipeline(
    *,
    dataset_dir: Path,
    pipeline_id: str,
    contract_lock_path: Path,
    inventory_path: Path | None = None,
    cache_dir: Path | None = None,
    run_id: str | None = None,
    timeout_seconds: int = 3600,
    env_path: Path = ENV_FILE,
    env_overrides: Mapping[str, str] | None = None,
    unset_environment_variables: set[str] | None = None,
    repair_run_reference: str | None = None,
    python_executable: Path | None = None,
) -> PipelineExecutionRun:
    """Execute one lock through a reusable pipeline and persist immutable evidence."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    workspace = DatasetPipelinePaths(dataset_dir.resolve())
    pipeline_dir = workspace.pipeline(pipeline_id)
    manifest, pipeline_contract = validate_pipeline_bundle(pipeline_dir)
    inventory = (inventory_path or workspace.inventory).resolve()
    selected_cache = (cache_dir or workspace.cache).resolve()
    lock_source = contract_lock_path.resolve()
    lock = read_contract_lock(lock_source)
    inventory_payload = json.loads(inventory.read_text(encoding="utf-8"))

    if lock.inventory_sha256 != stable_json_hash(inventory_payload):
        raise ValueError("Contract lock inventory hash does not match the execution inventory.")
    if lock.contract.dataset_slug != pipeline_contract.dataset_slug:
        raise ValueError("Contract lock belongs to a different dataset.")

    paths = workspace.run(pipeline_id, run_id or pipeline_run_id())
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    _copy_new(lock_source, paths.contract_lock)
    selected_cache.mkdir(parents=True, exist_ok=True)

    command = resolve_pipeline_command(
        pipeline_contract,
        contract_lock=paths.contract_lock,
        inventory=inventory,
        cache_dir=selected_cache,
        output_dir=paths.output_dir,
        run_receipt=paths.receipt,
        python_executable=python_executable,
    )
    load_env(env_path)
    child_env = dict(os.environ)
    if env_overrides:
        child_env.update(env_overrides)
    for name in unset_environment_variables or set():
        child_env.pop(name, None)
    if repair_run_reference:
        child_env["PIPELINE_REPAIR_REFERENCE"] = repair_run_reference
    secrets = _secret_values(child_env, pipeline_contract.credential_environment_variables)

    try:
        completed = subprocess.run(
            command,
            cwd=pipeline_dir,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = _redact(completed.stdout, secrets)
        stderr = _redact(completed.stderr, secrets)
    except subprocess.TimeoutExpired as error:
        stdout = _redact(_stream_text(error.stdout), secrets)
        stderr = _redact(_stream_text(error.stderr), secrets)
        stderr += f"\nPipeline execution timed out after {timeout_seconds} seconds.\n"
        _write_new(paths.stdout, stdout)
        _write_new(paths.stderr, stderr)
        if not paths.receipt.exists():
            _write_failure_receipt(
                paths=paths,
                pipeline_contract=pipeline_contract,
                manifest=manifest,
                lock_payload=json.loads(paths.contract_lock.read_text(encoding="utf-8")),
                inventory_payload=inventory_payload,
                command=command,
                cache_dir=selected_cache,
                diagnostic=f"TimeoutExpired: execution exceeded {timeout_seconds} seconds",
            )
        return PipelineExecutionRun(
            receipt=PipelineRunReceipt.model_validate_json(
                paths.receipt.read_text(encoding="utf-8")
            ),
            paths=paths,
        )

    _write_new(paths.stdout, stdout)
    _write_new(paths.stderr, stderr)
    if not paths.receipt.exists():
        _write_failure_receipt(
            paths=paths,
            pipeline_contract=pipeline_contract,
            manifest=manifest,
            lock_payload=json.loads(paths.contract_lock.read_text(encoding="utf-8")),
            inventory_payload=inventory_payload,
            command=command,
                cache_dir=selected_cache,
            diagnostic=f"Runner exited {completed.returncode} without writing a receipt.",
        )

    receipt = PipelineRunReceipt.model_validate_json(paths.receipt.read_text(encoding="utf-8"))
    _validate_receipt_links(
        receipt,
        pipeline_contract=pipeline_contract,
        manifest=manifest,
        lock_payload=json.loads(paths.contract_lock.read_text(encoding="utf-8")),
        inventory_payload=inventory_payload,
        paths=paths,
    )
    if receipt.exit_code != completed.returncode:
        raise ValueError("Pipeline receipt exit_code differs from the subprocess return code.")
    return PipelineExecutionRun(receipt=receipt, paths=paths)


def _validate_receipt_links(
    receipt: PipelineRunReceipt,
    *,
    pipeline_contract: PipelineContract,
    manifest: PipelineManifest,
    lock_payload: dict[str, object],
    inventory_payload: dict[str, object],
    paths: PipelineRunPaths,
) -> None:
    expected = {
        "run_id": paths.run_id,
        "pipeline_id": pipeline_contract.pipeline_id,
        "manifest_hash": pipeline_manifest_hash(manifest),
        "pipeline_contract_hash": pipeline_contract_hash(pipeline_contract),
        "contract_lock_hash": stable_json_hash(lock_payload),
        "inventory_hash": stable_json_hash(inventory_payload),
    }
    actual = {name: getattr(receipt, name) for name in expected}
    if actual != expected:
        raise ValueError(f"Pipeline receipt provenance mismatch: expected {expected}, got {actual}.")
    if receipt.final_status == "succeeded":
        expected_output = paths.output_dir.resolve()
        for artifact in receipt.outputs:
            if Path(artifact.path).resolve() != expected_output:
                raise ValueError("Pipeline receipt points outside its immutable output directory.")


def _write_failure_receipt(
    *,
    paths: PipelineRunPaths,
    pipeline_contract: PipelineContract,
    manifest: PipelineManifest,
    lock_payload: dict[str, object],
    inventory_payload: dict[str, object],
    command: list[str],
    cache_dir: Path,
    diagnostic: str,
) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    receipt = PipelineRunReceipt(
        run_id=paths.run_id,
        pipeline_id=pipeline_contract.pipeline_id,
        manifest_hash=pipeline_manifest_hash(manifest),
        pipeline_contract_hash=pipeline_contract_hash(pipeline_contract),
        contract_lock_hash=stable_json_hash(lock_payload),
        inventory_hash=stable_json_hash(inventory_payload),
        command=command,
        started_at=now,
        completed_at=now,
        duration_seconds=0,
        exit_code=1,
        final_status="failed",
        cache={"cache_dir": str(cache_dir)},
        diagnostics=[diagnostic],
    )
    _write_new(paths.receipt, receipt.model_dump_json(indent=2) + "\n")


def _secret_values(env: Mapping[str, str], declared_names: list[str]) -> tuple[str, ...]:
    names = set(declared_names)
    names.update(name for name in env if SENSITIVE_KEY_PATTERN.search(name))
    return tuple(
        sorted(
            {env[name] for name in names if name in env and len(env[name]) >= 4},
            key=len,
            reverse=True,
        )
    )


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _copy_new(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Immutable target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
