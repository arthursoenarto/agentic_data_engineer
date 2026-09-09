"""Isolated-clone execution harness for repository-extension proposals."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from backend.agents.etl_pipeline.extension import validate_extension_bundle
from backend.agents.etl_pipeline.extension_schemas import (
    ExtensionCommand,
    ExtensionCommandKind,
    ExtensionCommandResult,
    ExtensionContract,
    RepositoryExtensionRunReceipt,
)
from backend.agents.etl_pipeline.paths import DatasetPipelinePaths, pipeline_run_id


SENSITIVE_KEY_PATTERN = re.compile(r"(API_)?KEY|TOKEN|SECRET|PASSWORD|AUTH", re.IGNORECASE)
PYTEST_COUNT_PATTERNS = {
    "tests_collected": re.compile(r"collected\s+(\d+)\s+items?"),
    "tests_passed": re.compile(r"(\d+)\s+passed"),
    "tests_failed": re.compile(r"(\d+)\s+failed"),
}
DIAGNOSTIC_COUNT_PATTERN = re.compile(r"Found\s+(\d+)\s+diagnostics?")


@dataclass(frozen=True)
class RepositoryExtensionRun:
    """Paths and receipt for one isolated repository-extension execution."""

    run_dir: Path
    receipt: RepositoryExtensionRunReceipt

    @property
    def succeeded(self) -> bool:
        return self.receipt.final_status == "succeeded"


def execute_repository_extension(
    *,
    dataset_dir: Path,
    extension_id: str,
    source_repository: Path | None = None,
    run_id: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> RepositoryExtensionRun:
    """Apply and check an extension in an independent disposable clone."""

    dataset_dir = dataset_dir.resolve()
    extension_dir = dataset_dir / "pipelines" / extension_id
    manifest, contract, _ = validate_extension_bundle(extension_dir)
    source = (
        source_repository.expanduser().resolve()
        if source_repository
        else Path(contract.repository_path).expanduser().resolve()
    )
    if not source.is_dir():
        raise FileNotFoundError(f"Extension source repository not found: {source}")

    resolved_run_id = run_id or pipeline_run_id()
    paths = DatasetPipelinePaths(dataset_dir).run(extension_id, resolved_run_id)
    run_dir = paths.run_dir
    work_dir = run_dir.parent / ".work" / resolved_run_id
    clone = work_dir / "terraio-copy"
    run_dir.mkdir(parents=True, exist_ok=False)
    work_dir.mkdir(parents=True, exist_ok=False)

    patch_source = extension_dir / "patch.diff"
    patch_copy = run_dir / "patch.diff"
    shutil.copyfile(patch_source, patch_copy)
    source_hash_before = _tree_hash(source)
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    commands: list[ExtensionCommandResult] = []
    diagnostics: list[str] = []
    final_status = "patch_failed"
    patch_applied = False
    combined_stdout: list[str] = []
    combined_stderr: list[str] = []
    child_env = dict(os.environ)
    if env_overrides:
        child_env.update(env_overrides)
    secrets = _secret_values(child_env)

    try:
        _clone_exact_commit(source, clone, contract.base_commit)
        baseline_results = _run_commands(
            contract.baseline_commands,
            phase="baseline",
            repository=clone,
            run_dir=run_dir,
            env=child_env,
            secrets=secrets,
            start_index=len(commands),
        )
        commands.extend(baseline_results)
        _append_command_output(baseline_results, run_dir, combined_stdout, combined_stderr)
        if not all(result.succeeded for result in baseline_results):
            final_status = "baseline_failed"
        else:
            # Baseline tools may create caches or build artifacts. Re-clone so
            # patch validation always starts from the exact contracted commit.
            shutil.rmtree(clone)
            _clone_exact_commit(source, clone, contract.base_commit)
            patch_results = _apply_patch(
                clone=clone,
                patch=patch_copy,
                run_dir=run_dir,
                env=child_env,
                secrets=secrets,
                start_index=len(commands),
            )
            commands.extend(patch_results)
            _append_command_output(patch_results, run_dir, combined_stdout, combined_stderr)
            if not all(result.succeeded for result in patch_results):
                final_status = "patch_failed"
            else:
                patch_applied = True
                _verify_changed_paths(clone, manifest.changed_paths)
                interface_results = _run_interface_checks(
                    interfaces=contract.required_public_interfaces,
                    source_roots=contract.python_source_roots,
                    python_executable=_contract_python(contract),
                    repository=clone,
                    run_dir=run_dir,
                    env=child_env,
                    secrets=secrets,
                    start_index=len(commands),
                )
                commands.extend(interface_results)
                _append_command_output(
                    interface_results,
                    run_dir,
                    combined_stdout,
                    combined_stderr,
                )
                if not all(result.succeeded for result in interface_results):
                    final_status = "verification_failed"
                else:
                    verification_results = _run_commands(
                        contract.verification_commands,
                        phase="verification",
                        repository=clone,
                        run_dir=run_dir,
                        env=child_env,
                        secrets=secrets,
                        start_index=len(commands),
                    )
                    commands.extend(verification_results)
                    _append_command_output(
                        verification_results,
                        run_dir,
                        combined_stdout,
                        combined_stderr,
                    )
                    if not all(result.succeeded for result in verification_results):
                        final_status = "verification_failed"
                    else:
                        workflow_results = _run_commands(
                            contract.representative_workflow_commands,
                            phase="workflow",
                            repository=clone,
                            run_dir=run_dir,
                            env=child_env,
                            secrets=secrets,
                            start_index=len(commands),
                        )
                        commands.extend(workflow_results)
                        _append_command_output(
                            workflow_results,
                            run_dir,
                            combined_stdout,
                            combined_stderr,
                        )
                        final_status = (
                            "succeeded"
                            if all(result.succeeded for result in workflow_results)
                            else "workflow_failed"
                        )
    except Exception as error:
        diagnostics.append(f"{type(error).__name__}: {error}")
        combined_stderr.append(diagnostics[-1] + "\n")
    finally:
        source_hash_after = _tree_hash(source)
        if source_hash_after != source_hash_before:
            final_status = "source_modified"
            diagnostics.append("The authoritative source repository changed during execution.")

        completed = datetime.now(UTC)
        receipt = RepositoryExtensionRunReceipt(
            run_id=resolved_run_id,
            extension_id=extension_id,
            base_commit=contract.base_commit,
            patch_sha256=manifest.patch_sha256,
            source_tree_sha256_before=source_hash_before,
            source_tree_sha256_after=source_hash_after,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=time.perf_counter() - started_clock,
            final_status=final_status,
            patch_applied=patch_applied,
            changed_paths=manifest.changed_paths,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "git": _git_version(),
            },
            commands=commands,
            diagnostics=diagnostics,
        )
        _write_new(run_dir / "logs" / "stdout.log", "".join(combined_stdout))
        _write_new(run_dir / "logs" / "stderr.log", "".join(combined_stderr))
        _write_new(
            run_dir / "test_results.json",
            json.dumps(
                [result.model_dump(mode="json") for result in commands],
                indent=2,
            )
            + "\n",
        )
        _write_new(
            run_dir / "pipeline_run.json",
            receipt.model_dump_json(indent=2) + "\n",
        )
        shutil.rmtree(work_dir, ignore_errors=True)

    return RepositoryExtensionRun(run_dir=run_dir, receipt=receipt)


def _run_commands(
    command_specs: list[ExtensionCommand],
    *,
    phase: str,
    repository: Path,
    run_dir: Path,
    env: Mapping[str, str],
    secrets: tuple[str, ...],
    start_index: int,
) -> list[ExtensionCommandResult]:
    return [
        _run_command(
            command,
            phase=phase,
            repository=repository,
            run_dir=run_dir,
            env=env,
            secrets=secrets,
            index=start_index + offset + 1,
        )
        for offset, command in enumerate(command_specs)
    ]


def _apply_patch(
    *,
    clone: Path,
    patch: Path,
    run_dir: Path,
    env: Mapping[str, str],
    secrets: tuple[str, ...],
    start_index: int,
) -> list[ExtensionCommandResult]:
    commands = [
        ExtensionCommand(
            name="patch_check",
            kind=ExtensionCommandKind.WORKFLOW,
            argv=["git", "apply", "--check", str(patch)],
            timeout_seconds=120,
        ),
        ExtensionCommand(
            name="patch_apply",
            kind=ExtensionCommandKind.WORKFLOW,
            argv=["git", "apply", str(patch)],
            timeout_seconds=120,
        ),
    ]
    results: list[ExtensionCommandResult] = []
    for offset, command in enumerate(commands):
        result = _run_command(
            command,
            phase="verification",
            repository=clone,
            run_dir=run_dir,
            env=env,
            secrets=secrets,
            index=start_index + offset + 1,
        )
        results.append(result)
        if not result.succeeded:
            break
    return results


def _run_interface_checks(
    *,
    interfaces: list[str],
    source_roots: list[str],
    python_executable: str,
    repository: Path,
    run_dir: Path,
    env: Mapping[str, str],
    secrets: tuple[str, ...],
    start_index: int,
) -> list[ExtensionCommandResult]:
    roots = repr(source_roots)
    commands = []
    for target in interfaces:
        module, attribute = target.split(":", maxsplit=1)
        script = (
            "import importlib, sys; "
            f"[sys.path.insert(0, root) for root in {roots}]; "
            f"module = importlib.import_module({module!r}); "
            f"assert getattr(module, {attribute!r}) is not None"
        )
        commands.append(
            ExtensionCommand(
                name=f"interface_{module}_{attribute}",
                kind=ExtensionCommandKind.TYPECHECK,
                argv=[python_executable, "-c", script],
                timeout_seconds=120,
            )
        )
    return _run_commands(
        commands,
        phase="verification",
        repository=repository,
        run_dir=run_dir,
        env=env,
        secrets=secrets,
        start_index=start_index,
    )


def _contract_python(contract: ExtensionContract) -> str:
    command_groups = [
        getattr(contract, "baseline_commands", []),
        getattr(contract, "verification_commands", []),
        getattr(contract, "representative_workflow_commands", []),
    ]
    for commands in command_groups:
        for command in commands:
            executable = command.argv[0]
            if Path(executable).name in {"python", "python3"}:
                return executable
    return sys.executable


def _run_command(
    command: ExtensionCommand,
    *,
    phase: str,
    repository: Path,
    run_dir: Path,
    env: Mapping[str, str],
    secrets: tuple[str, ...],
    index: int,
) -> ExtensionCommandResult:
    argv = list(command.argv)
    if argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=repository,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=command.timeout_seconds,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        return_code = 124
        stdout = _stream_text(error.stdout)
        stderr = _stream_text(error.stderr)
        stderr += f"\nCommand timed out after {command.timeout_seconds} seconds.\n"

    stdout = _redact(stdout, secrets)
    stderr = _redact(stderr, secrets)
    completed_at = datetime.now(UTC)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", command.name).strip("_") or "command"
    stdout_path = run_dir / "logs" / "commands" / f"{index:03d}_{safe_name}.stdout.log"
    stderr_path = run_dir / "logs" / "commands" / f"{index:03d}_{safe_name}.stderr.log"
    _write_new(stdout_path, stdout)
    _write_new(stderr_path, stderr)
    counts = _pytest_counts(stdout + "\n" + stderr)
    diagnostic_match = DIAGNOSTIC_COUNT_PATTERN.search(stdout + "\n" + stderr)
    return ExtensionCommandResult(
        phase=phase,
        name=command.name,
        kind=command.kind,
        argv=argv,
        started_at=started.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_seconds=time.perf_counter() - started_clock,
        return_code=return_code,
        timed_out=timed_out,
        stdout_log=stdout_path.relative_to(run_dir).as_posix(),
        stderr_log=stderr_path.relative_to(run_dir).as_posix(),
        expected_return_codes=command.expected_return_codes,
        diagnostic_count=(
            int(diagnostic_match.group(1)) if diagnostic_match else None
        ),
        expected_diagnostic_count=command.expected_diagnostic_count,
        **counts,
    )


def _append_command_output(
    results: list[ExtensionCommandResult],
    run_dir: Path,
    combined_stdout: list[str],
    combined_stderr: list[str],
) -> None:
    for result in results:
        header = f"\n[{result.phase}:{result.name}]\n"
        combined_stdout.extend(
            [header, (run_dir / result.stdout_log).read_text(encoding="utf-8")]
        )
        combined_stderr.extend(
            [header, (run_dir / result.stderr_log).read_text(encoding="utf-8")]
        )


def _clone_exact_commit(source: Path, clone: Path, commit: str) -> None:
    completed = subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(source), str(clone)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Independent repository clone failed: {completed.stderr.strip()}")
    checkout = subprocess.run(
        ["git", "-C", str(clone), "checkout", "--quiet", "--detach", commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if checkout.returncode != 0:
        raise RuntimeError(f"Base commit checkout failed: {checkout.stderr.strip()}")
    actual = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != commit:
        raise ValueError(f"Clone resolved commit {actual}, expected {commit}.")


def _verify_changed_paths(clone: Path, expected_paths: list[str]) -> None:
    subprocess.run(
        ["git", "-C", str(clone), "add", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = sorted(
        path
        for path in subprocess.run(
            ["git", "-C", str(clone), "diff", "--cached", "--name-only", "--"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if path
    )
    if actual != sorted(expected_paths):
        raise ValueError(
            f"Applied patch changed unexpected paths: expected {sorted(expected_paths)}, got {actual}."
        )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if ".git" not in candidate.relative_to(root).parts
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(f"L\0{relative}\0{os.readlink(path)}\0".encode("utf-8"))
        elif path.is_file():
            digest.update(f"F\0{relative}\0".encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        elif path.is_dir():
            digest.update(f"D\0{relative}\0".encode("utf-8"))
    return digest.hexdigest()


def _pytest_counts(output: str) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for field, pattern in PYTEST_COUNT_PATTERNS.items():
        match = pattern.search(output)
        counts[field] = int(match.group(1)) if match else None
    return counts


def _secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for name, value in env.items()
                if SENSITIVE_KEY_PATTERN.search(name) and len(value) >= 4
            },
            key=len,
            reverse=True,
        )
    )


def _redact(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _git_version() -> str:
    return subprocess.run(
        ["git", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
