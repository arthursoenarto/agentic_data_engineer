"""Bounded execution-feedback loop for repairing generated pipelines."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from backend.agents.pipeline_repair.schemas import (
    FailureOrigin,
    PipelineExecutionResult,
    PipelineRepairAttempt,
    PipelineRepairLog,
    PipelineRepairProposal,
)
from backend.env import ENV_FILE, load_env
from backend.llm import LLMClient, LLMUsageSummary, LLMUsageTracker


MAX_REPAIR_ATTEMPTS = 3
MAX_CONTEXT_CHARS = 180_000
MAX_LOG_CHARS = 20_000
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SENSITIVE_KEY_PATTERN = re.compile(r"(API_)?KEY|TOKEN|SECRET|PASSWORD|AUTH", re.IGNORECASE)
CONTEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data",
    "processed",
    "raw",
    "raw_downloads",
    "repairs",
}
READ_ONLY_PIPELINE_CONTEXT = {"pipeline_contract.json", "run_pipeline.py"}
EXECUTION_INPUT_FLAGS = {"--contract", "--inventory"}

_ISOLATED_PATH_FLAGS = {
    "--output-dir": "output",
    "--run-receipt": "pipeline_run.json",
}


class PipelineRepairAgent:
    """Repair and re-execute a generated pipeline at most three times."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def repair_pipeline(
        self,
        *,
        pipeline_dir: Path,
        command: Sequence[str],
        prompt_name: str = "default",
        max_attempts: int = MAX_REPAIR_ATTEMPTS,
        timeout_seconds: int = 3600,
        env_path: Path = ENV_FILE,
        env_overrides: Mapping[str, str] | None = None,
        initial_execution: PipelineExecutionResult | None = None,
        repair_root: Path | None = None,
        protected_paths: set[str] | None = None,
    ) -> tuple[PipelineRepairLog, Path]:
        """Run, minimally repair, and verify a generated pipeline."""

        pipeline_dir = pipeline_dir.resolve()
        if not pipeline_dir.is_dir():
            raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")
        if not command:
            raise ValueError("A non-empty pipeline command is required.")
        if not 0 <= max_attempts <= MAX_REPAIR_ATTEMPTS:
            raise ValueError(f"max_attempts must be between 0 and {MAX_REPAIR_ATTEMPTS}.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not re.fullmatch(r"[a-z0-9_]+", prompt_name):
            raise ValueError("prompt_name must contain only lowercase letters, numbers, and underscores.")

        system_prompt = _prompt_text(prompt_name, "system")
        user_prompt_template = _prompt_text(prompt_name, "user")

        started_at = _now()
        resolved_repair_root = (repair_root or pipeline_dir / "repairs").resolve()
        resolved_repair_root.mkdir(parents=True, exist_ok=True)
        repair_dir = resolved_repair_root / f"repair_{_timestamp()}"
        repair_dir.mkdir(parents=True, exist_ok=False)

        load_env(env_path)
        child_env = dict(os.environ)
        if env_overrides:
            child_env.update(env_overrides)
        secrets = _known_secret_values(child_env)
        raw_command = [str(part) for part in command]
        raw_command = _normalize_command(raw_command, pipeline_dir=pipeline_dir)
        redacted_command = [_redact_text(part, secrets) for part in raw_command]
        execution_inputs = _execution_input_context(raw_command)

        initial = initial_execution or _run_isolated_command(
            raw_command,
            label="initial",
            repair_dir=repair_dir,
            cwd=pipeline_dir,
            env=child_env,
            timeout_seconds=timeout_seconds,
            secrets=secrets,
        )
        initial = _classify_execution(_redact_execution(initial, secrets))

        attempts: list[PipelineRepairAttempt] = []
        final_status = (
            "already_succeeded"
            if initial.ok
            else (
                "exhausted"
                if initial.failure_origin == "candidate"
                else "non_candidate_failure"
            )
        )
        current_failure = initial
        previous_repair_error: str | None = None

        if not initial.ok and initial.failure_origin == "candidate":
            for attempt_number in range(1, max_attempts + 1):
                attempt_started_at = _now()
                context_files: list[str] = []
                context_size = 0
                context_truncated = False
                proposal: PipelineRepairProposal | None = None
                usage: LLMUsageSummary | None = None
                tracker = LLMUsageTracker()

                try:
                    context, context_files, context_truncated = _pipeline_context(
                        pipeline_dir,
                        current_failure.stderr,
                        protected_paths=protected_paths or set(),
                        execution_inputs=execution_inputs,
                    )
                    context = _redact_text(context, secrets)
                    context_size = len(context)
                    proposal = self._llm.complete_json(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt_template.format(
                            command_json=json.dumps(redacted_command, indent=2),
                            failure_json=current_failure.model_dump_json(indent=2),
                            previous_repair_error=previous_repair_error or "None.",
                            pipeline_context=context,
                        ),
                        response_model=PipelineRepairProposal,
                        max_output_tokens=8192,
                        usage_tracker=tracker,
                    )
                    usage = tracker.summary()
                except Exception as error:
                    usage = _usage_if_recorded(tracker)
                    repair_error = _redact_text(str(error), secrets)
                    attempts.append(
                        PipelineRepairAttempt(
                            attempt_number=attempt_number,
                            started_at=attempt_started_at,
                            completed_at=_now(),
                            status="llm_failed",
                            failure_before=current_failure,
                            context_files=context_files,
                            context_size_chars=context_size,
                            context_truncated=context_truncated,
                            repair_error=repair_error,
                            llm_usage=usage,
                        )
                    )
                    previous_repair_error = repair_error
                    continue

                try:
                    _apply_edits(
                        pipeline_dir,
                        proposal,
                        protected_paths=protected_paths or set(),
                    )
                except (OSError, ValueError) as error:
                    repair_error = _redact_text(str(error), secrets)
                    attempts.append(
                        PipelineRepairAttempt(
                            attempt_number=attempt_number,
                            started_at=attempt_started_at,
                            completed_at=_now(),
                            status="patch_failed",
                            failure_before=current_failure,
                            context_files=context_files,
                            context_size_chars=context_size,
                            context_truncated=context_truncated,
                            proposal=proposal,
                            repair_error=repair_error,
                            llm_usage=usage,
                        )
                    )
                    previous_repair_error = repair_error
                    continue

                execution_after = _classify_execution(_run_isolated_command(
                    raw_command,
                    label=f"attempt_{attempt_number:02d}",
                    repair_dir=repair_dir,
                    cwd=pipeline_dir,
                    env=child_env,
                    timeout_seconds=timeout_seconds,
                    secrets=secrets,
                ))
                succeeded = execution_after.ok
                attempts.append(
                    PipelineRepairAttempt(
                        attempt_number=attempt_number,
                        started_at=attempt_started_at,
                        completed_at=_now(),
                        status="succeeded" if succeeded else "execution_failed",
                        failure_before=current_failure,
                        context_files=context_files,
                        context_size_chars=context_size,
                        context_truncated=context_truncated,
                        proposal=proposal,
                        execution_after=execution_after,
                        llm_usage=usage,
                    )
                )
                if succeeded:
                    final_status = "repaired"
                    break
                if execution_after.failure_origin != "candidate":
                    final_status = "non_candidate_failure"
                    break
                current_failure = execution_after
                previous_repair_error = None

        log = PipelineRepairLog(
            pipeline_dir=pipeline_dir.name,
            command=redacted_command,
            model=getattr(self._llm, "model", "unknown"),
            reasoning_effort=getattr(self._llm, "reasoning_effort", None),
            prompt_name=prompt_name,
            max_attempts=max_attempts,
            started_at=started_at,
            completed_at=_now(),
            final_status=final_status,
            initial_execution=initial,
            attempts=attempts,
            aggregate_llm_usage=_aggregate_usage(
                [attempt.llm_usage for attempt in attempts if attempt.llm_usage]
            ),
        )
        log_path = repair_dir / "repair_log.json"
        _write_atomic(log_path, log.model_dump_json(indent=2) + "\n")
        return log, log_path


def _prompt_text(prompt_name: str, kind: str) -> str:
    path = PROMPTS_DIR / f"{prompt_name}_{kind}.md"
    if not path.exists():
        raise RuntimeError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _pipeline_context(
    pipeline_dir: Path,
    stderr: str,
    *,
    protected_paths: set[str],
    execution_inputs: Sequence[tuple[str, Path]] = (),
) -> tuple[str, list[str], bool]:
    pipeline_dir = pipeline_dir.resolve()
    referenced = _referenced_paths(stderr, pipeline_dir)
    terms = _failure_terms(stderr)
    candidates: list[tuple[Path, str]] = []
    truncated = False
    for path in pipeline_dir.rglob("*"):
        if not _is_context_file(path, pipeline_dir):
            continue
        relative = path.relative_to(pipeline_dir).as_posix()
        if relative in protected_paths and relative not in READ_ONLY_PIPELINE_CONTEXT:
            continue
        try:
            candidates.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError):
            truncated = True
    candidates.sort(
        key=lambda item: (
            item[0].relative_to(pipeline_dir).as_posix() not in referenced,
            -_failure_relevance(item[1], terms),
            item[0].suffix.lower() != ".py",
            len(item[0].relative_to(pipeline_dir).parts),
            item[0].relative_to(pipeline_dir).as_posix(),
        )
    )

    rendered: list[str] = []
    selected: list[str] = []
    current_size = 0
    for flag, path in execution_inputs:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            truncated = True
            continue
        block = (
            f'<execution_input flag="{flag}" path="{path.name}" editable="false">\n'
            f"{content}\n</execution_input>"
        )
        if current_size + len(block) > MAX_CONTEXT_CHARS:
            truncated = True
            continue
        rendered.append(block)
        selected.append(f"execution_input:{flag}:{path.name}")
        current_size += len(block)

    for path, content in candidates:
        relative_path = path.relative_to(pipeline_dir).as_posix()
        editable = relative_path not in protected_paths
        block = (
            f'<pipeline_file path="{relative_path}" editable="{str(editable).lower()}">\n'
            f"{content}\n</pipeline_file>"
        )
        if current_size + len(block) > MAX_CONTEXT_CHARS:
            truncated = True
            continue
        rendered.append(block)
        selected.append(relative_path)
        current_size += len(block)

    if not rendered:
        raise RuntimeError("No repairable text files were found in the pipeline directory.")
    return "\n\n".join(rendered), selected, truncated


def _execution_input_context(command: Sequence[str]) -> list[tuple[str, Path]]:
    """Return bounded, non-secret public inputs referenced by framework flags."""

    inputs: list[tuple[str, Path]] = []
    for index, part in enumerate(command[:-1]):
        if part not in EXECUTION_INPUT_FLAGS:
            continue
        path = Path(command[index + 1]).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}:
            inputs.append((part, path))
    return inputs


def _failure_terms(stderr: str) -> set[str]:
    """Extract distinctive words used to prioritize likely failing source files."""

    ignored = {
        "error",
        "failed",
        "failure",
        "file",
        "line",
        "pipeline",
        "raise",
        "runtime",
        "valueerror",
    }
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", stderr[-MAX_LOG_CHARS:])
        if word.lower() not in ignored
    }


def _failure_relevance(content: str, terms: set[str]) -> int:
    lowered = content.lower()
    return sum(1 for term in terms if term in lowered)


def _referenced_paths(stderr: str, pipeline_dir: Path) -> set[str]:
    paths: set[str] = set()
    for value in re.findall(r'File "([^"]+)"', stderr):
        path = Path(value)
        if not path.is_absolute():
            path = pipeline_dir / path
        try:
            paths.add(path.resolve().relative_to(pipeline_dir).as_posix())
        except ValueError:
            continue
    return paths


def _is_context_file(path: Path, pipeline_dir: Path) -> bool:
    if not path.is_file() or path.name == ".env":
        return False
    try:
        relative = path.resolve().relative_to(pipeline_dir)
    except ValueError:
        return False
    if any(part in EXCLUDED_PARTS or part.endswith(".zarr") for part in relative.parts):
        return False
    return path.suffix.lower() in CONTEXT_SUFFIXES


def _apply_edits(
    pipeline_dir: Path,
    proposal: PipelineRepairProposal,
    *,
    protected_paths: set[str],
) -> None:
    updates: dict[Path, str] = {}
    for edit in proposal.edits:
        path = _repairable_path(
            pipeline_dir,
            edit.relative_path,
            protected_paths=protected_paths,
        )
        content = updates.get(path)
        if content is None:
            content = path.read_text(encoding="utf-8")
        matches = content.count(edit.old_text)
        if matches != 1:
            raise ValueError(
                f"Repair edit for {edit.relative_path} expected one exact old_text match; found {matches}."
            )
        updates[path] = content.replace(edit.old_text, edit.new_text, 1)

    _validate_updated_files(updates)
    for path, content in updates.items():
        _write_atomic(path, content)


def _validate_updated_files(updates: Mapping[Path, str]) -> None:
    """Reject malformed structured or Python patches before changing any file."""

    for path, content in updates.items():
        try:
            if path.suffix.lower() == ".py":
                ast.parse(content, filename=str(path))
            elif path.suffix.lower() == ".json":
                json.loads(content)
            elif path.suffix.lower() == ".toml":
                tomllib.loads(content)
        except (SyntaxError, ValueError, tomllib.TOMLDecodeError) as error:
            raise ValueError(
                f"Repair would make {path.name} invalid: {error}"
            ) from error


def _repairable_path(
    pipeline_dir: Path,
    relative_path: str,
    *,
    protected_paths: set[str],
) -> Path:
    path = (pipeline_dir / relative_path).resolve()
    try:
        relative = path.relative_to(pipeline_dir)
    except ValueError as error:
        raise ValueError(f"Repair path escapes the pipeline directory: {relative_path}") from error
    if relative.as_posix() in protected_paths:
        raise ValueError(f"Repair target is framework-owned: {relative_path}")
    if not path.is_file():
        raise ValueError(f"Repair target is not an existing file: {relative_path}")
    if path.name == "manifest.json":
        raise ValueError("The generation manifest is immutable and cannot be repaired.")
    if any(part in EXCLUDED_PARTS or part.endswith(".zarr") for part in relative.parts):
        raise ValueError(f"Repair target is not source/configuration text: {relative_path}")
    if path.suffix.lower() not in CONTEXT_SUFFIXES:
        raise ValueError(f"Repair target has an unsupported file type: {relative_path}")
    return path


def _run_command(
    command: list[str],
    *,
    execution_label: str | None = None,
    execution_dir: Path | None = None,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    secrets: list[str],
) -> PipelineExecutionResult:
    started_at = _now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        return_code = None
        stdout = _output_text(error.stdout)
        stderr = _output_text(error.stderr) or f"Command timed out after {timeout_seconds} seconds."
        timed_out = True
    except OSError as error:
        return_code = None
        stdout = ""
        stderr = str(error)
        timed_out = False

    duration = time.monotonic() - started
    return PipelineExecutionResult(
        command=[_redact_text(part, secrets) for part in command],
        execution_label=execution_label,
        execution_dir=str(execution_dir) if execution_dir is not None else None,
        started_at=started_at,
        completed_at=_now(),
        duration_seconds=duration,
        return_code=return_code,
        timed_out=timed_out,
        stdout=_tail(_redact_text(stdout, secrets)),
        stderr=_tail(_redact_text(stderr, secrets)),
        ok=return_code == 0 and not timed_out,
    )


def _normalize_command(command: list[str], *, pipeline_dir: Path) -> list[str]:
    """Resolve path-like executables before subprocess changes into pipeline_dir."""

    normalized = list(command)
    executable = Path(normalized[0]).expanduser()
    if executable.is_absolute() or len(executable.parts) == 1:
        return normalized
    caller_relative = executable.resolve()
    pipeline_relative = (pipeline_dir / executable).resolve()
    if caller_relative.is_file():
        normalized[0] = str(caller_relative)
    elif pipeline_relative.is_file():
        normalized[0] = str(pipeline_relative)
    return normalized


def _classify_execution(result: PipelineExecutionResult) -> PipelineExecutionResult:
    """Classify root-cause ownership before deciding whether to call an LLM."""

    if result.ok:
        return result.model_copy(update={"failure_origin": None, "failure_code": None})
    text = f"{result.stderr}\n{result.stdout}".lower()
    origin: FailureOrigin = "candidate"
    code = "CANDIDATE_EXECUTION_FAILURE"

    if result.return_code is None and not result.timed_out and any(
        marker in text
        for marker in (
            "no such file or directory",
            "permission denied",
            "exec format error",
        )
    ):
        origin, code = "harness", "HARNESS_COMMAND_LAUNCH_FAILED"
    elif _missing_harness_dependency(result.command, text):
        origin, code = "environment", "ENVIRONMENT_HARNESS_DEPENDENCY_MISSING"
    elif any(
        marker in text
        for marker in (
            "library not loaded",
            "symbol not found",
            "wrong architecture",
            "numpy.dtype size changed",
        )
    ):
        origin, code = "environment", "ENVIRONMENT_BINARY_INCOMPATIBILITY"
    elif any(
        marker in text
        for marker in (
            "http 502",
            "http error 502",
            "bad gateway",
            "service unavailable",
            "connectionerror",
            "connection refused",
            "connection reset",
            "temporary failure in name resolution",
            "name or service not known",
            "invalid api key",
            "http error 401",
            "401 client error",
            "http error 403",
            "403 client error",
        )
    ):
        origin, code = "provider", "PROVIDER_OR_NETWORK_FAILURE"
    elif any(
        marker in text
        for marker in (
            "evaluation output policy differs from the generation-visible pipeline policy",
            "shared public output policy hash mismatch",
            "family_pipeline_interface.v3 requires a zarr output policy",
        )
    ):
        origin, code = "policy", "PUBLIC_OUTPUT_POLICY_FAILURE"

    return result.model_copy(
        update={"failure_origin": origin, "failure_code": code}
    )


def _missing_harness_dependency(command: list[str], text: str) -> bool:
    """Distinguish a missing pytest harness from candidate-owned imports."""

    match = re.search(r"no module named ['\"]?([a-zA-Z0-9_.-]+)", text)
    if match is None or match.group(1).split(".", 1)[0] != "pytest":
        return False
    return any(
        command[index : index + 2] == ["-m", "pytest"]
        for index in range(max(0, len(command) - 1))
    )


def _run_isolated_command(
    command: list[str],
    *,
    label: str,
    repair_dir: Path,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    secrets: list[str],
) -> PipelineExecutionResult:
    """Run one verification with fresh output and receipt paths."""

    execution_dir = repair_dir / "executions" / label
    execution_dir.mkdir(parents=True, exist_ok=False)
    isolated = _isolated_command(command, execution_dir)
    return _run_command(
        isolated,
        execution_label=label,
        execution_dir=execution_dir,
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
        secrets=secrets,
    )


def _isolated_command(command: list[str], execution_dir: Path) -> list[str]:
    """Materialize path placeholders and replace stale-artifact CLI targets."""

    replacements = {
        "{execution_dir}": str(execution_dir),
        "{output_dir}": str(execution_dir / "output"),
        "{run_receipt}": str(execution_dir / "pipeline_run.json"),
    }
    materialized = [
        _replace_placeholders(part, replacements) for part in command
    ]
    for index, part in enumerate(materialized[:-1]):
        relative = _ISOLATED_PATH_FLAGS.get(part)
        if relative is not None:
            materialized[index + 1] = str(execution_dir / relative)
    return materialized


def _replace_placeholders(value: str, replacements: Mapping[str, str]) -> str:
    rendered = value
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)
    return rendered


def _redact_execution(
    execution: PipelineExecutionResult,
    secrets: list[str],
) -> PipelineExecutionResult:
    return execution.model_copy(
        update={
            "command": [_redact_text(part, secrets) for part in execution.command],
            "stdout": _tail(_redact_text(execution.stdout, secrets)),
            "stderr": _tail(_redact_text(execution.stderr, secrets)),
        }
    )


def _known_secret_values(env: Mapping[str, str]) -> list[str]:
    return [
        value
        for key, value in env.items()
        if SENSITIVE_KEY_PATTERN.search(key) and len(value) >= 4
    ]


def _redact_text(value: str, secrets: list[str]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _tail(value: str) -> str:
    if len(value) <= MAX_LOG_CHARS:
        return value
    return f"[...truncated...]\n{value[-MAX_LOG_CHARS:]}"


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _usage_if_recorded(tracker: LLMUsageTracker) -> LLMUsageSummary | None:
    try:
        return tracker.summary()
    except RuntimeError:
        return None


def _aggregate_usage(usages: list[LLMUsageSummary]) -> LLMUsageSummary | None:
    if not usages:
        return None
    pricing = {
        item.model_family: item
        for usage in usages
        for item in usage.pricing
    }
    return LLMUsageSummary(
        call_count=sum(usage.call_count for usage in usages),
        input_tokens=sum(usage.input_tokens for usage in usages),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        reasoning_tokens=sum(usage.reasoning_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        estimated_cost_usd=sum(usage.estimated_cost_usd for usage in usages),
        pricing=list(pricing.values()),
    )


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
