"""Run project-local dataset access probe scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.env import ENV_FILE, load_env
from backend.access_probes.schemas import AccessProbeResult


PROBE_SCRIPT_NAME = "access_probe.py"
PROBE_RESULT_NAME = "access_probe.json"
SENSITIVE_KEY_PATTERN = re.compile(r"(API_)?KEY|TOKEN|SECRET|PASSWORD|AUTH", re.IGNORECASE)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _script_path(dataset_dir: Path) -> Path:
    return dataset_dir / PROBE_SCRIPT_NAME


def _result_path(dataset_dir: Path) -> Path:
    return dataset_dir / PROBE_RESULT_NAME


def read_access_probe(dataset_dir: Path) -> AccessProbeResult | None:
    """Read the latest access probe result if present."""

    path = _result_path(dataset_dir)
    if not path.exists():
        return None
    return AccessProbeResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def write_access_probe(dataset_dir: Path, result: AccessProbeResult) -> None:
    """Persist a probe result beside the dataset artifacts."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _result_path(dataset_dir).write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _known_secret_values(env_values: dict[str, str]) -> list[str]:
    secrets: list[str] = []
    for key, value in env_values.items():
        if SENSITIVE_KEY_PATTERN.search(key) and len(value) >= 4:
            secrets.append(value)
    return secrets


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value


def _load_probe_env(env_path: Path, *, dataset_slug: str, timeout_seconds: int) -> tuple[dict[str, str], dict[str, str]]:
    load_env(env_path)
    env_file_values = _read_env_file(env_path)
    env_values = {**{key: value for key, value in os.environ.items() if value}, **env_file_values}
    child_env = {
        **os.environ,
        **env_file_values,
        "DATASET_SLUG": dataset_slug,
        "ACCESS_PROBE_TIMEOUT_SECONDS": str(timeout_seconds),
    }
    return env_values, child_env


def _parse_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("Probe script produced no JSON on stdout.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end < start:
            raise
        return json.loads(text[start : end + 1])


def _result(
    *,
    dataset_slug: str,
    provider: str,
    ok: bool,
    status: str,
    credential_names: list[str],
    summary: str,
    details: dict[str, Any] | None = None,
) -> AccessProbeResult:
    return AccessProbeResult(
        dataset_slug=dataset_slug,
        provider=provider,
        ok=ok,
        status=status,
        checked_at=_now(),
        credential_names=credential_names,
        summary=summary,
        details=details or {},
    )


def run_project_access_probe(
    *,
    dataset_slug: str,
    dataset_dir: Path,
    env_path: Path = ENV_FILE,
    timeout_seconds: int = 20,
    credential_names: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> AccessProbeResult:
    """Execute project/datasets/{slug}/access_probe.py and persist its JSON result."""

    dataset_dir = dataset_dir.resolve()
    script_path = _script_path(dataset_dir)
    credential_names = credential_names or []

    if not script_path.exists():
        result = _result(
            dataset_slug=dataset_slug,
            provider="unknown",
            ok=False,
            status="not_implemented",
            credential_names=credential_names,
            summary=f"No {PROBE_SCRIPT_NAME} file exists for this dataset yet.",
            details={"expected_path": str(script_path)},
        )
        write_access_probe(dataset_dir, result)
        return result

    env_values, child_env = _load_probe_env(env_path, dataset_slug=dataset_slug, timeout_seconds=timeout_seconds)
    if env_overrides:
        env_values.update(env_overrides)
        child_env.update(env_overrides)
    secrets = _known_secret_values(env_values)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=dataset_dir,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        result = _result(
            dataset_slug=dataset_slug,
            provider="unknown",
            ok=False,
            status="timeout",
            credential_names=credential_names,
            summary=f"Access probe timed out after {timeout_seconds} seconds.",
        )
        write_access_probe(dataset_dir, result)
        return result

    if completed.returncode != 0:
        result = _result(
            dataset_slug=dataset_slug,
            provider="unknown",
            ok=False,
            status="failed",
            credential_names=credential_names,
            summary=f"Access probe exited with status code {completed.returncode}.",
            details={"stderr": _redact(completed.stderr.strip()[:1200], secrets)},
        )
        write_access_probe(dataset_dir, result)
        return result

    try:
        payload = _parse_stdout(completed.stdout)
        payload = _redact(payload, secrets)
        result = AccessProbeResult.model_validate(payload)
        if result.dataset_slug != dataset_slug:
            result = result.model_copy(update={"dataset_slug": dataset_slug})
        if not result.credential_names and credential_names:
            result = result.model_copy(update={"credential_names": credential_names})
    except (ValueError, json.JSONDecodeError) as error:
        result = _result(
            dataset_slug=dataset_slug,
            provider="unknown",
            ok=False,
            status="invalid_output",
            credential_names=credential_names,
            summary="Access probe did not return valid access_probe.v1 JSON.",
            details={"error": str(error)},
        )

    write_access_probe(dataset_dir, result)
    return result
