"""Format-neutral hashing, path confinement, redaction, and secret evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from backend.contracts import content_hash
from backend.evaluation.constrained_schemas import (
    CandidateSourceSpec,
    FrozenFile,
    FrozenPath,
)


_TEXT_SUFFIXES = {
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_ZARR_METADATA_NAMES = {".zattrs", ".zgroup", ".zmetadata", "zarr.json"}
_SENSITIVE_NAME = re.compile(r"(?:API_?)?KEY|TOKEN|SECRET|PASSWORD|AUTH", re.IGNORECASE)
_ASSIGNED_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)"
    r"\s*[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']"
)


def logical_field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    """Return the stable logical ID for one contract field-selector combination."""

    ordered = sorted(
        ((str(item["dimension"]), item.get("value")) for item in selectors),
        key=lambda item: item[0],
    )
    suffix = ",".join(
        f"{dimension}={json.dumps(value, sort_keys=True, separators=(',', ':'))}"
        for dimension, value in ordered
    )
    return f"{name}[{suffix}]" if suffix else name


def canonical_json_file_hash(path: Path) -> str:
    """Hash JSON meaning rather than incidental whitespace."""

    return content_hash(json.loads(path.read_text(encoding="utf-8")))


def frozen_path_hash(path: Path) -> str:
    """Hash exact bytes and relative file names for a file or directory."""

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    files = _regular_files(resolved)
    if not files:
        raise ValueError(f"Frozen path contains no files: {resolved}")
    digest = hashlib.sha256()
    for item in files:
        relative = (
            item.name if resolved.is_file() else item.relative_to(resolved).as_posix()
        )
        digest.update(relative.encode("utf-8") + b"\0")
        _update_file_hash(digest, item)
        digest.update(b"\0")
    return digest.hexdigest()


def source_bundle_hash(spec: CandidateSourceSpec, repository_root: Path) -> str:
    """Hash only suite-declared candidate source files, never the whole cwd."""

    cwd = repo_path(spec.cwd, repository_root)
    paths = [repo_path(path, repository_root) for path in spec.paths]
    for path in paths:
        require_within(path, cwd, "Candidate source path")
    digest = hashlib.sha256()
    seen: set[Path] = set()
    for source in sorted(paths):
        files = _regular_files(source)
        if not files:
            raise ValueError(f"Candidate source path contains no files: {source}")
        for item in files:
            if item in seen:
                raise ValueError(f"Candidate source paths overlap at {item}")
            seen.add(item)
            digest.update(item.relative_to(cwd).as_posix().encode("utf-8") + b"\0")
            _update_file_hash(digest, item)
            digest.update(b"\0")
    return digest.hexdigest()


def verify_frozen_file(spec: FrozenFile, repository_root: Path) -> Path:
    path = repo_path(spec.path, repository_root)
    if canonical_json_file_hash(path) != spec.sha256:
        raise ValueError(f"Frozen JSON hash mismatch: {spec.path}")
    return path


def verify_frozen_path(spec: FrozenPath, repository_root: Path) -> Path:
    path = repo_path(spec.path, repository_root)
    if frozen_path_hash(path) != spec.sha256:
        raise ValueError(f"Frozen artifact hash mismatch: {spec.path}")
    return path


def repo_path(raw: str, repository_root: Path) -> Path:
    path = Path(raw).expanduser()
    resolved = (
        path.resolve() if path.is_absolute() else (repository_root / path).resolve()
    )
    require_within(resolved, repository_root.resolve(), "Suite path")
    return resolved


def require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed root: {path}") from error


def resolve_evidence_path(raw: str, repository_root: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def path_size(path: Path) -> tuple[int, int]:
    files = _regular_files(path)
    return sum(item.stat().st_size for item in files), len(files)


def pipeline_artifact_hash(path: Path) -> str:
    """Match the generated-family receipt's physical artifact fingerprint."""

    digest = hashlib.sha256()
    files = _regular_files(path)
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        _update_file_hash(digest, item)
    return digest.hexdigest()


def source_files_hash(paths: Iterable[Path]) -> str:
    """Hash evaluator source files for report provenance."""

    files = sorted(path.resolve() for path in paths)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        _update_file_hash(digest, path)
        digest.update(b"\0")
    return digest.hexdigest()


def secret_values(declared_names: list[str]) -> list[str]:
    names = set(declared_names)
    names.update(name for name in os.environ if _SENSITIVE_NAME.search(name))
    return sorted(
        {os.environ[name] for name in names if len(os.environ.get(name, "")) >= 4},
        key=len,
        reverse=True,
    )


def scan_secret_files(
    paths: list[Path], *, known_values: list[str], max_bytes: int
) -> list[dict[str, str]]:
    """Return locations and categories, never matched secret text."""

    findings: list[dict[str, str]] = []
    for path in paths:
        if (
            path.suffix.lower() not in _TEXT_SUFFIXES
            and path.name not in _ZARR_METADATA_NAMES
        ):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        reason: str | None = None
        if any(value in text for value in known_values):
            reason = "known_environment_secret"
        elif "-----BEGIN PRIVATE KEY-----" in text:
            reason = "private_key_material"
        elif _ASSIGNED_SECRET.search(text):
            reason = "assigned_secret_like_value"
        if reason:
            findings.append({"path": str(path), "reason": reason})
    return findings


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "<redacted>")
    return text


def display_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:8]}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _update_file_hash(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)


def _regular_files(path: Path) -> list[Path]:
    """Return files beneath a declared path and reject nested symlink escapes."""

    if path.is_symlink():
        raise ValueError(f"Declared evidence path cannot be a symbolic link: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    descendants = sorted(path.rglob("*"))
    symlinks = [item for item in descendants if item.is_symlink()]
    if symlinks:
        raise ValueError(
            f"Declared evidence tree contains a symbolic link: {symlinks[0]}"
        )
    return [item for item in descendants if item.is_file()]
