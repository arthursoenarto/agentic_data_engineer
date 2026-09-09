"""Immutable local source fixtures shared by generated pipeline candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from backend.file_copy import copy_file_isolated


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceFixtureEntryRequest(_StrictModel):
    """One remote object or repository-local source copied into a fixture."""

    entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,119}$")
    relative_path: str
    url: HttpUrl | None = None
    local_path: str | None = None
    credential_environment_variable: str | None = None
    credential_header: str = "X-API-Key"

    @model_validator(mode="after")
    def source_and_path_are_safe(self) -> "SourceFixtureEntryRequest":
        if (self.url is None) == (self.local_path is None):
            raise ValueError("Exactly one of url or local_path is required")
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("relative_path must stay inside the fixture")
        if self.credential_environment_variable and self.url is None:
            raise ValueError("Credentials apply only to remote fixture entries")
        return self


class SourceFixtureRequest(_StrictModel):
    """Versioned instructions for a one-time local source acquisition."""

    schema_version: Literal["source_fixture_request.v1"] = "source_fixture_request.v1"
    fixture_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
    dataset_slug: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target_size_bytes: int | None = Field(default=None, ge=1)
    entries: list[SourceFixtureEntryRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_are_unique(self) -> "SourceFixtureRequest":
        ids = [entry.entry_id for entry in self.entries]
        paths = [entry.relative_path for entry in self.entries]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("Fixture entry IDs and relative paths must be unique")
        return self


class SourceFixtureEntry(_StrictModel):
    entry_id: str
    relative_path: str
    source: str
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    credential_environment_variable: str | None = None


class SourceFixtureManifest(_StrictModel):
    """Content-addressed evidence for one immutable local source snapshot."""

    schema_version: Literal["source_fixture_manifest.v1"] = (
        "source_fixture_manifest.v1"
    )
    fixture_id: str
    dataset_slug: str
    description: str
    created_at: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_size_bytes: int | None = None
    total_size_bytes: int = Field(ge=1)
    entries: list[SourceFixtureEntry] = Field(min_length=1)


def load_source_fixture_request(path: Path) -> SourceFixtureRequest:
    return SourceFixtureRequest.model_validate_json(path.read_text(encoding="utf-8"))


def fetch_source_fixture(
    request: SourceFixtureRequest,
    *,
    dataset_dir: Path,
    repository_root: Path,
    timeout_seconds: float = 120.0,
) -> tuple[SourceFixtureManifest, Path, bool]:
    """Fetch once, publish atomically, and verify existing fixtures on reuse."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    dataset = dataset_dir.resolve()
    if dataset.name != request.dataset_slug:
        raise ValueError("Fixture dataset_slug does not match dataset directory")
    destination = dataset / "data" / "source_fixtures" / request.fixture_id
    manifest_path = destination / "source_fixture_manifest.json"
    request_hash = _model_sha256(request)
    if manifest_path.is_file():
        manifest = SourceFixtureManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        _verify_fixture(manifest, destination, request_hash=request_hash)
        return manifest, manifest_path, False
    if destination.exists():
        raise FileExistsError(f"Incomplete immutable fixture exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{request.fixture_id}-", dir=destination.parent)
    )
    try:
        entries: list[SourceFixtureEntry] = []
        for requested in request.entries:
            target = temporary / requested.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if requested.url is not None:
                _download(
                    str(requested.url),
                    target,
                    credential_environment_variable=(
                        requested.credential_environment_variable
                    ),
                    credential_header=requested.credential_header,
                    timeout_seconds=timeout_seconds,
                )
                source = str(requested.url)
            else:
                assert requested.local_path is not None
                source_path = _repository_path(requested.local_path, repository_root)
                if not source_path.is_file():
                    raise FileNotFoundError(f"Fixture source does not exist: {source_path}")
                copy_file_isolated(source_path, target)
                source = requested.local_path
            size = target.stat().st_size
            if size <= 0:
                raise ValueError(f"Fixture entry is empty: {requested.entry_id}")
            entries.append(
                SourceFixtureEntry(
                    entry_id=requested.entry_id,
                    relative_path=requested.relative_path,
                    source=source,
                    size_bytes=size,
                    sha256=_sha256_file(target),
                    credential_environment_variable=(
                        requested.credential_environment_variable
                    ),
                )
            )
        manifest = SourceFixtureManifest(
            fixture_id=request.fixture_id,
            dataset_slug=request.dataset_slug,
            description=request.description,
            created_at=datetime.now(UTC).isoformat(),
            request_sha256=request_hash,
            target_size_bytes=request.target_size_bytes,
            total_size_bytes=sum(entry.size_bytes for entry in entries),
            entries=entries,
        )
        (temporary / "source_fixture_manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest, manifest_path, True


def materialize_source_cache(
    source_fixture: Path,
    destination: Path,
    *,
    verify_source: bool = True,
) -> Path:
    """Copy only manifest-declared source entries into an isolated cache."""

    source = source_fixture.resolve()
    target = destination.resolve()
    manifest_path = source / "source_fixture_manifest.json"
    manifest = SourceFixtureManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if target.exists():
        raise FileExistsError(f"Source cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)
    )
    try:
        copy_file_isolated(manifest_path, temporary / manifest_path.name)
        for entry in manifest.entries:
            relative = PurePosixPath(entry.relative_path)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(
                    f"Fixture entry path is unsafe: {entry.relative_path}"
                )
            source_path = source / Path(*relative.parts)
            if not source_path.is_file() or source_path.stat().st_size != entry.size_bytes:
                raise ValueError(f"Fixture entry is missing or changed: {source_path}")
            if verify_source and _sha256_file(source_path) != entry.sha256:
                raise ValueError(f"Fixture entry failed content verification: {source_path}")
            cache_path = temporary / Path(*relative.parts)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            copy_file_isolated(source_path, cache_path)
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _download(
    url: str,
    target: Path,
    *,
    credential_environment_variable: str | None,
    credential_header: str,
    timeout_seconds: float,
) -> None:
    headers = {"User-Agent": "agentic-data-engineer-source-fixture/1"}
    if credential_environment_variable:
        value = os.environ.get(credential_environment_variable)
        if not value:
            raise RuntimeError(
                f"Missing fixture credential: {credential_environment_variable}"
            )
        headers[credential_header] = value
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        with target.open("xb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def _verify_fixture(
    manifest: SourceFixtureManifest,
    directory: Path,
    *,
    request_hash: str,
) -> None:
    if manifest.request_sha256 != request_hash:
        raise ValueError("Existing fixture was created from a different request")
    observed_total = 0
    for entry in manifest.entries:
        path = directory / entry.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Fixture entry is missing: {path}")
        if path.stat().st_size != entry.size_bytes or _sha256_file(path) != entry.sha256:
            raise ValueError(f"Fixture entry failed content verification: {path}")
        observed_total += entry.size_bytes
    if observed_total != manifest.total_size_bytes:
        raise ValueError("Fixture total size does not match its entries")


def _repository_path(value: str, repository_root: Path) -> Path:
    root = repository_root.resolve()
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Fixture local_path escapes the repository: {value}") from error
    return resolved


def _model_sha256(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
