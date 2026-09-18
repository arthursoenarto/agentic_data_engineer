from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class FixtureError(RuntimeError):
    """Raised when the authoritative source fixture is absent or invalid."""


@dataclass(frozen=True)
class FixtureEntry:
    entry_id: str
    relative_path: str
    size_bytes: int
    sha256: str

    @property
    def safe_key(self) -> str:
        return f"fixture:{self.entry_id}:{self.sha256[:16]}"


@dataclass(frozen=True)
class VerifiedFixture:
    entries: list[FixtureEntry]
    paths: list[Path]


def verify_source_fixture(cache_dir: Path) -> VerifiedFixture:
    """Verify cache_dir/source_fixture_manifest.json and every listed raw file.

    The function performs only read operations, so read-only fixture caches are
    supported. It must be called before any provider or credential activity.
    """
    manifest_path = cache_dir / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise FixtureError("source_fixture_manifest.json is required for offline deterministic execution")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise FixtureError("unsupported source fixture manifest schema_version")
    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise FixtureError("source fixture manifest must contain at least one entry")

    entries: list[FixtureEntry] = []
    paths: list[Path] = []
    root = cache_dir.resolve()
    for raw in entries_raw:
        rel = raw.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise FixtureError("fixture relative_path must be a relative string")
        path = (cache_dir / rel).resolve()
        if root not in path.parents and path != root:
            raise FixtureError(f"fixture path escapes cache_dir: {rel}")
        if not path.is_file():
            raise FixtureError(f"fixture file not found: {rel}")
        size = path.stat().st_size
        expected_size = int(raw.get("size_bytes"))
        if size != expected_size:
            raise FixtureError(f"fixture size mismatch for {rel}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        expected_digest = str(raw.get("sha256"))
        if digest != expected_digest:
            raise FixtureError(f"fixture sha256 mismatch for {rel}")
        entries.append(FixtureEntry(str(raw.get("entry_id")), rel, size, digest))
        paths.append(path)
    return VerifiedFixture(entries=entries, paths=paths)
