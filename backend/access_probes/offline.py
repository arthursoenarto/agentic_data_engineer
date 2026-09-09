"""Network-free replay probe for immutable scientific source fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.access_probes.schemas import AccessProbeResult
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.contracts import file_hash


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenAccessProbeMetadata(_StrictModel):
    schema_version: Literal["offline_fixture_access_probe.v1"] = (
        "offline_fixture_access_probe.v1"
    )
    dataset_slug: str
    provider: str
    access_method: Literal["immutable_local_fixture_replay"] = (
        "immutable_local_fixture_replay"
    )
    fixture_manifest_path: str
    fixture_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    credential_references: list[str]
    provider_network_enabled: Literal[False] = False
    inventory_source_path: str
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenCharacterisationArtifacts(_StrictModel):
    output_dir: Path
    candidate: Path
    probe_script: Path
    probe_metadata: Path
    probe_result: Path
    inventory: Path


_PROBE_SCRIPT = '''"""Trusted offline fixture-integrity probe; provider networking is not used."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_path = Path(os.environ["SOURCE_FIXTURE_MANIFEST"]).resolve()
dataset_slug = os.environ["DATASET_SLUG"]
provider = os.environ.get("DATASET_PROVIDER", "unknown")
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = payload.get("entries", [])
checked = []
errors = []
for entry in entries:
    path = manifest_path.parent / entry["relative_path"]
    if not path.is_file():
        errors.append(f"missing:{entry['relative_path']}")
        continue
    size = path.stat().st_size
    digest = sha256(path)
    if size != entry["size_bytes"]:
        errors.append(f"size:{entry['relative_path']}")
    if digest != entry["sha256"]:
        errors.append(f"sha256:{entry['relative_path']}")
    checked.append({"relative_path": entry["relative_path"], "size_bytes": size, "sha256": digest})

result = {
    "schema_version": "access_probe.v1",
    "dataset_slug": dataset_slug,
    "provider": provider,
    "ok": not errors and bool(entries),
    "status": "verified_offline_fixture" if not errors and entries else "fixture_invalid",
    "checked_at": datetime.now(UTC).isoformat(),
    "credential_names": [],
    "summary": "Frozen source fixture is readable and identity-verified." if not errors else "Frozen source fixture verification failed.",
    "details": {
        "provider_network_enabled": False,
        "fixture_id": payload.get("fixture_id"),
        "entry_count": len(entries),
        "checked_entries": checked,
        "errors": errors,
    },
}
print(json.dumps(result, sort_keys=True))
'''


def characterize_frozen_dataset(
    *,
    inventory_path: Path,
    fixture_manifest_path: Path,
    output_dir: Path,
    candidate_path: Path | None = None,
    access_context_path: Path | None = None,
) -> tuple[FrozenAccessProbeMetadata, AccessProbeResult, FrozenCharacterisationArtifacts]:
    """Validate one local fixture and publish source-grounded characterisation."""

    inventory_path = inventory_path.resolve()
    fixture_manifest_path = fixture_manifest_path.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Characterisation output already exists: {destination}")
    inventory = DatasetInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    credentials = _credential_references(
        fixture_manifest_path=fixture_manifest_path,
        access_context_path=access_context_path,
    )
    metadata = FrozenAccessProbeMetadata(
        dataset_slug=inventory.dataset_slug,
        provider=inventory.provider or "unknown",
        fixture_manifest_path=str(fixture_manifest_path),
        fixture_manifest_sha256=file_hash(fixture_manifest_path),
        credential_references=credentials,
        inventory_source_path=str(inventory_path),
        inventory_sha256=file_hash(inventory_path),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        paths = _artifact_paths(temporary)
        if candidate_path is not None and candidate_path.is_file():
            shutil.copyfile(candidate_path, paths.candidate)
        else:
            paths.candidate.write_text(
                json.dumps(
                    {
                        "name": inventory.title,
                        "url": str(inventory.source_url),
                        "slug": inventory.dataset_slug,
                        "description": inventory.description,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        paths.probe_script.write_text(_PROBE_SCRIPT, encoding="utf-8")
        paths.probe_metadata.write_text(
            metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        shutil.copyfile(inventory_path, paths.inventory)
        completed = subprocess.run(
            [sys.executable, str(paths.probe_script)],
            cwd=temporary,
            env={
                "DATASET_SLUG": inventory.dataset_slug,
                "DATASET_PROVIDER": inventory.provider or "unknown",
                "SOURCE_FIXTURE_MANIFEST": str(fixture_manifest_path),
            },
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Offline fixture probe failed with exit {completed.returncode}: "
                f"{completed.stderr[:500]}"
            )
        result = AccessProbeResult.model_validate_json(completed.stdout)
        result = result.model_copy(update={"credential_names": credentials})
        paths.probe_result.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if not result.ok:
            raise ValueError(f"Offline fixture probe did not pass: {result.status}")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metadata, result, _artifact_paths(destination)


def _credential_references(
    *, fixture_manifest_path: Path, access_context_path: Path | None
) -> list[str]:
    result: set[str] = set()
    manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("entries", []):
        name = entry.get("credential_environment_variable")
        if isinstance(name, str) and name:
            result.add(name)
    if access_context_path is not None and access_context_path.is_file():
        context = json.loads(access_context_path.read_text(encoding="utf-8"))
        for item in context.get("credential_env_vars", []):
            name = item.get("env_var")
            if isinstance(name, str) and name:
                result.add(name)
    return sorted(result)


def _artifact_paths(root: Path) -> FrozenCharacterisationArtifacts:
    return FrozenCharacterisationArtifacts(
        output_dir=root,
        candidate=root / "candidate.json",
        probe_script=root / "access_probe.py",
        probe_metadata=root / "access_probe.metadata.json",
        probe_result=root / "access_probe.json",
        inventory=root / "dataset_inventory.json",
    )
