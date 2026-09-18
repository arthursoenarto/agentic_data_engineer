"""Trusted offline fixture-integrity probe; provider networking is not used."""
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
