from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from backend.access_probes.offline import characterize_frozen_dataset
from backend.agents.dataset_inventory.schemas import DatasetInventory


class OfflineAccessProbeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        inventory = root / "dataset_inventory.json"
        inventory.write_text(
            DatasetInventory(
                dataset_slug="example",
                title="Example",
                source_url="https://example.com/data",
                generated_at="2026-09-07T00:00:00Z",
                provider="example",
                extraction_method="deterministic",
                extractor_name="fixture",
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        fixture = root / "fixture"
        raw = fixture / "raw"
        raw.mkdir(parents=True)
        data = raw / "data.bin"
        data.write_bytes(b"trusted fixture")
        manifest = fixture / "source_fixture_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "source_fixture_manifest.v1",
                    "dataset_slug": "example",
                    "fixture_id": "example-v1",
                    "total_size_bytes": data.stat().st_size,
                    "entries": [
                        {
                            "relative_path": "raw/data.bin",
                            "size_bytes": data.stat().st_size,
                            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                            "credential_environment_variable": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return inventory, manifest

    def test_probe_verifies_local_fixture_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, manifest = self._fixture(root)
            metadata, result, artifacts = characterize_frozen_dataset(
                inventory_path=inventory,
                fixture_manifest_path=manifest,
                output_dir=root / "run/characterisation",
            )
            self.assertTrue(result.ok)
            self.assertFalse(metadata.provider_network_enabled)
            self.assertEqual(result.details["entry_count"], 1)
            self.assertTrue(artifacts.probe_script.is_file())

    def test_probe_rejects_corrupted_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, manifest = self._fixture(root)
            (manifest.parent / "raw/data.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                characterize_frozen_dataset(
                    inventory_path=inventory,
                    fixture_manifest_path=manifest,
                    output_dir=root / "run/characterisation",
                )


if __name__ == "__main__":
    unittest.main()
