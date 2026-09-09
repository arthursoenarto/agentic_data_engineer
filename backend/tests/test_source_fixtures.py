from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.source_fixtures import (
    SourceFixtureEntryRequest,
    SourceFixtureRequest,
    fetch_source_fixture,
    materialize_source_cache,
)


class SourceFixtureTests(unittest.TestCase):
    def test_local_fixture_is_immutable_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text("time,value\n2024-01-01,1\n", encoding="utf-8")
            dataset = root / "station_data"
            dataset.mkdir()
            request = SourceFixtureRequest(
                fixture_id="pilot_v1",
                dataset_slug="station_data",
                description="Small fixture",
                entries=[
                    SourceFixtureEntryRequest(
                        entry_id="source",
                        relative_path="raw/source.csv",
                        local_path="source.csv",
                    )
                ],
            )

            first, path, created = fetch_source_fixture(
                request, dataset_dir=dataset, repository_root=root
            )
            second, repeated_path, repeated_created = fetch_source_fixture(
                request, dataset_dir=dataset, repository_root=root
            )

            self.assertTrue(created)
            self.assertFalse(repeated_created)
            self.assertEqual(path, repeated_path)
            self.assertEqual(first, second)
            self.assertEqual(first.total_size_bytes, source.stat().st_size)

    def test_modified_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text("a,b\n1,2\n", encoding="utf-8")
            dataset = root / "station_data"
            dataset.mkdir()
            request = SourceFixtureRequest(
                fixture_id="pilot_v1",
                dataset_slug="station_data",
                description="Small fixture",
                entries=[
                    SourceFixtureEntryRequest(
                        entry_id="source",
                        relative_path="raw/source.csv",
                        local_path="source.csv",
                    )
                ],
            )
            fetch_source_fixture(request, dataset_dir=dataset, repository_root=root)
            (dataset / "data/source_fixtures/pilot_v1/raw/source.csv").write_text(
                "changed", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "content verification"):
                fetch_source_fixture(request, dataset_dir=dataset, repository_root=root)

    def test_source_cache_contains_only_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            source.write_text("a,b\n1,2\n", encoding="utf-8")
            dataset = root / "station_data"
            dataset.mkdir()
            request = SourceFixtureRequest(
                fixture_id="pilot_v1",
                dataset_slug="station_data",
                description="Small fixture",
                entries=[
                    SourceFixtureEntryRequest(
                        entry_id="source",
                        relative_path="raw/source.csv",
                        local_path="source.csv",
                    )
                ],
            )
            fetch_source_fixture(request, dataset_dir=dataset, repository_root=root)
            fixture = dataset / "data/source_fixtures/pilot_v1"
            (fixture / "reference.parquet").write_bytes(b"evaluator-only")

            cache = materialize_source_cache(fixture, root / "candidate_cache")

            self.assertTrue((cache / "source_fixture_manifest.json").is_file())
            self.assertEqual((cache / "raw/source.csv").read_text(), source.read_text())
            self.assertFalse((cache / "reference.parquet").exists())


if __name__ == "__main__":
    unittest.main()
