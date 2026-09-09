from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from backend.agents.contract_drafting.schemas import (
    ContractFieldSpec,
    ContractSelectorSpec,
    DatasetContract,
)
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.contracts import (
    content_hash,
    load_dataset_contract,
    lock_dataset_contract,
    write_editable_contract,
)


def _inventory(*, combinations: bool = False) -> DatasetInventory:
    constraints = {}
    if combinations:
        constraints["field_selector_combinations"] = {
            "temperature": [{"pressure_level": "500"}],
            "geopotential": [{"pressure_level": "850"}],
        }
    return DatasetInventory(
        dataset_slug="era5_pressure",
        title="ERA5 pressure levels",
        source_url="https://example.com/era5",
        generated_at="2026-07-28T00:00:00+00:00",
        provider="ECMWF",
        dataset_id="reanalysis-era5-pressure-levels",
        extractor_name="fixture",
        extraction_method="deterministic",
        options={
            "product_type": ["reanalysis"],
            "variable": ["temperature", "geopotential"],
            "pressure_level": ["500", "850"],
            "year": ["2024"],
            "month": ["01", "02"],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": ["00:00", "12:00"],
            "data_format": ["grib", "netcdf"],
        },
        defaults={
            "product_type": ["reanalysis"],
            "area": [90, -180, -90, 180],
            "data_format": "grib",
        },
        constraints=constraints,
    )


def _contract(
    *,
    field: str = "temperature",
    level: str = "500",
    end_date: str = "2024-01-01",
) -> DatasetContract:
    return DatasetContract(
        dataset_slug="era5_pressure",
        title="Small ERA5 request",
        source_url="https://example.com/era5",
        intent="Retrieve selected pressure-level data.",
        summary="A small deterministic request.",
        fields=[
            ContractFieldSpec(
                name=field,
                selectors=[
                    ContractSelectorSpec(
                        dimension="pressure_level",
                        value=level,
                        unit="hPa",
                    )
                ],
            )
        ],
        scope={
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": end_date,
                "inclusive": True,
            },
            "time": {"timezone": "UTC", "selected_times": ["00:00"]},
        },
    )


class ContractLockingTests(unittest.TestCase):
    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = root / "dataset_inventory.json"
            inventory_path.write_text(_inventory().model_dump_json(indent=2), encoding="utf-8")
            editable = root / "contracts" / "contract.yaml"
            editable.parent.mkdir()
            editable.write_text(
                """
schema_version: dataset_contract.v1
dataset_slug: era5_pressure
dataset_slug: duplicate
title: Example
source_url: https://example.com/era5
intent: Example intent
summary: Example summary
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                lock_dataset_contract(
                    editable_contract_path=editable,
                    inventory_path=inventory_path,
                )

    def test_lock_resolves_defaults_is_canonical_and_versions_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = _inventory()
            inventory_path = root / "dataset_inventory.json"
            inventory_path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
            editable = root / "contracts" / "contract.yaml"
            write_editable_contract(_contract(), editable)

            first = lock_dataset_contract(
                editable_contract_path=editable,
                inventory_path=inventory_path,
                created_at=datetime(2026, 7, 28, tzinfo=UTC),
            )
            repeated = lock_dataset_contract(
                editable_contract_path=editable,
                inventory_path=inventory_path,
            )

            self.assertTrue(first.created)
            self.assertFalse(repeated.created)
            self.assertEqual(first.path, repeated.path)
            self.assertEqual(first.sha256, repeated.sha256)
            self.assertEqual(first.lock.contract.scope["product_type"], "reanalysis")
            self.assertEqual(
                first.lock.contract.scope["geography"]["cds_area"],
                [90, -180, -90, 180],
            )
            self.assertEqual(first.lock.contract.advanced_options["data_format"], "grib")
            serialized = first.path.read_text(encoding="utf-8")
            self.assertEqual(
                content_hash(json.loads(serialized)),
                first.sha256,
            )

            changed = _contract(end_date="2024-01-02")
            write_editable_contract(changed, editable)
            second = lock_dataset_contract(
                editable_contract_path=editable,
                inventory_path=inventory_path,
            )
            self.assertEqual(second.lock.lock_version, 2)
            self.assertEqual(second.path.name, "contract_v2.lock.json")
            self.assertTrue(first.path.exists())

    def test_inventory_values_and_combinations_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = root / "dataset_inventory.json"
            inventory_path.write_text(
                _inventory(combinations=True).model_dump_json(indent=2),
                encoding="utf-8",
            )
            editable = root / "contracts" / "contract.yaml"

            write_editable_contract(_contract(level="850"), editable)
            with self.assertRaisesRegex(ValueError, "not a supported combination"):
                lock_dataset_contract(
                    editable_contract_path=editable,
                    inventory_path=inventory_path,
                )

            write_editable_contract(_contract(field="unknown"), editable)
            with self.assertRaisesRegex(ValueError, "outside the frozen inventory"):
                lock_dataset_contract(
                    editable_contract_path=editable,
                    inventory_path=inventory_path,
                )

    def test_historical_contract_json_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            contract = _contract()
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_contract_run.v1",
                        "contract": contract.model_dump(mode="json"),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_dataset_contract(path), contract)

    def test_secret_values_are_rejected_from_lock_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = root / "dataset_inventory.json"
            inventory_path.write_text(_inventory().model_dump_json(indent=2), encoding="utf-8")
            editable = root / "contracts" / "contract.yaml"
            contract = _contract().model_copy(
                update={"advanced_options": {"api_key": "must-not-be-persisted"}}
            )
            write_editable_contract(contract, editable)

            with self.assertRaisesRegex(ValueError, "store only an environment-variable name"):
                lock_dataset_contract(
                    editable_contract_path=editable,
                    inventory_path=inventory_path,
                )


if __name__ == "__main__":
    unittest.main()
