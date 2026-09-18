"""Standalone ERA5 pressure-level dataset-family adapter.

Framework entry point:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from era5_adapter.contract import extract_and_validate_contract
from era5_adapter.fixture import verify_source_fixture
from era5_adapter.publish import build_public_dataset, publish_dataset
from era5_adapter.source import open_fixture_datasets


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize a contract-selected ERA5 regular-grid Zarr v3 dataset.

    `cache_dir` is an external input/cache root. If it contains
    `source_fixture_manifest.json`, all listed raw files are verified and used
    before any provider or credential logic. This adapter intentionally performs
    no provider construction or network access; a complete fixture is the
    authoritative source for deterministic offline execution.
    """
    cache_root = Path(cache_dir)
    out_root = Path(output_dir)

    contract = extract_and_validate_contract(contract_lock, inventory)
    fixture = verify_source_fixture(cache_root)
    datasets = open_fixture_datasets(fixture.paths)
    public_ds, artifact_channels = build_public_dataset(contract, inventory, datasets)
    store_rel = "era5_pressure_levels.zarr"
    artifact = publish_dataset(public_ds, out_root, store_rel, artifact_channels)

    reused_keys = [entry.safe_key for entry in fixture.entries]
    return {
        "cache": {
            "hits": len(reused_keys),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused_keys,
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [],
    }
