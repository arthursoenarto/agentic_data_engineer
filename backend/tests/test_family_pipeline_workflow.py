from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.access_probes import AccessContext, CredentialEnvVar
from backend.agents.contract_drafting.schemas import (
    ContractFieldSpec,
    ContractSelectorSpec,
    DatasetContract,
)
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.etl_pipeline import (
    ETLPipelineAgent,
    GeneratedFile,
    PipelineGenerationResult,
    PipelineManifest,
    PipelinePolicy,
    PipelineVariant,
    pipeline_manifest_hash,
    resolve_pipeline_command,
    stable_json_hash,
    accept_family_pipeline,
    validate_pipeline_bundle,
)
from backend.contracts import canonical_json, lock_dataset_contract, write_editable_contract


PIPELINE_IMPLEMENTATION = r'''
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path


def _key(field, selected_date, times, area):
    payload = {
        "field": field["name"],
        "selectors": field.get("selectors", []),
        "date": selected_date,
        "times": times,
        "area": area,
        "policy": "fixture_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_pipeline(*, contract_lock, inventory, cache_dir, output_dir):
    print(os.environ.get("TEST_DATA_API_KEY", ""))
    contract = contract_lock["contract"]
    scope = contract["scope"]
    start = date.fromisoformat(scope["date_range"]["start_date"])
    end = date.fromisoformat(scope["date_range"]["end_date"])
    times = scope["time"]["selected_times"]
    area = scope["geography"]["cds_area"]
    reused = []
    acquired = []
    rows = []
    current = start
    while current <= end:
        for field in contract["fields"]:
            key = _key(field, current.isoformat(), times, area)
            target = cache_dir / f"{key}.json"
            expected = {"key": key, "field": field, "date": current.isoformat()}
            valid = False
            if target.exists():
                try:
                    valid = json.loads(target.read_text()) == expected
                except (OSError, ValueError):
                    valid = False
            if valid:
                reused.append(key)
            else:
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(expected, sort_keys=True))
                if json.loads(temporary.read_text()) != expected:
                    raise ValueError("cache validation failed")
                temporary.replace(target)
                acquired.append(key)
            rows.append(expected)
        current = date.fromordinal(current.toordinal() + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output_dir / "dataset.json.tmp"
    temporary_output.write_text(json.dumps(rows, sort_keys=True))
    json.loads(temporary_output.read_text())
    temporary_output.replace(output_dir / "dataset.json")
    return {
        "cache": {
            "hits": len(reused),
            "misses": len(acquired),
            "acquired": len(acquired),
            "reused_keys": reused,
            "acquired_keys": acquired,
            "notes": ["fixture semantic cache"],
        }
    }
'''.lstrip()


ZARR_PIPELINE_IMPLEMENTATION = r'''
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import zarr


def run_pipeline(*, contract_lock, inventory, cache_dir, output_dir):
    if os.environ.get("TEST_DATA_API_KEY"):
        raise RuntimeError("acceptance must remove credentials")
    if not (cache_dir / "ready.bin").is_file():
        raise RuntimeError("external prepared cache was not supplied")
    store = output_dir / "dataset.zarr"
    group = zarr.open_group(str(store), mode="w", zarr_format=3)
    def create(name, data, dimension_names):
        if hasattr(group, "create_array"):
            group.create_array(
                name,
                data=data,
                dimension_names=dimension_names,
                overwrite=True,
            )
        else:
            group.create_dataset(
                name,
                data=data,
                shape=data.shape,
                dtype=data.dtype,
                overwrite=True,
            )
            group[name].attrs["_ARRAY_DIMENSIONS"] = list(dimension_names)
    create("time", np.array([0], dtype="int64"), ("time",))
    create("latitude", np.array([52.0, 51.0], dtype="float32"), ("latitude",))
    create("longitude", np.array([-1.0, 0.0], dtype="float32"), ("longitude",))
    create(
        "temperature__pressure_level__500",
        np.arange(4, dtype="float32").reshape(1, 1, 2, 2),
        ("time", "pressure_level", "latitude", "longitude"),
    )
    create("pressure_selector", np.array([500], dtype="int32"), ("pressure_level",))
    zarr.consolidate_metadata(store)
    return {
        "cache": {
            "hits": 1,
            "misses": 0,
            "acquired": 0,
            "reused_keys": ["ready"],
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": "dataset.zarr",
            "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
            "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
            "channels": [
                {
                    "field_id": 'temperature[pressure_level="500"]',
                    "array_path": "temperature__pressure_level__500",
                    "selectors": {},
                    "selector_coordinate_paths": {"pressure_level": "pressure_selector"},
                }
            ],
        },
    }
'''.lstrip()


class FakeFamilyStrategy:
    variant = PipelineVariant.DIRECT_LLM

    def generate(self, **kwargs: Any) -> PipelineGenerationResult:
        output_dir = kwargs["output_dir"]
        return PipelineGenerationResult(
            manifest=PipelineManifest(
                output_dir=output_dir,
                dataset_slug=kwargs["contract"].dataset_slug,
            ),
            files=[
                GeneratedFile(
                    relative_path=f"{output_dir}/pipeline_impl.py",
                    content=PIPELINE_IMPLEMENTATION,
                ),
                GeneratedFile(
                    relative_path=f"{output_dir}/requirements.txt",
                    content="",
                ),
            ],
        )


class FakeZarrFamilyStrategy:
    variant = PipelineVariant.DIRECT_LLM

    def generate(self, **kwargs: Any) -> PipelineGenerationResult:
        output_dir = kwargs["output_dir"]
        return PipelineGenerationResult(
            manifest=PipelineManifest(
                output_dir=output_dir,
                dataset_slug=kwargs["contract"].dataset_slug,
            ),
            files=[
                GeneratedFile(
                    relative_path=f"{output_dir}/pipeline_impl.py",
                    content=ZARR_PIPELINE_IMPLEMENTATION,
                ),
                GeneratedFile(
                    relative_path=f"{output_dir}/requirements.txt",
                    content="numpy>=2,<3\nzarr>=3.1,<4\n",
                ),
            ],
            tests=[
                GeneratedFile(
                    relative_path=f"{output_dir}/tests/test_generated.py",
                    content="def test_generated_contract():\n    assert True\n",
                )
            ],
        )


def _inventory() -> DatasetInventory:
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
            "month": ["01"],
            "day": ["01", "02", "03"],
            "time": ["00:00", "12:00"],
            "data_format": ["grib"],
        },
        defaults={
            "product_type": ["reanalysis"],
            "area": [90, -180, -90, 180],
            "data_format": "grib",
        },
    )


def _contract(
    *,
    field: str,
    level: str,
    start: str,
    end: str,
    times: list[str],
    area: list[float] | None = None,
) -> DatasetContract:
    selected_area = area or [90.0, -180.0, -90.0, 180.0]
    return DatasetContract(
        dataset_slug="era5_pressure",
        title="ERA5 runtime request",
        source_url="https://example.com/era5",
        intent="Retrieve one runtime request.",
        summary="Runtime family-adapter contract.",
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
            "product_type": "reanalysis",
            "date_range": {"start_date": start, "end_date": end, "inclusive": True},
            "time": {"timezone": "UTC", "selected_times": times},
            "geography": {
                "area": "custom",
                "cds_area": selected_area,
                "cds_area_order": ["north", "west", "south", "east"],
            },
        },
        advanced_options={"data_format": "grib"},
    )


class FamilyPipelineWorkflowTests(unittest.TestCase):
    def test_v3_manifest_hash_ignores_additive_v4_fields(self) -> None:
        historical_payload = {
            "schema_version": "etl_pipeline_manifest.v3",
            "dataset_slug": "era5_pressure",
            "variant": "direct_llm",
            "prompt_name": "family_adapter_v1",
            "model": "gpt-5.5",
            "contract_hash": "contract",
            "output_dir": "pipelines/historical",
            "generated_at": "2026-07-28T00:00:00+00:00",
            "generated_files": ["pipelines/historical/pipeline_impl.py"],
            "reference_context_files": [],
            "reference_context_size": None,
            "llm_usage": None,
            "generation_mode": "dataset_family",
            "pipeline_id": "historical",
            "inventory_hash": "inventory",
            "inventory_schema_version": "dataset_inventory.v1",
            "seed_contract_hash": "seed",
            "pipeline_contract_hash": "reciprocal",
            "fixed_policy": None,
            "notes": [],
        }
        manifest = PipelineManifest.model_validate(historical_payload)
        expected = dict(historical_payload)
        expected.pop("pipeline_contract_hash")

        self.assertEqual(pipeline_manifest_hash(manifest), stable_json_hash(expected))

    def test_one_adapter_runs_multiple_locks_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "project" / "datasets" / "era5_pressure"
            dataset_dir.mkdir(parents=True)
            inventory = _inventory()
            inventory_path = dataset_dir / "dataset_inventory.json"
            inventory_path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
            access = AccessContext(
                dataset_slug="era5_pressure",
                provider="fixture",
                verified=True,
                credential_env_vars=[
                    CredentialEnvVar(
                        purpose="fixture",
                        env_var="TEST_DATA_API_KEY",
                    )
                ],
            )
            agent = ETLPipelineAgent(  # type: ignore[arg-type]
                {PipelineVariant.DIRECT_LLM: FakeFamilyStrategy()}
            )
            build = agent.build_family_pipeline(
                seed_contract=_contract(
                    field="temperature",
                    level="500",
                    start="2024-01-01",
                    end="2024-01-01",
                    times=["00:00"],
                ),
                inventory=inventory,
                dataset_dir=dataset_dir,
                pipeline_id="era5_pressure_family_fixture",
                policy=PipelinePolicy(
                    provider="fixture",
                    dataset_id=inventory.dataset_id,
                    acquisition_format="grib",
                    publication_format="json",
                    data_model="records",
                ),
                access_context=access,
            )
            manifest, pipeline_contract = validate_pipeline_bundle(build.pipeline_dir)
            self.assertEqual(manifest.pipeline_id, build.pipeline_id)
            self.assertEqual(pipeline_contract.pipeline_id, build.pipeline_id)

            editable = dataset_dir / "contracts" / "contract.yaml"
            lock_paths: list[Path] = []
            contracts = [
                _contract(
                    field="temperature",
                    level="500",
                    start="2024-01-01",
                    end="2024-01-01",
                    times=["00:00"],
                ),
                _contract(
                    field="temperature",
                    level="500",
                    start="2024-01-01",
                    end="2024-01-02",
                    times=["00:00"],
                ),
                _contract(
                    field="geopotential",
                    level="850",
                    start="2024-01-03",
                    end="2024-01-03",
                    times=["12:00"],
                    area=[60.0, -10.0, 40.0, 10.0],
                ),
            ]
            for contract in contracts:
                write_editable_contract(contract, editable)
                lock_paths.append(
                    lock_dataset_contract(
                        editable_contract_path=editable,
                        inventory_path=inventory_path,
                    ).path
                )

            first = agent.execute_family_pipeline(
                dataset_dir=dataset_dir,
                pipeline_id=build.pipeline_id,
                contract_lock_path=lock_paths[0],
                env_path=dataset_dir / "missing.env",
                env_overrides={"TEST_DATA_API_KEY": "super-secret-value"},
            )
            exact = agent.execute_family_pipeline(
                dataset_dir=dataset_dir,
                pipeline_id=build.pipeline_id,
                contract_lock_path=lock_paths[0],
                env_path=dataset_dir / "missing.env",
                env_overrides={"TEST_DATA_API_KEY": "super-secret-value"},
            )
            extension = agent.execute_family_pipeline(
                dataset_dir=dataset_dir,
                pipeline_id=build.pipeline_id,
                contract_lock_path=lock_paths[1],
                env_path=dataset_dir / "missing.env",
                env_overrides={"TEST_DATA_API_KEY": "super-secret-value"},
            )
            changed = agent.execute_family_pipeline(
                dataset_dir=dataset_dir,
                pipeline_id=build.pipeline_id,
                contract_lock_path=lock_paths[2],
                env_path=dataset_dir / "missing.env",
                env_overrides={"TEST_DATA_API_KEY": "super-secret-value"},
                repair_run_reference=(
                    f"pipelines/{build.pipeline_id}/runs/"
                    "repair-example/repairs/repair_1"
                ),
            )

            self.assertEqual((first.receipt.cache.hits, first.receipt.cache.acquired), (0, 1))
            self.assertEqual((exact.receipt.cache.hits, exact.receipt.cache.acquired), (1, 0))
            self.assertEqual(
                (extension.receipt.cache.hits, extension.receipt.cache.acquired),
                (1, 1),
            )
            self.assertEqual((changed.receipt.cache.hits, changed.receipt.cache.acquired), (0, 1))
            self.assertEqual(
                changed.receipt.repair_run_reference,
                (
                    f"pipelines/{build.pipeline_id}/runs/"
                    "repair-example/repairs/repair_1"
                ),
            )
            for run in (first, exact, extension, changed):
                self.assertTrue(run.succeeded)
                self.assertEqual(
                    run.paths.run_dir.parent,
                    build.pipeline_dir / "runs",
                )
                self.assertEqual(
                    run.paths.output_dir,
                    run.paths.run_dir / "output",
                )
                self.assertTrue(run.paths.receipt.is_file())
                self.assertTrue(run.paths.contract_lock.is_file())
                self.assertTrue(run.paths.stdout.is_file())
                self.assertTrue(run.paths.stderr.is_file())
                self.assertTrue(run.paths.output_dir.is_dir())
                self.assertNotIn(
                    "super-secret-value",
                    run.paths.stdout.read_text(encoding="utf-8"),
                )
                self.assertIn("[REDACTED]", run.paths.stdout.read_text(encoding="utf-8"))
                self.assertNotIn(
                    "super-secret-value",
                    run.paths.receipt.read_text(encoding="utf-8"),
                )

            command = resolve_pipeline_command(
                pipeline_contract,
                contract_lock=lock_paths[0],
                inventory=inventory_path,
                cache_dir=dataset_dir / "data" / "cache",
                output_dir=build.pipeline_dir / "runs" / "example" / "output",
                run_receipt=(
                    build.pipeline_dir
                    / "runs"
                    / "example"
                    / "pipeline_run.json"
                ),
            )
            self.assertIn("--contract", command)
            self.assertFalse(any(part.startswith("{") for part in command))

    def test_invalid_runtime_lock_fails_before_implementation_and_cleans_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "era5_pressure"
            dataset_dir.mkdir()
            inventory = _inventory()
            inventory_path = dataset_dir / "dataset_inventory.json"
            inventory_path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
            seed = _contract(
                field="temperature",
                level="500",
                start="2024-01-01",
                end="2024-01-01",
                times=["00:00"],
            )
            agent = ETLPipelineAgent(  # type: ignore[arg-type]
                {PipelineVariant.DIRECT_LLM: FakeFamilyStrategy()}
            )
            build = agent.build_family_pipeline(
                seed_contract=seed,
                inventory=inventory,
                dataset_dir=dataset_dir,
                pipeline_id="era5_pressure_invalid_fixture",
                policy=PipelinePolicy(publication_format="json"),
            )
            editable = dataset_dir / "contracts" / "contract.yaml"
            write_editable_contract(seed, editable)
            valid_lock = lock_dataset_contract(
                editable_contract_path=editable,
                inventory_path=inventory_path,
            )
            payload = json.loads(valid_lock.path.read_text(encoding="utf-8"))
            payload["contract"]["fields"][0]["name"] = "not_in_inventory"
            invalid_lock = editable.parent / "invalid.lock.json"
            invalid_lock.write_text(canonical_json(payload) + "\n", encoding="utf-8")

            run = agent.execute_family_pipeline(
                dataset_dir=dataset_dir,
                pipeline_id=build.pipeline_id,
                contract_lock_path=invalid_lock,
                env_path=dataset_dir / "missing.env",
            )

            self.assertFalse(run.succeeded)
            self.assertFalse(run.paths.output_dir.exists())
            self.assertIn(
                "outside the frozen inventory",
                run.paths.stderr.read_text(encoding="utf-8"),
            )
            self.assertEqual(run.receipt.cache.acquired, 0)

    def test_model_cannot_replace_framework_owned_family_files(self) -> None:
        for protected_path in (
            "run_pipeline.py",
            "pipeline_contract.json",
            "manifest.json",
            "pipeline_run.json",
        ):
            class UnsafeStrategy(FakeFamilyStrategy):
                def generate(self, **kwargs: Any) -> PipelineGenerationResult:
                    result = super().generate(**kwargs)
                    return result.model_copy(
                        update={
                            "files": [
                                *result.files,
                                GeneratedFile(
                                    relative_path=(
                                        f"{kwargs['output_dir']}/{protected_path}"
                                    ),
                                    content="unsafe\n",
                                ),
                            ]
                        }
                    )

            with self.subTest(protected_path=protected_path):
                with tempfile.TemporaryDirectory() as temporary:
                    dataset_dir = Path(temporary) / "era5_pressure"
                    dataset_dir.mkdir()
                    inventory = _inventory()
                    agent = ETLPipelineAgent(  # type: ignore[arg-type]
                        {PipelineVariant.DIRECT_LLM: UnsafeStrategy()}
                    )
                    with self.assertRaisesRegex(ValueError, "framework-owned"):
                        agent.build_family_pipeline(
                            seed_contract=_contract(
                                field="temperature",
                                level="500",
                                start="2024-01-01",
                                end="2024-01-01",
                                times=["00:00"],
                            ),
                            inventory=inventory,
                            dataset_dir=dataset_dir,
                            pipeline_id="era5_pressure_unsafe_fixture",
                            policy=PipelinePolicy(publication_format="json"),
                        )

    def test_generation_acceptance_uses_external_read_only_cache_and_real_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "era5_pressure"
            dataset_dir.mkdir()
            inventory = _inventory()
            inventory_path = dataset_dir / "dataset_inventory.json"
            inventory_path.write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
            contract = _contract(
                field="temperature",
                level="500",
                start="2024-01-01",
                end="2024-01-01",
                times=["00:00"],
            )
            editable = dataset_dir / "contracts" / "contract.yaml"
            write_editable_contract(contract, editable)
            lock = lock_dataset_contract(
                editable_contract_path=editable,
                inventory_path=inventory_path,
            )
            access = AccessContext(
                dataset_slug="era5_pressure",
                provider="fixture",
                verified=True,
                credential_env_vars=[
                    CredentialEnvVar(purpose="fixture", env_var="TEST_DATA_API_KEY")
                ],
            )
            agent = ETLPipelineAgent(  # type: ignore[arg-type]
                {PipelineVariant.DIRECT_LLM: FakeZarrFamilyStrategy()}
            )
            build = agent.build_family_pipeline(
                seed_contract=contract,
                inventory=inventory,
                dataset_dir=dataset_dir,
                pipeline_id="era5_pressure_zarr_acceptance",
                policy=PipelinePolicy(
                    provider="fixture",
                    dataset_id=inventory.dataset_id,
                    acquisition_format="grib",
                    publication_format="zarr",
                    data_model="xarray_dataset",
                ),
                access_context=access,
            )
            prepared_cache = dataset_dir / "prepared_cache"
            prepared_cache.mkdir()
            (prepared_cache / "ready.bin").write_bytes(b"provider-native-fixture")

            previous = os.environ.get("TEST_DATA_API_KEY")
            os.environ["TEST_DATA_API_KEY"] = "must-not-reach-candidate"
            try:
                report, report_path = accept_family_pipeline(
                    dataset_dir=dataset_dir,
                    pipeline_id=build.pipeline_id,
                    contract_lock_path=lock.path,
                    prepared_cache_dir=prepared_cache,
                    env_path=dataset_dir / "missing.env",
                )
            finally:
                if previous is None:
                    os.environ.pop("TEST_DATA_API_KEY", None)
                else:
                    os.environ["TEST_DATA_API_KEY"] = previous

            self.assertEqual(
                report.final_status,
                "passed",
                report.model_dump_json(indent=2),
            )
            self.assertTrue(report_path.is_file())
            self.assertEqual(
                report.generated_source_sha256_before,
                report.generated_source_sha256_after,
            )
            self.assertEqual(
                report.frozen_cache_sha256_before,
                report.frozen_cache_sha256_after,
            )
            self.assertEqual(
                {check.status for check in report.checks},
                {"pass"},
            )
            self.assertEqual(
                report.dataset_artifact.channels[0].field_id,
                'temperature[pressure_level="500"]',
            )
            policy_check = next(
                check
                for check in report.checks
                if check.check_id == "generation.zarr_v3_consolidated"
            )
            self.assertEqual(policy_check.status, "pass")
            self.assertEqual(
                policy_check.evidence["initial"]["zarr_format"],
                3,
            )
            exact_bytes = next(
                check
                for check in report.checks
                if check.check_id == "generation.exact_output_rerun"
            )
            self.assertFalse(exact_bytes.required)


if __name__ == "__main__":
    unittest.main()
