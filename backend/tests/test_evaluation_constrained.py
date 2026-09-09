from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from backend.contracts import content_hash
from backend.evaluation.evidence import (
    canonical_json_file_hash,
    frozen_path_hash,
    logical_field_id,
    source_bundle_hash,
)
from backend.evaluation.constrained_schemas import (
    CandidateSourceSpec,
)
from backend.evaluation.constrained_workflow import run_constrained_evaluation_file
from backend.evaluation.schemas import CheckStatus


_RUNNER = r"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def artifact(path):
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    size = 0
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        data = item.read_bytes()
        size += len(data)
        digest.update(data)
    return size, len(files), digest.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--inventory", type=Path, required=True)
parser.add_argument("--cache-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--run-receipt", type=Path, required=True)
parser.add_argument("--mutation", default="none")
args = parser.parse_args()
if args.mutation == "execution_failure":
    print("candidate failed before creating output", file=sys.stderr)
    raise SystemExit(1)
started_at = datetime.now(UTC).isoformat()
started = time.monotonic()
contract = read_json(args.contract)
inventory = read_json(args.inventory)
pipeline_contract = read_json(Path(__file__).parent / "pipeline_contract.json")
manifest = read_json(Path(__file__).parent / "manifest.json")
args.output_dir.mkdir(parents=True)
store_path = args.output_dir / "dataset.zarr"
shutil.copytree(args.cache_dir / "source.zarr", store_path)
group = zarr.open_group(store_path, mode="r+")
mutation = args.mutation
if mutation == "value":
    values = group["observations/signal"][:]
    values.reshape(-1)[0] += 3
    group["observations/signal"][:] = values
elif mutation == "missingness":
    values = group["observations/signal"][:]
    values.reshape(-1)[0] = np.nan
    group["observations/signal"][:] = values
elif mutation == "coordinate_shift":
    values = group["coords/row"][:]
    values[0] += 0.5
    group["coords/row"][:] = values
elif mutation == "coordinate_reverse":
    group["coords/column"][:] = group["coords/column"][:][::-1]
elif mutation == "coordinate_reverse_authorized":
    group["coords/column"][:] = group["coords/column"][:][::-1]
    signal = group["observations/signal"][:]
    quality = group["observations/quality"][:]
    group["observations/signal"][:] = signal[:, :, :, ::-1]
    group["observations/quality"][:] = quality[:, :, ::-1]
elif mutation == "unit":
    group["observations/signal"].attrs["units"] = "wrong_unit"
elif mutation == "fill_value":
    metadata_path = store_path / "observations" / "signal" / ".zarray"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["fill_value"] = -999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
elif mutation == "missing_array":
    del group["observations/signal"]
elif mutation == "missing_coordinate":
    del group["coords/row"]
elif mutation == "static_layout":
    values = group["observations/quality"][0]
    del group["observations/quality"]
    array = group.create_array("observations/quality", data=values, chunks=values.shape)
    array.attrs.update({"_ARRAY_DIMENSIONS": ["row", "column"], "units": "unitless"})
elif mutation == "rerun_difference" and args.run_receipt.parent.name == "rerun":
    values = group["observations/quality"][:]
    values.reshape(-1)[0] += 1
    group["observations/quality"][:] = values
elif mutation == "rerun_metadata" and args.run_receipt.parent.name == "rerun":
    group["observations/quality"].attrs["units"] = "changed_on_rerun"
elif mutation == "leak_secret":
    print(os.environ.get("EVAL_TEST_TOKEN", ""))
elif mutation == "tamper_reference":
    (Path(__file__).parent.parent / "reference" / "candidate-note.txt").write_text("tampered", encoding="utf-8")
if mutation == "corrupt":
    chunk_files = [item for item in (store_path / "observations" / "signal").iterdir() if not item.name.startswith(".")]
    chunk_files[0].write_bytes(b"corrupt-zarr-chunk")
(args.output_dir / "candidate-diagnostics.txt").write_text("not part of the mapped native store", encoding="utf-8")
size, count, digest = artifact(args.output_dir)
manifest_core = dict(manifest)
manifest_core.pop("pipeline_contract_hash", None)
receipt = {
    "schema_version": "etl_pipeline_run.v1",
    "run_id": args.run_receipt.parent.name,
    "pipeline_id": pipeline_contract["pipeline_id"],
    "manifest_hash": canonical(manifest_core),
    "pipeline_contract_hash": canonical(pipeline_contract),
    "contract_lock_hash": canonical(contract),
    "inventory_hash": canonical(inventory),
    "command": [sys.executable, *sys.argv],
    "started_at": started_at,
    "completed_at": datetime.now(UTC).isoformat(),
    "duration_seconds": time.monotonic() - started,
    "exit_code": 0,
    "final_status": "succeeded",
    "cache": {"hits": 2, "misses": 0, "acquired": 0, "reused_keys": ["frozen"], "acquired_keys": []},
    "outputs": [{"artifact_id": pipeline_contract["output_artifact"]["artifact_id"], "path": str(args.output_dir), "storage_format": "zarr", "size_bytes": size, "file_count": count, "sha256": digest}],
    "warnings": [],
    "diagnostics": [],
}
if mutation == "invalid_receipt":
    receipt["inventory_hash"] = "0" * 64
args.run_receipt.write_text(json.dumps(receipt), encoding="utf-8")
"""


class _SuiteFixture:
    def __init__(
        self,
        root: Path,
        *,
        mutation: str = "none",
        names: tuple[str, str, str, str, str, str] = (
            "frame",
            "row",
            "column",
            "tick",
            "northing",
            "easting",
        ),
        candidate_dtype: str = "float32",
        expected_dtype: str | None = "float32",
        assert_units: bool = True,
        mapped_fields: list[str] | None = None,
        value_atol: float = 0.0,
        x_normalization: str = "none",
        datetime_sample: bool = False,
        cf_datetime_sample: bool = False,
        zarr_format: int = 2,
        required_y_units: str | None = None,
    ) -> None:
        self.zarr_format = zarr_format
        self.root = root
        self.mutation = mutation
        candidate_sample, candidate_y, candidate_x, ref_sample, ref_y, ref_x = names
        self.candidate_dims = (candidate_sample, candidate_y, candidate_x)
        self.reference_dims = (ref_sample, ref_y, ref_x)
        self.sample_values = (
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                dtype="datetime64[s]",
            )
            if datetime_sample
            else np.array([0, 1])
        )
        self.sample_attrs = (
            {"units": "hours since 2024-01-01 00:00:00", "calendar": "standard"}
            if cf_datetime_sample
            else {}
        )
        (root / "candidate").mkdir(parents=True)
        (root / "benchmarks").mkdir()
        (root / "cache").mkdir()
        (root / "reference").mkdir()
        (root / "candidate" / "runner.py").write_text(_RUNNER, encoding="utf-8")

        self.fields = [
            {"name": "signal", "selectors": [{"dimension": "level", "value": 10}]},
            {"name": "quality", "selectors": []},
        ]
        inventory = {
            "schema_version": "dataset_inventory.v1",
            "dataset_slug": "generic_grid",
            "title": "Synthetic generic grid",
            "source_url": "https://example.invalid/grid",
            "generated_at": "2026-08-03T00:00:00+00:00",
            "extractor_name": "test_fixture",
            "extraction_method": "deterministic",
            "options": {"variable": ["signal", "quality"], "level": [10]},
            "defaults": {},
            "constraints": {},
            "availability": {},
        }
        self._write_json(root / "inventory.json", inventory)
        contract = {
            "schema_version": "dataset_contract.v1",
            "dataset_slug": "generic_grid",
            "title": "Synthetic generic grid",
            "source_url": "https://example.invalid/grid",
            "intent": "Test a generic regular grid.",
            "summary": "Two logical channels.",
            "fields": self.fields,
            "scope": {},
            "human_confirmed": True,
        }
        lock = {
            "schema_version": "dataset_contract_lock.v1",
            "lock_version": 1,
            "created_at": "2026-08-03T00:00:00+00:00",
            "source_yaml_sha256": "1" * 64,
            "inventory_sha256": content_hash(inventory),
            "contract_schema_version": "dataset_contract.v1",
            "inventory_schema_version": "dataset_inventory.v1",
            "contract": contract,
        }
        self._write_json(root / "contract.lock.json", lock)

        command = [
            "python",
            "runner.py",
            "--contract",
            "{contract_lock_json}",
            "--inventory",
            "{dataset_inventory_json}",
            "--cache-dir",
            "{cache_dir}",
            "--output-dir",
            "{output_dir}",
            "--run-receipt",
            "{pipeline_run_json}",
            "--mutation",
            mutation,
        ]
        manifest_core = {
            "schema_version": "etl_pipeline_manifest.v1",
            "pipeline_id": "generic-candidate",
            "strategy": "fixture",
        }
        pipeline_contract = {
            "schema_version": "etl_pipeline_contract.v1",
            "pipeline_id": "generic-candidate",
            "dataset_slug": "generic_grid",
            "command_template": command,
            "output_artifact": {
                "artifact_id": "generic_grid_dataset",
                "storage_format": "zarr",
            },
            "generation_manifest_reference": "manifest.json",
            "generation_manifest_hash": content_hash(manifest_core),
        }
        if zarr_format == 3:
            pipeline_contract["policy"] = {
                "zarr": {
                    "schema_version": "regular_grid_zarr_output_policy.v1",
                    "dataset_class": "regular_rectilinear_grid",
                    "format_version": 3,
                    "consolidated_metadata": True,
                    "coordinate_metadata": {"sample": {}, "y": {}, "x": {}},
                }
            }
            if required_y_units is not None:
                pipeline_contract["policy"]["zarr"]["coordinate_metadata"]["y"] = {
                    "units": required_y_units
                }
        manifest = {
            **manifest_core,
            "pipeline_contract_hash": content_hash(pipeline_contract),
        }
        self._write_json(
            root / "candidate" / "pipeline_contract.json", pipeline_contract
        )
        self._write_json(root / "candidate" / "manifest.json", manifest)

        logical = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        quality = logical + 100
        self._write_candidate_store(
            root / "cache" / "source.zarr",
            logical.astype(candidate_dtype),
            quality.astype(candidate_dtype),
        )
        self._write_reference_store(
            root / "reference" / "oracle.zarr", logical, quality
        )

        field_ids = [
            logical_field_id(field["name"], field["selectors"]) for field in self.fields
        ]
        selected = mapped_fields if mapped_fields is not None else field_ids
        channel_configs = []
        by_id = {
            field_ids[0]: {
                "candidate": {
                    "store_path": "{output_dir}/dataset.zarr",
                    "array_path": "observations/signal",
                    "selectors": {"layer": 10},
                    "selector_coordinate_paths": {"layer": "coords/layer_values"},
                },
                "reference": {
                    "store_path": "reference/oracle.zarr",
                    "array_path": "truth/primary",
                    "selectors": {"vertical": 10},
                    "selector_coordinate_paths": {"vertical": "axes/vertical_values"},
                },
            },
            field_ids[1]: {
                "candidate": {
                    "store_path": "{output_dir}/dataset.zarr",
                    "array_path": "observations/quality",
                },
                "reference": {
                    "store_path": "reference/oracle.zarr",
                    "array_path": "truth/secondary",
                },
            },
        }
        for field_id in selected:
            item = {"field_id": field_id, **by_id.get(field_id, by_id[field_ids[1]])}
            if expected_dtype is not None:
                item["expected_dtype"] = expected_dtype
            item["expected_fill_value"] = 0
            if assert_units:
                item["metadata"] = {"units": "unitless"}
            channel_configs.append(item)

        candidate_source = CandidateSourceSpec(
            cwd="candidate",
            paths=["candidate/runner.py"],
            sha256="0" * 64,
        )
        candidate_source.sha256 = source_bundle_hash(candidate_source, root)
        config = {
            "schema_version": "evaluation_constrained.regular_grid_zarr.v2",
            "suite_id": f"generic_{mutation}",
            "suite_version": "2026-08-03",
            "candidate_id": "generic-candidate",
            "contract_lock": self._frozen_file("contract.lock.json"),
            "inventory": self._frozen_file("inventory.json"),
            "candidate_source": candidate_source.model_dump(mode="json"),
            "execution": {
                "command": command,
                "timeout_seconds": 30,
                "pipeline_contract": self._frozen_file(
                    "candidate/pipeline_contract.json"
                ),
                "manifest": self._frozen_file("candidate/manifest.json"),
                "frozen_cache": self._frozen_path("cache"),
            },
            "reference": {
                "artifact": self._frozen_path("reference"),
                "content_id": f"sha256:{frozen_path_hash(root / 'reference')}",
                "provenance": "Frozen synthetic oracle produced independently by the test fixture.",
            },
            "grid": {
                "expected_shape": {"sample": 2, "y": 3, "x": 4},
                "candidate_grid": {
                    "dimensions": {
                        "sample": candidate_sample,
                        "y": candidate_y,
                        "x": candidate_x,
                    },
                    "sample_coordinate": {
                        "store_path": "{output_dir}/dataset.zarr",
                        "array_path": "coords/frame",
                    },
                    "y_coordinate": {
                        "store_path": "{output_dir}/dataset.zarr",
                        "array_path": "coords/row",
                    },
                    "x_coordinate": {
                        "store_path": "{output_dir}/dataset.zarr",
                        "array_path": "coords/column",
                        "normalization": x_normalization,
                    },
                },
                "reference_grid": {
                    "dimensions": {"sample": ref_sample, "y": ref_y, "x": ref_x},
                    "sample_coordinate": {
                        "store_path": "reference/oracle.zarr",
                        "array_path": "axes/tick",
                    },
                    "y_coordinate": {
                        "store_path": "reference/oracle.zarr",
                        "array_path": "axes/north",
                    },
                    "x_coordinate": {
                        "store_path": "reference/oracle.zarr",
                        "array_path": "axes/east",
                    },
                },
                "channels": channel_configs,
            },
            "comparison": {
                "max_block_bytes": 1024,
                "mismatch_examples": 3,
                "value_policy": {"atol": value_atol},
            },
            "workload": {"repetitions": 2, "warmup_passes": 1},
            "security": {"secret_environment_variables": ["EVAL_TEST_TOKEN"]},
        }
        if zarr_format == 3:
            config["output_policy"] = {
                "format_version": 3,
                "consolidated_metadata": True,
                "open_latency_repetitions": 2,
            }
            if required_y_units is not None:
                config["output_policy"]["coordinate_metadata"] = {
                    "sample": {},
                    "y": {"units": required_y_units},
                    "x": {},
                }
        if datetime_sample or cf_datetime_sample:
            for side in ("candidate_grid", "reference_grid"):
                coordinate = config["grid"][side]["sample_coordinate"]
                coordinate["expected_step_seconds"] = 3600
                if cf_datetime_sample:
                    coordinate["normalization"] = "cf_datetime"
        self.config_path = root / "benchmarks" / "suite.json"
        self._write_json(self.config_path, config)

    def run(self):
        return run_constrained_evaluation_file(
            self.config_path,
            repository_root=self.root,
            output_dir=self.root / "results" / self.mutation,
        )[0]

    def _write_candidate_store(
        self, path: Path, signal: np.ndarray, quality: np.ndarray
    ) -> None:
        sample, y, x = self.candidate_dims
        group = zarr.open_group(path, mode="w", zarr_format=self.zarr_format)
        self._array(
            group, "coords/frame", self.sample_values, [sample], **self.sample_attrs
        )
        self._array(group, "coords/row", np.array([10.0, 20.0, 30.0]), [y])
        self._array(group, "coords/column", np.array([2.0, 4.0, 6.0, 8.0]), [x])
        self._array(group, "coords/layer_values", np.array([10, 20]), ["layer"])
        stacked = np.stack([signal, signal + 1000], axis=1)
        self._array(
            group,
            "observations/signal",
            stacked,
            [sample, "layer", y, x],
            units="unitless",
        )
        self._array(
            group,
            "observations/quality",
            quality,
            [sample, y, x],
            units="unitless",
        )
        if self.zarr_format == 3:
            zarr.consolidate_metadata(path)

    def _write_reference_store(
        self, path: Path, signal: np.ndarray, quality: np.ndarray
    ) -> None:
        sample, y, x = self.reference_dims
        group = zarr.open_group(path, mode="w", zarr_format=self.zarr_format)
        self._array(
            group, "axes/tick", self.sample_values, [sample], **self.sample_attrs
        )
        self._array(group, "axes/north", np.array([10.0, 20.0, 30.0]), [y])
        self._array(group, "axes/east", np.array([2.0, 4.0, 6.0, 8.0]), [x])
        self._array(group, "axes/vertical_values", np.array([10, 20]), ["vertical"])
        stacked = np.stack([signal, signal + 1000], axis=0)
        self._array(
            group,
            "truth/primary",
            stacked,
            ["vertical", sample, y, x],
            units="unitless",
        )
        self._array(
            group,
            "truth/secondary",
            quality,
            [sample, y, x],
            units="unitless",
        )
        if self.zarr_format == 3:
            zarr.consolidate_metadata(path)

    @staticmethod
    def _array(group, name, data, dimensions, **attrs):
        array = group.create_array(name, data=data, chunks=data.shape)
        array.attrs.update({"_ARRAY_DIMENSIONS": dimensions, **attrs})

    def _frozen_file(self, relative: str) -> dict[str, str]:
        path = self.root / relative
        return {"path": relative, "sha256": canonical_json_file_hash(path)}

    def _frozen_path(self, relative: str) -> dict[str, str]:
        return {"path": relative, "sha256": frozen_path_hash(self.root / relative)}

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class ConstrainedEvaluationTests(unittest.TestCase):
    def _evaluate(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = _SuiteFixture(Path(temporary.name), **kwargs)
        return fixture, fixture.run()

    def test_two_generic_dimension_vocabularies_are_feasible(self) -> None:
        cases = [
            ("frame", "row", "column", "tick", "northing", "easting"),
            ("observation", "scanline", "pixel", "cycle", "v_axis", "u_axis"),
        ]
        for index, names in enumerate(cases):
            with self.subTest(names=names):
                _, run = self._evaluate(names=names, datetime_sample=index == 1)
                self.assertTrue(run.summary.feasible)
                self.assertEqual(
                    set(run.summary.objectives),
                    {
                        "materialization_seconds",
                        "consumer_samples_per_second",
                        "output_bytes",
                    },
                )

    def test_standard_cf_numeric_time_coordinates_are_feasible(self) -> None:
        _, run = self._evaluate(cf_datetime_sample=True)
        self.assertTrue(run.summary.feasible)

    def test_zarr_v3_policy_is_a_gate_and_reports_storage_diagnostics(self) -> None:
        _, run = self._evaluate(zarr_format=3)

        self.assertTrue(run.summary.feasible)
        storage = run.summary.diagnostics["output"]
        self.assertGreater(storage["chunk_bytes"], 0)
        self.assertGreater(storage["metadata_bytes"], 0)
        self.assertGreater(storage["object_count"], 0)
        self.assertEqual(
            run.summary.objectives["output_bytes"],
            storage["chunk_bytes"] + storage["metadata_bytes"],
        )
        self.assertEqual(
            storage["dataset_open_latency_seconds"]["repetitions"],
            2,
        )
        policy_check = next(
            check
            for constraint in run.constraints
            for check in constraint.checks
            if check.check_id == "contract.zarr_output_policy"
        )
        self.assertEqual(policy_check.status, CheckStatus.PASS)

    def test_coordinate_metadata_is_hard_only_when_publicly_required(self) -> None:
        _, optional = self._evaluate(zarr_format=3)
        self.assertTrue(optional.summary.feasible)

        _, required = self._evaluate(
            zarr_format=3,
            required_y_units="degrees_north",
        )
        self.assertFalse(required.summary.feasible)
        self.assertIn(
            "PUBLIC_OUTPUT_POLICY_INVALID",
            [item.code for item in required.summary.feedback],
        )

    def test_evaluator_cannot_strengthen_generation_visible_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SuiteFixture(Path(temporary), zarr_format=3)
            suite = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            suite["output_policy"]["coordinate_metadata"] = {
                "sample": {},
                "y": {"units": "degrees_north"},
                "x": {},
            }
            fixture._write_json(fixture.config_path, suite)
            run = fixture.run()

        self.assertFalse(run.summary.feasible)
        self.assertEqual(run.summary.objectives, {})
        self.assertIn(
            "CONTRACT_GROUNDING_INVALID",
            [item.code for item in run.summary.feedback],
        )

    def test_contract_grounding_rejects_missing_or_substituted_fields(self) -> None:
        full = ["signal[level=10]", "quality"]
        for mapped in ([full[0]], [full[0], "substitute"]):
            with self.subTest(mapped=mapped):
                _, run = self._evaluate(mapped_fields=mapped)
                self.assertFalse(run.summary.feasible)
                self.assertEqual(run.summary.objectives, {})
                self.assertIn(
                    "CONTRACT_GROUNDING_INVALID",
                    [item.code for item in run.summary.feedback],
                )

    def test_scope_coordinate_mutations_fail(self) -> None:
        for mutation in (
            "coordinate_shift",
            "coordinate_reverse",
            "missing_coordinate",
        ):
            with self.subTest(mutation=mutation):
                _, run = self._evaluate(mutation=mutation)
                self.assertFalse(run.summary.feasible)
                codes = {item.code for item in run.summary.feedback}
                self.assertTrue(
                    {"SCOPE_COORDINATE_MISMATCH", "ZARR_INTEGRITY_FAILED"} & codes
                )

        _, authorized = self._evaluate(
            mutation="coordinate_reverse_authorized",
            x_normalization="ascending",
        )
        self.assertTrue(authorized.summary.feasible)

    def test_value_and_missingness_mutations_fail_semantics(self) -> None:
        for mutation in ("value", "missingness"):
            with self.subTest(mutation=mutation):
                _, run = self._evaluate(mutation=mutation)
                self.assertFalse(run.summary.feasible)
                semantic = run.summary.diagnostics["semantic_comparison"]
                self.assertGreater(semantic["mismatch_count"], 0)
                self.assertIn(
                    "SEMANTIC_MISMATCH",
                    [item.code for item in run.summary.feedback],
                )

        _, tolerated = self._evaluate(mutation="value", value_atol=3.0)
        self.assertTrue(tolerated.summary.feasible)

    def test_metadata_assertions_are_conditional(self) -> None:
        _, units = self._evaluate(mutation="unit")
        self.assertFalse(units.summary.feasible)
        self.assertIn(
            "DECLARED_METADATA_MISMATCH", [item.code for item in units.summary.feedback]
        )

        _, dtype = self._evaluate(candidate_dtype="float64", expected_dtype="float32")
        self.assertFalse(dtype.summary.feasible)
        self.assertIn(
            "DECLARED_METADATA_MISMATCH", [item.code for item in dtype.summary.feedback]
        )

        _, fill_value = self._evaluate(mutation="fill_value")
        self.assertFalse(fill_value.summary.feasible)
        self.assertIn(
            "DECLARED_METADATA_MISMATCH",
            [item.code for item in fill_value.summary.feedback],
        )

        _, conditional = self._evaluate(
            candidate_dtype="float64", expected_dtype=None, assert_units=False
        )
        self.assertTrue(conditional.summary.feasible)

        _, unit_not_required = self._evaluate(mutation="unit", assert_units=False)
        self.assertTrue(unit_not_required.summary.feasible)

    def test_missing_and_corrupt_zarr_fail_before_objectives(self) -> None:
        for mutation in ("missing_array", "corrupt"):
            with self.subTest(mutation=mutation):
                _, run = self._evaluate(mutation=mutation)
                self.assertFalse(run.summary.feasible)
                self.assertEqual(run.summary.objectives, {})
                self.assertIn(
                    "ZARR_INTEGRITY_FAILED",
                    [item.code for item in run.summary.feedback],
                )

    def test_execution_failure_without_output_is_a_diagnostic_result(self) -> None:
        _, run = self._evaluate(mutation="execution_failure")
        self.assertFalse(run.summary.feasible)
        self.assertEqual(run.summary.objectives, {})
        self.assertEqual(run.summary.diagnostic_metrics, {})
        self.assertEqual(
            run.summary.diagnostics["candidate_artifact"],
            {
                "execution_succeeded": False,
                "mapped_zarr_readable": False,
                "output_directory_bytes": 0,
                "output_directory_files": 0,
            },
        )

    def test_unsupported_static_layout_is_not_assessed(self) -> None:
        _, run = self._evaluate(mutation="static_layout")
        self.assertFalse(run.summary.feasible)
        self.assertEqual(run.summary.objectives, {})
        self.assertIn(
            "UNSUPPORTED_GRID_CLASS",
            [item.code for item in run.summary.feedback],
        )

    def test_controlled_rerun_fingerprints_detect_change(self) -> None:
        _, stable = self._evaluate()
        self.assertEqual(stable.summary.constraints["rerun_safety"].value, "pass")
        self.assertEqual(
            stable.provenance["logical_output_initial"],
            stable.provenance["logical_output_rerun"],
        )

        _, changed = self._evaluate(mutation="rerun_difference")
        self.assertEqual(changed.summary.constraints["rerun_safety"].value, "fail")
        self.assertIn(
            "RERUN_FINGERPRINT_MISMATCH",
            [item.code for item in changed.summary.feedback],
        )

        _, metadata = self._evaluate(mutation="rerun_metadata")
        self.assertEqual(metadata.summary.constraints["rerun_safety"].value, "fail")
        self.assertIn(
            "RERUN_METADATA_MISMATCH",
            [item.code for item in metadata.summary.feedback],
        )

    def test_invalid_receipt_and_secret_leak_return_safe_feedback(self) -> None:
        _, invalid = self._evaluate(mutation="invalid_receipt")
        self.assertEqual(
            invalid.summary.constraints["provenance_security"].value, "fail"
        )
        self.assertIn(
            "RECEIPT_INVALID", [item.code for item in invalid.summary.feedback]
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SuiteFixture(Path(temporary))
            claim_path = fixture.root / "candidate" / "pipeline_contract.json"
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim["dataset_slug"] = "candidate_claimed_other_dataset"
            fixture._write_json(claim_path, claim)
            suite = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            suite["execution"]["pipeline_contract"]["sha256"] = (
                canonical_json_file_hash(claim_path)
            )
            fixture._write_json(fixture.config_path, suite)
            claim_run = fixture.run()
        self.assertEqual(
            claim_run.summary.constraints["contract_correctness"].value, "pass"
        )
        self.assertEqual(
            claim_run.summary.constraints["provenance_security"].value, "fail"
        )
        self.assertIn(
            "CANDIDATE_CLAIMS_INVALID",
            [item.code for item in claim_run.summary.feedback],
        )

        _, tampered = self._evaluate(mutation="tamper_reference")
        self.assertEqual(
            tampered.summary.constraints["provenance_security"].value, "fail"
        )
        tamper_codes = [item.code for item in tampered.summary.feedback]
        self.assertNotIn("TRUSTED_INPUT_TAMPERED", tamper_codes)
        self.assertIn("INITIAL_EXECUTION_FAILED", tamper_codes)

        secret = "thesis-test-secret-7391"
        with patch.dict(os.environ, {"EVAL_TEST_TOKEN": secret}):
            _, leaked = self._evaluate(mutation="leak_secret")
        serialized = leaked.model_dump_json()
        self.assertNotIn(secret, serialized)
        self.assertEqual(
            leaked.summary.constraints["provenance_security"].value, "fail"
        )
        self.assertIn(
            "SECRET_MATERIAL_DETECTED",
            [item.code for item in leaked.summary.feedback],
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "tree"
            tree.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (tree / "escape.txt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                frozen_path_hash(tree)

    def test_feasible_signal_has_only_frozen_objectives_and_no_weighted_score(
        self,
    ) -> None:
        fixture, feasible = self._evaluate()
        self.assertEqual(feasible.summary.target, "generic-candidate")
        self.assertEqual(
            feasible.summary.objective_directions,
            {
                "materialization_seconds": "minimize",
                "consumer_samples_per_second": "maximize",
                "output_bytes": "minimize",
            },
        )
        self.assertNotIn("score", feasible.summary.model_dump())
        self.assertNotIn("ranking", feasible.summary.model_dump())
        for scenario in feasible.scenarios:
            self.assertTrue((fixture.root / scenario.output_path).is_dir())
            self.assertTrue((fixture.root / scenario.receipt_path).is_file())
        initial_output = fixture.root / feasible.scenarios[0].output_path
        mapped_bytes = sum(
            path.stat().st_size
            for path in (initial_output / "dataset.zarr").rglob("*")
            if path.is_file()
        )
        all_output_bytes = sum(
            path.stat().st_size for path in initial_output.rglob("*") if path.is_file()
        )
        self.assertEqual(feasible.summary.objectives["output_bytes"], mapped_bytes)
        self.assertLess(mapped_bytes, all_output_bytes)
        with self.assertRaises(FileExistsError):
            fixture.run()

        _, infeasible = self._evaluate(mutation="value")
        self.assertEqual(infeasible.summary.objectives, {})
        self.assertNotIn("weighted", infeasible.model_dump_json().lower())


if __name__ == "__main__":
    unittest.main()
