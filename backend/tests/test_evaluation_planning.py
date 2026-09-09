from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.agents.evaluation_planning import (
    EvaluationPlanningAgent,
    EvaluationPlanningCall,
    EvaluationPlanningDraft,
)
from backend.evaluation.evidence import canonical_json_file_hash
from backend.evaluation.planning import (
    compile_evaluation_planning_run,
    create_evaluation_planning_artifacts,
    load_evaluation_planning_run,
)


class _FakeLLM:
    model = "fake-planner"

    def __init__(self, draft: EvaluationPlanningDraft) -> None:
        self.draft = draft

    def complete_json(self, **_: Any) -> EvaluationPlanningDraft:
        return self.draft


class EvaluationPlanningTests(unittest.TestCase):
    def _draft(
        self,
        *,
        field_id: str = 'temperature[pressure_level="500"]',
    ) -> EvaluationPlanningDraft:
        return EvaluationPlanningDraft.model_validate(
            {
                "compatibility": "compatible",
                "target": {
                    "data_class": "regular_rectilinear_grid",
                    "output_format": "zarr",
                    "output_format_version": 3,
                    "workload_kind": "full_field_tensor",
                    "engineering_profile": "general_pipeline",
                },
                "target_rationale": (
                    "The frozen contract describes sampled gridded atmospheric data."
                ),
                "domain_tags": ["domain:earth-science"],
                "axes": [
                    {
                        "role": "sample",
                        "semantic": "requested valid timestamps",
                        "candidate_name_hints": ["time", "valid_time"],
                        "evidence": "The contract selects dates and UTC times.",
                    },
                    {
                        "role": "y",
                        "semantic": "latitude axis",
                        "candidate_name_hints": ["latitude", "lat"],
                        "evidence": "The contract contains a geographic area.",
                    },
                    {
                        "role": "x",
                        "semantic": "longitude axis",
                        "candidate_name_hints": ["longitude", "lon"],
                        "evidence": "The contract contains a geographic area.",
                    },
                ],
                "fields": [
                    {
                        "field_id": field_id,
                        "candidate_array_hints": ["temperature", "t"],
                        "selector_dimensions": ["pressure_level"],
                        "comparison_mode": "exact_native",
                        "rationale": "Native reanalysis values should remain unchanged.",
                    }
                ],
                "oracle": {
                    "strategy": "independent_provider_materialization",
                    "description": (
                        "Materialize the frozen request independently from candidate code."
                    ),
                    "independence_requirements": [
                        "do not derive expected values from candidate output"
                    ],
                },
                "proposed_checks": [],
            }
        )

    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        dataset = root / "project" / "datasets" / "era5"
        dataset.mkdir(parents=True)
        inventory = dataset / "dataset_inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema_version": "dataset_inventory.v1",
                    "dataset_slug": "era5",
                    "dataset_id": "reanalysis-era5-pressure-levels",
                    "provider": "ECMWF",
                    "options": {"variable": ["temperature"]},
                }
            ),
            encoding="utf-8",
        )
        contract = dataset / "contract.lock.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": "dataset_contract_lock.v1",
                    "inventory_sha256": canonical_json_file_hash(inventory),
                    "contract": {
                        "dataset_slug": "era5",
                        "dataset_family": "reanalysis",
                        "fields": [
                            {
                                "name": "temperature",
                                "selectors": [
                                    {
                                        "dimension": "pressure_level",
                                        "value": "500",
                                    }
                                ],
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        return dataset, inventory, contract

    def _call(self, draft: EvaluationPlanningDraft) -> EvaluationPlanningCall:
        return EvaluationPlanningCall(
            draft=draft,
            model="fake-planner",
            rendered_system_prompt="system",
            rendered_user_prompt="user",
            system_prompt_sha256="a" * 64,
            user_prompt_sha256="b" * 64,
            llm_usage=None,
        )

    def test_deterministic_compiler_resolves_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, inventory, contract = self._inputs(root)
            run = compile_evaluation_planning_run(
                plan_id="era5-plan-v1",
                profile_id="era5-grid-zarr-v3",
                inventory_path=inventory,
                contract_lock_path=contract,
                call=self._call(self._draft()),
                repository_root=root,
                engineering_enabled=True,
                extensibility_probe_enabled=True,
            )
        self.assertFalse(run.suite_ready)
        self.assertTrue(run.planning_complete)
        self.assertTrue(run.outstanding_suite_inputs)
        self.assertEqual(len(run.check_plan.checks), 28)
        self.assertIn("dataset:reanalysis-era5-pressure-levels", run.profile.dataset_tags)
        self.assertIn(
            "robustness.alternate_contract",
            {check.check_id for check in run.check_plan.checks},
        )

    def test_deterministic_compiler_rejects_field_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, inventory, contract = self._inputs(root)
            with self.assertRaisesRegex(ValueError, "match the frozen contract"):
                compile_evaluation_planning_run(
                    plan_id="era5-plan-v1",
                    profile_id="era5-grid-zarr-v3",
                    inventory_path=inventory,
                    contract_lock_path=contract,
                    call=self._call(self._draft(field_id="temperature")),
                    repository_root=root,
                    engineering_enabled=True,
                    extensibility_probe_enabled=False,
                )

    def test_workflow_writes_profile_plan_and_prompt_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _, contract = self._inputs(root)
            run, artifacts = create_evaluation_planning_artifacts(
                agent=EvaluationPlanningAgent(_FakeLLM(self._draft())),
                plan_id="era5-plan-v1",
                dataset_dir=dataset,
                contract_lock_path=contract,
                repository_root=root,
                extensibility_probe_enabled=True,
            )
            loaded = load_evaluation_planning_run(artifacts.planning_run)
            self.assertEqual(loaded, run)
            self.assertTrue(artifacts.evaluation_profile.is_file())
            self.assertTrue(artifacts.llm_draft.is_file())
            proposals = json.loads(artifacts.proposed_checks.read_text())
            self.assertEqual(proposals["proposals"], [])
            with self.assertRaises(FileExistsError):
                create_evaluation_planning_artifacts(
                    agent=EvaluationPlanningAgent(_FakeLLM(self._draft())),
                    plan_id="era5-plan-v1",
                    dataset_dir=dataset,
                    contract_lock_path=contract,
                    repository_root=root,
                    extensibility_probe_enabled=True,
                )

    def test_station_plan_uses_station_mapping_without_grid_axes(self) -> None:
        draft = EvaluationPlanningDraft.model_validate(
            {
                "compatibility": "compatible",
                "target": {
                    "data_class": "station_time_series",
                    "output_format": "parquet",
                    "output_format_version": 1,
                    "workload_kind": "filtered_station_scan",
                    "engineering_profile": "general_pipeline",
                },
                "target_rationale": "The contract selects timestamped station observations.",
                "domain_tags": ["earth-science"],
                "axes": [],
                "station_mapping": {
                    "station_key_semantic": "provider station identifier",
                    "timestamp_semantic": "UTC observation timestamp",
                    "field_semantic": "contract logical field identifier",
                    "value_semantic": "native observation value and missingness",
                    "candidate_column_hints": {
                        "station_id": ["station_id"],
                        "timestamp_utc": ["timestamp_utc", "datetime"],
                        "field_id": ["field_id", "parameter"],
                        "value": ["value"],
                    },
                    "evidence": "The inventory and contract identify stations and dates.",
                },
                "fields": [
                    {
                        "field_id": 'pm10[sensor_id="3919"]',
                        "candidate_array_hints": ["pm10"],
                        "selector_dimensions": ["sensor_id"],
                        "comparison_mode": "exact_native",
                        "rationale": "The source observation must be preserved.",
                    }
                ],
                "oracle": {
                    "strategy": "independent_trusted_reference",
                    "description": "Build a canonical station table outside candidate code.",
                    "independence_requirements": ["use the frozen source fixture"],
                },
            }
        )
        self.assertEqual(draft.target.data_class, "station_time_series")
        self.assertIsNotNone(draft.station_mapping)

    def test_station_plan_rejects_grid_axes(self) -> None:
        payload = self._draft().model_dump(mode="json")
        payload["target"] = {
            "data_class": "station_time_series",
            "output_format": "parquet",
            "output_format_version": 1,
            "workload_kind": "filtered_station_scan",
            "engineering_profile": "general_pipeline",
        }
        with self.assertRaisesRegex(ValueError, "cannot contain grid"):
            EvaluationPlanningDraft.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
