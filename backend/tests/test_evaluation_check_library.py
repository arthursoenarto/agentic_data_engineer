from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from backend.evaluation.check_library import (
    EvaluationProfile,
    MANDATORY_CORE_GATE_IDS,
    bind_declarative_check_instance,
    compile_evaluation_check_plan,
    evaluation_check_catalog,
    load_evaluation_profile,
    validate_emitted_check_ids,
)
from backend.evaluation.objective_v3_schemas import ConstrainedEvaluationConfigV3
from backend.tests.test_evaluation_constrained import _SuiteFixture


class EvaluationCheckLibraryTests(unittest.TestCase):
    def _profile(self) -> EvaluationProfile:
        return EvaluationProfile(
            profile_id="era5-grid-zarr-v3",
            data_class="regular_rectilinear_grid",
            output_format="zarr",
            output_format_version=3,
            workload_kind="full_field_tensor",
            engineering_profile="general_pipeline",
            dataset_tags=[
                "dataset:era5-pressure-levels",
                "domain:earth-science",
            ],
        )

    def test_catalog_is_unique_tagged_and_documented(self) -> None:
        catalog = evaluation_check_catalog()
        ids = [check.check_id for check in catalog]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(check.tags for check in catalog))
        self.assertTrue(all(check.description for check in catalog))

    def test_profile_compiles_core_and_adapter_checks_deterministically(self) -> None:
        plan = compile_evaluation_check_plan(
            self._profile(),
            engineering_enabled=True,
            extensibility_probe_enabled=False,
        )
        ids = {check.check_id for check in plan.checks}
        self.assertIn("contract.grounding", ids)
        self.assertIn("contract.scope_coordinates", ids)
        self.assertIn("contract.zarr_output_policy", ids)
        self.assertIn("contract.pytorch_consumer", ids)
        self.assertIn("engineering.meroda", ids)
        self.assertNotIn("robustness.alternate_contract", ids)
        self.assertIn("dataset:era5-pressure-levels", plan.profile_tags)

        with_probe = compile_evaluation_check_plan(
            self._profile(),
            engineering_enabled=True,
            extensibility_probe_enabled=True,
        )
        self.assertIn(
            "robustness.alternate_contract",
            {check.check_id for check in with_probe.checks},
        )

    def test_station_parquet_profile_selects_station_and_parquet_checks(self) -> None:
        profile = EvaluationProfile(
            profile_id="station-parquet-v1",
            data_class="station_time_series",
            output_format="parquet",
            output_format_version=1,
            workload_kind="filtered_station_scan",
            engineering_profile="general_pipeline",
            dataset_tags=["dataset:example-stations"],
        )
        plan = compile_evaluation_check_plan(
            profile,
            engineering_enabled=True,
            extensibility_probe_enabled=False,
        )
        ids = {check.check_id for check in plan.checks}
        self.assertIn("contract.station_scope_and_keys", ids)
        self.assertIn("contract.parquet_integrity", ids)
        self.assertIn("semantic.station_values_and_missingness", ids)
        self.assertIn("contract.filtered_station_scan", ids)
        self.assertIn("objective.consumer_samples_per_second", ids)
        self.assertNotIn("contract.zarr_output_policy", ids)
        self.assertTrue(MANDATORY_CORE_GATE_IDS.issubset(ids))

    def test_dataset_tags_cannot_select_evaluator_capabilities(self) -> None:
        with self.assertRaisesRegex(ValidationError, "descriptive only"):
            EvaluationProfile(
                profile_id="invalid-profile",
                data_class="regular_rectilinear_grid",
                output_format="zarr",
                output_format_version=3,
                workload_kind="full_field_tensor",
                engineering_profile="general_pipeline",
                dataset_tags=["zarr-v3"],
            )

    def test_yaml_profile_is_typed_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation_profile.yaml"
            path.write_text(
                """\
schema_version: evaluation_profile.v1
profile_id: era5-grid-zarr-v3
data_class: regular_rectilinear_grid
output_format: zarr
output_format_version: 3
workload_kind: full_field_tensor
engineering_profile: general_pipeline
dataset_tags:
  - dataset:era5-pressure-levels
""",
                encoding="utf-8",
            )
            profile = load_evaluation_profile(path)
        self.assertEqual(profile.output_format_version, 3)
        self.assertEqual(profile.dataset_tags, ["dataset:era5-pressure-levels"])

    def test_v3_suite_rejects_profile_and_plan_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _SuiteFixture(Path(temporary))
            payload = json.loads(fixture.config_path.read_text(encoding="utf-8"))
            payload["schema_version"] = "evaluation_constrained.regular_grid_zarr.v3"
            config = ConstrainedEvaluationConfigV3.model_validate(payload)
            frozen = config.model_dump(mode="json")

            wrong_profile = json.loads(json.dumps(frozen))
            wrong_profile["evaluation_profile"]["data_class"] = "station_time_series"
            with self.assertRaisesRegex(ValidationError, "conflicts"):
                ConstrainedEvaluationConfigV3.model_validate(wrong_profile)

            missing_check = json.loads(json.dumps(frozen))
            missing_check["check_plan"]["checks"].pop()
            with self.assertRaisesRegex(ValidationError, "exactly match"):
                ConstrainedEvaluationConfigV3.model_validate(missing_check)

    def test_runtime_check_ids_must_belong_to_the_frozen_plan(self) -> None:
        plan = compile_evaluation_check_plan(
            self._profile(),
            engineering_enabled=True,
            extensibility_probe_enabled=False,
        )
        validate_emitted_check_ids(plan, {"contract.grounding"})
        with self.assertRaisesRegex(ValueError, "outside the frozen plan"):
            validate_emitted_check_ids(plan, {"era5.hidden_candidate_check"})

    def test_declarative_instances_bind_trusted_logic_only(self) -> None:
        instance = bind_declarative_check_instance(
            check_id="contract.filtered_station_scan",
            parameters={"workload": {"station_id": "example"}},
        )
        self.assertEqual(instance.check_id, "contract.filtered_station_scan")
        with self.assertRaisesRegex(ValueError, "trusted catalog checks only"):
            bind_declarative_check_instance(
                check_id="proposal.generated_python",
                parameters={},
            )
        with self.assertRaisesRegex(ValueError, "missing required parameters"):
            bind_declarative_check_instance(
                check_id="contract.filtered_station_scan",
                parameters={},
            )


if __name__ == "__main__":
    unittest.main()
