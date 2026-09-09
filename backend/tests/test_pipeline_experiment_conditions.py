from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.etl_pipeline.conditions import pipeline_condition, strategy_status
from backend.agents.etl_pipeline.agent import ETLPipelineAgent
from backend.agents.etl_pipeline.execution import validate_pipeline_bundle
from backend.agents.etl_pipeline.schemas import (
    PipelineDataModel,
    GeneratedFile,
    GenerationMode,
    PipelineConditionName,
    PipelineGenerationResult,
    PipelineManifest,
    PipelinePolicy,
    PipelineVariant,
    ResearchRole,
)
from backend.agents.etl_pipeline.experiment import (
    PipelineConditionTrialResult,
    summarize_matched_trials,
)
from backend.agents.etl_pipeline.prompt_lock import (
    create_pipeline_prompt_lock,
)
from backend.agents.etl_pipeline.strategies import DirectLLMStrategy, TerraioDirectStrategy
from backend.agents.etl_pipeline.strategy_schemas import FamilyPipelineCompletion


ROOT = Path(__file__).resolve().parents[2]


class RecordingLLM:
    model = "gpt-5.5"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def complete_json(
        self,
        **kwargs: Any,
    ) -> PipelineGenerationResult | FamilyPipelineCompletion:
        self.requests.append(kwargs)
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 25,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 125,
                },
            },
            requested_model=self.model,
        )
        files = [
            GeneratedFile(
                relative_path="pipeline/pipeline_impl.py",
                content="def run_pipeline(**kwargs):\n    return {}\n",
            )
        ]
        if kwargs["response_model"] is FamilyPipelineCompletion:
            return FamilyPipelineCompletion(files=files)
        return PipelineGenerationResult(manifest=PipelineManifest(), files=files)


def _contract() -> DatasetContract:
    return DatasetContract(
        dataset_slug="era5",
        title="ERA5",
        source_url="https://example.com/era5",
        intent="Retrieve selected fields.",
        summary="Fixture contract.",
    )


def _inventory() -> DatasetInventory:
    return DatasetInventory(
        dataset_slug="era5",
        source_url="https://example.com/era5",
        generated_at="2026-07-28T00:00:00+00:00",
        extractor_name="fixture",
        extraction_method="deterministic",
        options={"variable": ["temperature"]},
    )


def _generate(
    strategy: DirectLLMStrategy | TerraioDirectStrategy,
    condition_name: PipelineConditionName,
) -> PipelineGenerationResult:
    condition = pipeline_condition(condition_name)
    return strategy.generate(
        contract=_contract(),
        dataset_dir=Path("project/datasets/era5"),
        prompt_name=condition.prompt_name,
        output_dir="pipelines/era5_condition_test",
        generation_mode=GenerationMode.DATASET_FAMILY,
        inventory=_inventory(),
        policy=PipelinePolicy(),
        pipeline_id="era5_condition_test",
        condition=condition,
    )


class PipelineExperimentConditionTests(unittest.TestCase):
    def test_policy_enums_reject_unsupported_values_before_generation(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            PipelinePolicy(data_model="xarray")  # type: ignore[arg-type]
        self.assertEqual(
            PipelinePolicy().data_model,
            PipelineDataModel.XARRAY_DATASET,
        )

    def test_family_completion_coalesces_only_identical_duplicate_paths(self) -> None:
        path = "pipelines/example/tests/test_offline.py"
        duplicate = GeneratedFile(relative_path=path, content="def test_ok(): pass\n")
        completion = FamilyPipelineCompletion(
            files=[
                GeneratedFile(
                    relative_path="pipelines/example/pipeline_impl.py",
                    content="def run_pipeline(**kwargs): return {}\n",
                ),
                duplicate,
            ],
            tests=[duplicate],
        )
        self.assertEqual(len(completion.files), 2)
        self.assertEqual(completion.tests, [])

        with self.assertRaisesRegex(ValueError, "conflicting content"):
            FamilyPipelineCompletion(
                files=[duplicate],
                tests=[GeneratedFile(relative_path=path, content="different\n")],
            )

    def test_condition_pipeline_persists_a_valid_v4_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "era5"
            dataset_dir.mkdir()
            llm = RecordingLLM()
            agent = ETLPipelineAgent(  # type: ignore[arg-type]
                {PipelineVariant.DIRECT_LLM: DirectLLMStrategy(llm)}
            )
            build = agent.generate(
                condition_name=PipelineConditionName.EXPERT_DIRECT_LLM,
                seed_contract=_contract(),
                inventory=_inventory(),
                dataset_dir=dataset_dir,
                policy=PipelinePolicy(zarr=None, parquet=None),
                pipeline_id="era5_expert_condition_v1",
                prompt_lock=create_pipeline_prompt_lock(lock_id="test-lock"),
            )

            manifest, pipeline_contract = validate_pipeline_bundle(build.pipeline_dir)
            self.assertEqual(manifest.schema_version, "etl_pipeline_manifest.v4")
            self.assertEqual(
                manifest.condition.name,
                PipelineConditionName.EXPERT_DIRECT_LLM,
            )
            self.assertEqual(pipeline_contract.pipeline_id, build.pipeline_id)
            self.assertEqual(
                pipeline_contract.schema_version,
                "etl_pipeline_contract.v3",
            )
            self.assertEqual(
                pipeline_contract.implementation_interface_version,
                "family_pipeline_interface.v3",
            )
            self.assertEqual(pipeline_contract.policy.zarr.format_version, 3)
            self.assertTrue(
                pipeline_contract.policy.zarr.consolidated_metadata
            )

    def test_prompt_lock_drift_fails_before_the_llm_call(self) -> None:
        lock = create_pipeline_prompt_lock(lock_id="test-lock")
        first = lock.conditions[0]
        tampered_source = first.sources[0].model_copy(
            update={"sha256": "0" * 64}
        )
        tampered = lock.model_copy(
            update={
                "conditions": [
                    first.model_copy(
                        update={"sources": [tampered_source, *first.sources[1:]]}
                    ),
                    *lock.conditions[1:],
                ]
            }
        )
        llm = RecordingLLM()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Frozen prompt changed"):
                ETLPipelineAgent(  # type: ignore[arg-type]
                    {PipelineVariant.DIRECT_LLM: DirectLLMStrategy(llm)}
                ).build_condition_pipeline(
                    condition_name=PipelineConditionName.NAIVE_LLM,
                    seed_contract=_contract(),
                    inventory=_inventory(),
                    dataset_dir=Path(temporary),
                    prompt_lock=tampered,
                )
        self.assertEqual(llm.requests, [])

    def test_matched_summary_reports_success_at_1_and_after_repairs(self) -> None:
        trials = [
            PipelineConditionTrialResult(
                experiment_id="experiment",
                condition=PipelineConditionName.NAIVE_LLM,
                repetition=1,
                prompt_lock_id="lock",
                generation_succeeded=True,
                pipeline_id="naive-v1",
                prompt_sources_verified=True,
                initial_acceptance_passed=True,
                final_acceptance_passed=True,
                manifest_path="naive-v1/manifest.json",
                initial_acceptance_report="naive-v1/acceptance.json",
                final_acceptance_report="naive-v1/acceptance.json",
            ),
            PipelineConditionTrialResult(
                experiment_id="experiment",
                condition=PipelineConditionName.EXPERT_DIRECT_LLM,
                repetition=1,
                prompt_lock_id="lock",
                generation_succeeded=True,
                pipeline_id="expert-v1",
                prompt_sources_verified=True,
                repairs_attempted=2,
                repair_final_status="repaired",
                final_acceptance_passed=True,
                manifest_path="expert-v1/manifest.json",
                initial_acceptance_report="expert-v1/initial-acceptance.json",
                repair_log_paths=["expert-v1/repair-log.json"],
                final_acceptance_report="expert-v1/final-acceptance.json",
            ),
            PipelineConditionTrialResult(
                experiment_id="experiment",
                condition=PipelineConditionName.TERRAIO_REFERENCED,
                repetition=1,
                prompt_lock_id="lock",
                generation_succeeded=False,
                generation_error="Provider response was invalid.",
            ),
        ]
        summary = summarize_matched_trials(
            experiment_id="experiment",
            repetitions_per_condition=1,
            trials=trials,
        )
        by_condition = {item.condition: item for item in summary.conditions}
        self.assertEqual(
            by_condition[PipelineConditionName.NAIVE_LLM].success_at_1_rate,
            1.0,
        )
        self.assertEqual(
            by_condition[
                PipelineConditionName.EXPERT_DIRECT_LLM
            ].success_after_le_3_repairs_rate,
            1.0,
        )
        self.assertEqual(
            by_condition[
                PipelineConditionName.TERRAIO_REFERENCED
            ].generation_failure_count,
            1,
        )

    def test_naive_and_expert_are_prompt_variants_over_one_mechanism(self) -> None:
        naive_llm = RecordingLLM()
        expert_llm = RecordingLLM()

        naive = _generate(
            DirectLLMStrategy(naive_llm),  # type: ignore[arg-type]
            PipelineConditionName.NAIVE_LLM,
        )
        expert = _generate(
            DirectLLMStrategy(expert_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM,
        )

        self.assertEqual(naive.manifest.variant, PipelineVariant.DIRECT_LLM)
        self.assertEqual(expert.manifest.variant, PipelineVariant.DIRECT_LLM)
        self.assertIs(
            naive_llm.requests[0]["response_model"],
            FamilyPipelineCompletion,
        )
        self.assertIs(
            expert_llm.requests[0]["response_model"],
            FamilyPipelineCompletion,
        )
        self.assertEqual(naive.manifest.condition.name, PipelineConditionName.NAIVE_LLM)
        self.assertEqual(
            expert.manifest.condition.name,
            PipelineConditionName.EXPERT_DIRECT_LLM,
        )
        naive_instructions = (
            naive_llm.requests[0]["system_prompt"] + naive_llm.requests[0]["user_prompt"]
        ).lower()
        self.assertNotIn("semantic cache", naive_instructions)
        self.assertNotIn("atomic publication", naive_instructions)
        naive_exact_request = naive.manifest.prompt_provenance.calls[0].user_prompt.lower()
        self.assertNotIn("atomic_cache_publication", naive_exact_request)
        self.assertNotIn("semantic cache", naive_exact_request)
        expert_instructions = expert_llm.requests[0]["system_prompt"]
        self.assertIn("source_fixture_manifest.json", expert_instructions)
        self.assertIn("inclusive date-only end bound", expert_instructions)
        self.assertIn("consolidated Zarr v3", expert_instructions)
        self.assertIn("canonical long-form", expert_instructions)
        self.assertIn("Publish outputs atomically", expert_instructions)
        self.assertIn("family_pipeline_interface.v3", naive_instructions)
        self.assertIn('"format_version": 3', naive_instructions)
        self.assertIn('"consolidated_metadata": true', naive_instructions)
        self.assertIn("dataset_artifact_layout.v1", naive_instructions)
        self.assertEqual(
            naive.manifest.prompt_provenance.calls[0].max_output_tokens,
            expert.manifest.prompt_provenance.calls[0].max_output_tokens,
        )

    def test_expert_and_reference_conditions_share_expert_prompt_sources(self) -> None:
        expert_llm = RecordingLLM()
        reference_llm = RecordingLLM()
        expert = _generate(
            DirectLLMStrategy(expert_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM,
        )
        referenced = _generate(
            TerraioDirectStrategy(  # type: ignore[arg-type]
                reference_llm,
                terraio_root=ROOT / "terraio",
            ),
            PipelineConditionName.TERRAIO_REFERENCED,
        )

        self.assertEqual(
            expert_llm.requests[0]["system_prompt"],
            reference_llm.requests[0]["system_prompt"],
        )
        expert_sources = [
            source.model_dump(mode="json")
            for source in expert.manifest.prompt_provenance.sources
        ]
        reference_sources = [
            source.model_dump(mode="json")
            for source in referenced.manifest.prompt_provenance.sources
        ]
        self.assertEqual(expert_sources, reference_sources)
        self.assertNotIn("<reference_file", expert_llm.requests[0]["user_prompt"])
        self.assertIn("<reference_file", reference_llm.requests[0]["user_prompt"])
        self.assertIsNotNone(referenced.manifest.reference_context)
        assert referenced.manifest.reference_context is not None
        self.assertNotIn(
            "terraio/terraio/cds/era5/fetcher.py",
            referenced.manifest.reference_context.selected_file_paths,
        )

    def test_concise_expert_conditions_share_a_shorter_versioned_prompt(self) -> None:
        baseline_llm = RecordingLLM()
        concise_llm = RecordingLLM()
        reference_llm = RecordingLLM()

        _generate(
            DirectLLMStrategy(baseline_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM,
        )
        concise = _generate(
            DirectLLMStrategy(concise_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE,
        )
        referenced = _generate(
            TerraioDirectStrategy(  # type: ignore[arg-type]
                reference_llm,
                terraio_root=ROOT / "terraio",
            ),
            PipelineConditionName.TERRAIO_REFERENCED_CONCISE,
        )

        baseline_prompt = baseline_llm.requests[0]["system_prompt"]
        concise_prompt = concise_llm.requests[0]["system_prompt"]
        self.assertLess(len(concise_prompt), len(baseline_prompt))
        self.assertIn("source_fixture_manifest.json", concise_prompt)
        self.assertIn("consolidated Zarr v3", concise_prompt)
        self.assertIn("canonical long-form", concise_prompt)
        self.assertEqual(concise_prompt, reference_llm.requests[0]["system_prompt"])
        self.assertEqual(
            concise.manifest.condition.name,
            PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE,
        )
        self.assertEqual(
            referenced.manifest.condition.name,
            PipelineConditionName.TERRAIO_REFERENCED_CONCISE,
        )

    def test_trace_revised_concise_conditions_preserve_context_ablation(self) -> None:
        direct_llm = RecordingLLM()
        reference_llm = RecordingLLM()

        direct = _generate(
            DirectLLMStrategy(direct_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V2,
        )
        referenced = _generate(
            TerraioDirectStrategy(  # type: ignore[arg-type]
                reference_llm,
                terraio_root=ROOT / "terraio",
            ),
            PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V2,
        )

        prompt = direct_llm.requests[0]["system_prompt"]
        baseline_prompt = (
            ROOT
            / "backend/agents/etl_pipeline/prompts/shared/expert_family_v2_system.md"
        ).read_text(encoding="utf-8")
        self.assertLess(len(prompt), len(baseline_prompt))
        self.assertIn("multi-file arrays", prompt)
        self.assertIn("without duplicating selector axes", prompt)
        self.assertEqual(prompt, reference_llm.requests[0]["system_prompt"])
        self.assertNotIn("<reference_file", direct_llm.requests[0]["user_prompt"])
        self.assertIn("<reference_file", reference_llm.requests[0]["user_prompt"])
        self.assertEqual(
            direct.manifest.condition.name,
            PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V2,
        )
        self.assertEqual(
            referenced.manifest.condition.name,
            PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V2,
        )

    def test_semantic_trace_revision_remains_short_and_shared(self) -> None:
        direct_llm = RecordingLLM()
        reference_llm = RecordingLLM()

        _generate(
            DirectLLMStrategy(direct_llm),  # type: ignore[arg-type]
            PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V3,
        )
        _generate(
            TerraioDirectStrategy(  # type: ignore[arg-type]
                reference_llm,
                terraio_root=ROOT / "terraio",
            ),
            PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V3,
        )

        prompt = direct_llm.requests[0]["system_prompt"]
        baseline_prompt = (
            ROOT
            / "backend/agents/etl_pipeline/prompts/shared/expert_family_v2_system.md"
        ).read_text(encoding="utf-8")
        self.assertLess(len(prompt), len(baseline_prompt))
        self.assertIn("antimeridian-crossing longitude coverage", prompt)
        self.assertIn("valid CF `<unit> since <origin>`", prompt)
        self.assertEqual(prompt, reference_llm.requests[0]["system_prompt"])

    def test_staged_and_template_statuses_are_not_primary(self) -> None:
        self.assertEqual(
            strategy_status(PipelineVariant.STAGED_LLM).research_role,
            ResearchRole.LEGACY_ABLATION,
        )
        self.assertEqual(
            strategy_status(PipelineVariant.TERRAIO_STAGED).research_role,
            ResearchRole.LEGACY_ABLATION,
        )
        self.assertEqual(
            strategy_status(PipelineVariant.TEMPLATE_HYBRID).research_role,
            ResearchRole.OPTIONAL_ABLATION,
        )


if __name__ == "__main__":
    unittest.main()
