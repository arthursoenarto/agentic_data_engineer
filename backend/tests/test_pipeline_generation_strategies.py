from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.etl_pipeline.reference_context import TERRAIO_REFERENCE_FILES
from backend.agents.etl_pipeline.conditions import pipeline_condition
from backend.agents.etl_pipeline.schemas import (
    GeneratedFile,
    GenerationPromptOverride,
    GenerationMode,
    PipelineGenerationResult,
    PipelineManifest,
    PipelinePolicy,
)
from backend.agents.etl_pipeline.strategies import (
    DirectLLMStrategy,
    StagedLLMStrategy,
    TemplateHybridStrategy,
    TerraioDirectStrategy,
    TerraioStagedStrategy,
)
from backend.agents.etl_pipeline.strategy_schemas import (
    FamilyPipelineCompletion,
    PipelineDesign,
    PipelineDesignFile,
    PipelineReview,
    PipelineReviewIssue,
    TemplateFile,
    TemplatePipelineCompletion,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class QueueLLM:
    model = "gpt-5.5"

    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        response = next(self._responses)
        self.requests.append(kwargs)
        self.assert_response_type(response, kwargs["response_model"])
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 25,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 125,
                },
            },
            requested_model=self.model,
        )
        return response

    @staticmethod
    def assert_response_type(response: Any, response_model: type[Any]) -> None:
        if not isinstance(response, response_model):
            raise AssertionError(
                f"Queued {type(response).__name__}, expected {response_model.__name__}."
            )


def _contract() -> DatasetContract:
    return DatasetContract(
        dataset_slug="era5",
        title="ERA5 pressure levels",
        source_url="https://example.com/era5",
        intent="Retrieve a small pressure-level sample.",
        summary="Temperature and geopotential at selected pressure levels.",
    )


def _design() -> PipelineDesign:
    return PipelineDesign(
        architecture_summary="Small provider client, transform, and validated writer.",
        contract_requirements=["Preserve exact selected fields."],
        provider_requests=["Build one CDS request from explicit field selectors."],
        data_flow=["request", "download", "transform", "write", "validate"],
        file_layout=[
            PipelineDesignFile(
                relative_path="pipeline/run_pipeline.py",
                purpose="Stable entrypoint.",
            )
        ],
        runtime_dependencies=["cdsapi", "xarray", "zarr"],
        validation_plan=["Reopen the output and compare coordinates."],
        reliability_controls=["Use temporary files and atomic replacement."],
    )


def _pipeline_result(label: str) -> PipelineGenerationResult:
    return PipelineGenerationResult(
        manifest=PipelineManifest(),
        files=[
            GeneratedFile(
                relative_path="pipeline/run_pipeline.py",
                content=f"print({label!r})\n",
            ),
            GeneratedFile(
                relative_path="pipeline/requirements.txt",
                content="xarray\n",
            ),
        ],
    )


def _accepted_review() -> PipelineReview:
    return PipelineReview(
        verdict="accept",
        summary="The implementation is coherent and contract-preserving.",
        strengths=["Stable runner and explicit validation."],
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


def _family_result() -> PipelineGenerationResult:
    return PipelineGenerationResult(
        manifest=PipelineManifest(),
        files=[
            GeneratedFile(
                relative_path="pipeline/pipeline_impl.py",
                content="def run_pipeline(**kwargs):\n    return {}\n",
            ),
            GeneratedFile(
                relative_path="pipeline/requirements.txt",
                content="xarray\n",
            ),
        ],
    )


class StagedLLMStrategyTests(unittest.TestCase):
    def test_accept_path_uses_three_calls_and_aggregates_usage(self) -> None:
        llm = QueueLLM([_design(), _pipeline_result("draft"), _accepted_review()])
        strategy = StagedLLMStrategy(llm)  # type: ignore[arg-type]

        result = strategy.generate(
            contract=_contract(),
            dataset_dir=Path("project/datasets/era5"),
        )

        self.assertEqual(len(llm.requests), 3)
        self.assertEqual(
            [request["response_model"] for request in llm.requests],
            [PipelineDesign, PipelineGenerationResult, PipelineReview],
        )
        self.assertIn("print('draft')", result.files[0].content)
        self.assertIn("strategy_calls=3", result.manifest.notes)
        self.assertIn("review_verdict=accept", result.manifest.notes)
        self.assertIsNotNone(result.manifest.llm_usage)
        assert result.manifest.llm_usage is not None
        self.assertEqual(result.manifest.llm_usage.call_count, 3)
        self.assertEqual(result.manifest.llm_usage.total_tokens, 375)

    def test_revision_path_returns_the_full_revised_pipeline(self) -> None:
        review = PipelineReview(
            verdict="revise",
            summary="The draft can reuse incomplete downloads.",
            issues=[
                PipelineReviewIssue(
                    severity="high",
                    category="reliability",
                    relative_path="pipeline/run_pipeline.py",
                    description="Existing raw files are reused without validation.",
                    required_change="Validate completion before reuse.",
                )
            ],
        )
        llm = QueueLLM(
            [
                _design(),
                _pipeline_result("draft"),
                review,
                _pipeline_result("revised"),
            ]
        )
        strategy = StagedLLMStrategy(llm)  # type: ignore[arg-type]

        result = strategy.generate(
            contract=_contract(),
            dataset_dir=Path("project/datasets/era5"),
        )

        self.assertEqual(len(llm.requests), 4)
        self.assertIn("print('revised')", result.files[0].content)
        self.assertIn("strategy_calls=4", result.manifest.notes)
        self.assertIn("review_verdict=revise", result.manifest.notes)
        assert result.manifest.llm_usage is not None
        self.assertEqual(result.manifest.llm_usage.call_count, 4)


class TerraioStagedStrategyTests(unittest.TestCase):
    def test_reference_source_is_sent_only_to_the_design_call(self) -> None:
        llm = QueueLLM([_design(), _pipeline_result("draft"), _accepted_review()])
        strategy = TerraioStagedStrategy(  # type: ignore[arg-type]
            llm,
            terraio_root=REPOSITORY_ROOT / "terraio",
        )

        result = strategy.generate(
            contract=_contract(),
            dataset_dir=Path("project/datasets/era5"),
        )

        self.assertIn("<reference_file", llm.requests[0]["user_prompt"])
        self.assertNotIn("<reference_file", llm.requests[1]["user_prompt"])
        self.assertNotIn("<reference_file", llm.requests[2]["user_prompt"])
        self.assertEqual(
            len(result.manifest.reference_context_files),
            len(TERRAIO_REFERENCE_FILES),
        )
        self.assertGreater(result.manifest.reference_context_size or 0, 0)
        self.assertEqual(result.manifest.variant.value, "terraio_staged")


class TemplateHybridStrategyTests(unittest.TestCase):
    def test_framework_owns_runner_and_manifest_paths(self) -> None:
        completion = TemplatePipelineCompletion(
            runtime_dependencies=["xarray>=2025.1", "pytest>=8"],
            files=[
                TemplateFile(
                    relative_path="pipeline_impl.py",
                    content="def main():\n    return 0\n",
                ),
                TemplateFile(
                    relative_path="provider.py",
                    content="def request():\n    return {}\n",
                ),
            ],
            tests=[
                TemplateFile(
                    relative_path="tests/test_provider.py",
                    content="def test_request():\n    assert True\n",
                )
            ],
            readme="# Generated pipeline\n\nRun `python run_pipeline.py`.",
        )
        llm = QueueLLM([completion])
        strategy = TemplateHybridStrategy(llm)  # type: ignore[arg-type]

        result = strategy.generate(
            contract=_contract(),
            dataset_dir=Path("project/datasets/era5"),
        )

        files = {file.relative_path: file.content for file in result.files}
        self.assertIn("pipeline/run_pipeline.py", files)
        self.assertIn("from pipeline_impl import main", files["pipeline/run_pipeline.py"])
        self.assertIn("pipeline/pipeline_impl.py", files)
        self.assertIn("pipeline/requirements.txt", files)
        self.assertIn("xarray>=2025.1", files["pipeline/requirements.txt"])
        self.assertEqual(result.tests[0].relative_path, "pipeline/tests/test_provider.py")
        self.assertIn("template_scaffold=python_pipeline_v1", result.manifest.notes)
        assert result.manifest.llm_usage is not None
        self.assertEqual(result.manifest.llm_usage.call_count, 1)

    def test_model_cannot_override_framework_runner(self) -> None:
        with self.assertRaises(ValidationError):
            TemplateFile(
                relative_path="run_pipeline.py",
                content="print('override')\n",
            )


class DatasetFamilyStrategyTests(unittest.TestCase):
    def test_direct_strategy_records_frozen_experiment_prompt_override(self) -> None:
        completion = FamilyPipelineCompletion(files=_family_result().files)
        llm = QueueLLM([completion])
        strategy = DirectLLMStrategy(llm)  # type: ignore[arg-type]
        (REPOSITORY_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as directory:
            root = Path(directory)
            system = root / "system.md"
            user = root / "user.md"
            system.write_text("GEPA experiment system prompt\n", encoding="utf-8")
            user.write_text("Generate {pipeline_id} from {contract_json}.\n", encoding="utf-8")
            override = GenerationPromptOverride(
                prompt_id="gepa-p001",
                system_prompt=system,
                user_prompt=user,
                system_sha256=hashlib.sha256(system.read_bytes()).hexdigest(),
                user_sha256=hashlib.sha256(user.read_bytes()).hexdigest(),
            )

            result = strategy.generate(
                contract=_contract(),
                dataset_dir=Path("project/datasets/era5"),
                prompt_name="expert_family_v2",
                output_dir="pipelines/era5_gepa_test",
                generation_mode=GenerationMode.DATASET_FAMILY,
                inventory=_inventory(),
                policy=PipelinePolicy(publication_format="zarr"),
                pipeline_id="era5_gepa_test",
                condition=pipeline_condition("expert_direct_llm"),
                prompt_override=override,
            )

        self.assertEqual(result.manifest.prompt_name, "gepa-p001")
        assert result.manifest.condition is not None
        self.assertEqual(result.manifest.condition.name, "expert_direct_llm")
        self.assertIn("prompt_override=gepa-p001", result.manifest.notes)
        self.assertEqual(llm.requests[0]["system_prompt"], "GEPA experiment system prompt\n")
        assert result.manifest.prompt_provenance is not None
        self.assertEqual(len(result.manifest.prompt_provenance.sources), 2)

    def test_referenced_strategy_records_frozen_experiment_prompt_override(self) -> None:
        completion = FamilyPipelineCompletion(files=_family_result().files)
        llm = QueueLLM([completion])
        strategy = TerraioDirectStrategy(  # type: ignore[arg-type]
            llm,
            terraio_root=REPOSITORY_ROOT / "terraio",
        )
        (REPOSITORY_ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "tmp") as directory:
            root = Path(directory)
            system = root / "system.md"
            user = root / "user.md"
            system.write_text("Referenced experiment system prompt\n", encoding="utf-8")
            user.write_text(
                "Generate {pipeline_id} from {contract_json}.\n{reference_context_block}\n",
                encoding="utf-8",
            )
            override = GenerationPromptOverride(
                prompt_id="expert-workload-v1",
                system_prompt=system,
                user_prompt=user,
                system_sha256=hashlib.sha256(system.read_bytes()).hexdigest(),
                user_sha256=hashlib.sha256(user.read_bytes()).hexdigest(),
            )

            result = strategy.generate(
                contract=_contract(),
                dataset_dir=Path("project/datasets/era5"),
                prompt_name="expert_reference_family_v2",
                output_dir="pipelines/era5_referenced_test",
                generation_mode=GenerationMode.DATASET_FAMILY,
                inventory=_inventory(),
                policy=PipelinePolicy(publication_format="zarr"),
                pipeline_id="era5_referenced_test",
                condition=pipeline_condition("terraio_referenced"),
                prompt_override=override,
            )

        self.assertEqual(result.manifest.prompt_name, "expert-workload-v1")
        assert result.manifest.condition is not None
        self.assertEqual(result.manifest.condition.name, "terraio_referenced")
        self.assertIn("prompt_override=expert-workload-v1", result.manifest.notes)
        self.assertEqual(
            llm.requests[0]["system_prompt"],
            "Referenced experiment system prompt\n",
        )
        self.assertIn("<reference_file", llm.requests[0]["user_prompt"])
        assert result.manifest.prompt_provenance is not None
        self.assertEqual(len(result.manifest.prompt_provenance.sources), 2)

    def test_all_five_strategies_support_family_adapter_prompt(self) -> None:
        policy = PipelinePolicy(publication_format="zarr")
        inventory = _inventory()
        output_dir = "pipelines/era5_family_test"
        completion = TemplatePipelineCompletion(
            runtime_dependencies=["xarray"],
            files=[
                TemplateFile(
                    relative_path="pipeline_impl.py",
                    content="def run_pipeline(**kwargs):\n    return {}\n",
                )
            ],
            readme="# Family adapter",
        )
        cases = [
            (
                DirectLLMStrategy(QueueLLM([_family_result()])),  # type: ignore[arg-type]
                1,
            ),
            (
                StagedLLMStrategy(  # type: ignore[arg-type]
                    QueueLLM([_design(), _family_result(), _accepted_review()])
                ),
                3,
            ),
            (
                TerraioDirectStrategy(  # type: ignore[arg-type]
                    QueueLLM([_family_result()]),
                    terraio_root=REPOSITORY_ROOT / "terraio",
                ),
                1,
            ),
            (
                TerraioStagedStrategy(  # type: ignore[arg-type]
                    QueueLLM([_design(), _family_result(), _accepted_review()]),
                    terraio_root=REPOSITORY_ROOT / "terraio",
                ),
                3,
            ),
            (
                TemplateHybridStrategy(QueueLLM([completion])),  # type: ignore[arg-type]
                1,
            ),
        ]

        for strategy, call_count in cases:
            with self.subTest(variant=strategy.variant):
                result = strategy.generate(
                    contract=_contract(),
                    dataset_dir=Path("project/datasets/era5"),
                    prompt_name="family_adapter_v1",
                    output_dir=output_dir,
                    generation_mode=GenerationMode.DATASET_FAMILY,
                    inventory=inventory,
                    policy=policy,
                    pipeline_id="era5_family_test",
                )
                self.assertEqual(result.manifest.generation_mode, GenerationMode.DATASET_FAMILY)
                self.assertEqual(result.manifest.pipeline_id, "era5_family_test")
                self.assertEqual(result.manifest.fixed_policy, policy)
                self.assertEqual(result.manifest.llm_usage.call_count, call_count)
                self.assertTrue(
                    all(
                        file.relative_path.startswith(f"{output_dir}/")
                        for file in [*result.files, *result.tests]
                    )
                )
                llm = strategy._llm  # type: ignore[attr-defined]
                self.assertIn('"variable": [', llm.requests[0]["user_prompt"])
                self.assertTrue(
                    any(
                        "pipeline_impl.run_pipeline" in request["user_prompt"]
                        for request in llm.requests
                    )
                )


if __name__ == "__main__":
    unittest.main()
