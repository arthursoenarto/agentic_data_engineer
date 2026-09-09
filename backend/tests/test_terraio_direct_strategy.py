from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.etl_pipeline.reference_context import (
    TERRAIO_ARCHITECTURE_REFERENCE_FILES,
    TERRAIO_REFERENCE_FILES,
    load_terraio_reference_context,
)
from backend.agents.etl_pipeline.schemas import GeneratedFile, PipelineGenerationResult, PipelineManifest
from backend.agents.etl_pipeline.strategies import TerraioDirectStrategy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FakeLLM:
    model = "gpt-5.5"

    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def complete_json(self, **kwargs: Any) -> PipelineGenerationResult:
        self.request = kwargs
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 250},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 40},
                    "total_tokens": 1100,
                },
            },
            requested_model=self.model,
        )
        return PipelineGenerationResult(
            manifest=PipelineManifest(),
            files=[GeneratedFile(relative_path="pipeline/pipeline.py", content="print('ok')\n")],
        )


class TerraioDirectStrategyTests(unittest.TestCase):
    def test_reference_context_contains_only_selected_files(self) -> None:
        context = load_terraio_reference_context(REPOSITORY_ROOT / "terraio")

        self.assertEqual(len(context.selected_file_paths), len(TERRAIO_REFERENCE_FILES))
        self.assertEqual(context.context_size, len(context.rendered_context))
        self.assertEqual(len(context.base_commit), 40)
        self.assertEqual(len(context.context_sha256), 64)
        for path in context.selected_file_paths:
            self.assertIn(f'<reference_file path="{path}">', context.rendered_context)

    def test_reference_context_selects_eac4_module(self) -> None:
        context = load_terraio_reference_context(
            REPOSITORY_ROOT / "terraio",
            dataset_id="cams-global-reanalysis-eac4",
        )

        self.assertIn("terraio/terraio/cds/eac4/fetcher.py", context.selected_file_paths)
        self.assertNotIn("terraio/terraio/cds/era5/fetcher.py", context.selected_file_paths)

    def test_architecture_only_context_excludes_dataset_implementations(self) -> None:
        context = load_terraio_reference_context(
            REPOSITORY_ROOT / "terraio",
            dataset_id="reanalysis-era5-pressure-levels",
            architecture_only=True,
        )

        self.assertEqual(
            len(context.selected_file_paths),
            len(TERRAIO_ARCHITECTURE_REFERENCE_FILES),
        )
        self.assertIn("terraio/terraio/cds/backend/fetcher.py", context.selected_file_paths)
        self.assertNotIn("terraio/terraio/cds/era5/fetcher.py", context.selected_file_paths)
        self.assertNotIn("terraio/terraio/cds/eac4/fetcher.py", context.selected_file_paths)

    def test_strategy_sends_context_and_records_minimal_metadata(self) -> None:
        llm = FakeLLM()
        strategy = TerraioDirectStrategy(llm, terraio_root=REPOSITORY_ROOT / "terraio")  # type: ignore[arg-type]
        contract = DatasetContract(
            dataset_slug="era5",
            title="ERA5 pressure levels",
            source_url="https://example.com/era5",
            intent="Retrieve a small ERA5 sample.",
            summary="A small pressure-level data request.",
        )

        result = strategy.generate(contract=contract, dataset_dir=Path("project/datasets/era5"))

        user_prompt = llm.request["user_prompt"]
        self.assertIn("terraio/terraio/core/storage/store.py", user_prompt)
        self.assertIn("<reference_file", user_prompt)
        self.assertEqual(result.manifest.variant.value, "terraio_direct")
        self.assertEqual(len(result.manifest.reference_context_files), len(TERRAIO_REFERENCE_FILES))
        self.assertGreater(result.manifest.reference_context_size or 0, 0)
        self.assertIsNotNone(result.manifest.llm_usage)
        assert result.manifest.llm_usage is not None
        self.assertEqual(result.manifest.llm_usage.call_count, 1)
        self.assertEqual(result.manifest.llm_usage.input_tokens, 1000)
        self.assertGreater(result.manifest.llm_usage.estimated_cost_usd, 0)
        manifest_json = result.manifest.model_dump(mode="json")
        self.assertEqual(manifest_json["schema_version"], "etl_pipeline_manifest.v4")
        self.assertEqual(
            manifest_json["reference_context"]["base_commit"],
            result.manifest.reference_context.base_commit,
        )
        self.assertEqual(len(manifest_json["prompt_provenance"]["calls"]), 1)
        self.assertEqual(
            manifest_json["llm_usage"]["pricing"][0]["source_url"],
            "https://developers.openai.com/api/docs/models/gpt-5.5",
        )


if __name__ == "__main__":
    unittest.main()
