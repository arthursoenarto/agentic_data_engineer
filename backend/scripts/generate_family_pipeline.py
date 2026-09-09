"""Generate one reusable dataset-family pipeline version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.dataset_inventory.schemas import DatasetInventory  # noqa: E402
from backend.agents.etl_pipeline import (  # noqa: E402
    AcquisitionFormat,
    DirectLLMStrategy,
    PipelineGenerationAgent,
    PipelineConditionName,
    PipelineDataModel,
    PipelinePolicy,
    PipelineVariant,
    PublicationFormat,
    StagedLLMStrategy,
    TemplateHybridStrategy,
    TerraioDirectStrategy,
    TerraioStagedStrategy,
    load_and_validate_pipeline_prompt_lock,
)
from backend.contracts import load_dataset_contract  # noqa: E402
from backend.llm import LLMClient  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reusable pipeline under DATASET_DIR/pipelines/{pipeline_id}/.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-contract",
        type=Path,
        required=True,
        help="A lock JSON, editable YAML, or historical contract.json used as a development case.",
    )
    parser.add_argument(
        "--condition",
        choices=[condition.value for condition in PipelineConditionName],
        help=(
            "Frozen primary research condition. When set, it selects the matched "
            "strategy and prompt variant."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in PipelineVariant],
        default=PipelineVariant.DIRECT_LLM.value,
    )
    parser.add_argument("--prompt-name", default="family_adapter_v1")
    parser.add_argument("--pipeline-id")
    parser.add_argument(
        "--prompt-lock",
        type=Path,
        help="Frozen primary-condition prompt lock; required for matched experiments.",
    )
    parser.add_argument(
        "--acquisition-format",
        choices=[value.value for value in AcquisitionFormat],
        default=AcquisitionFormat.PROVIDER_NATIVE.value,
    )
    parser.add_argument(
        "--publication-format",
        choices=[value.value for value in PublicationFormat],
        default=PublicationFormat.ZARR.value,
    )
    parser.add_argument(
        "--data-model",
        choices=[value.value for value in PipelineDataModel],
        default=PipelineDataModel.XARRAY_DATASET.value,
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    dataset_dir = arguments.dataset_dir.resolve()
    inventory = DatasetInventory.model_validate(
        json.loads((dataset_dir / "dataset_inventory.json").read_text(encoding="utf-8"))
    )
    llm = LLMClient(timeout_seconds=arguments.timeout_seconds)
    strategies = {
        PipelineVariant.DIRECT_LLM: DirectLLMStrategy(llm),
        PipelineVariant.STAGED_LLM: StagedLLMStrategy(llm),
        PipelineVariant.TERRAIO_DIRECT: TerraioDirectStrategy(llm),
        PipelineVariant.TERRAIO_STAGED: TerraioStagedStrategy(llm),
        PipelineVariant.TEMPLATE_HYBRID: TemplateHybridStrategy(llm),
    }
    variant = PipelineVariant(arguments.variant)
    agent = PipelineGenerationAgent(strategies)
    common = {
        "seed_contract": load_dataset_contract(arguments.seed_contract),
        "inventory": inventory,
        "dataset_dir": dataset_dir,
        "pipeline_id": arguments.pipeline_id,
        "policy": PipelinePolicy(
            provider=inventory.provider,
            dataset_id=inventory.dataset_id,
            acquisition_format=arguments.acquisition_format,
            publication_format=arguments.publication_format,
            data_model=arguments.data_model,
        ),
    }
    build = (
        agent.build_condition_pipeline(
            condition_name=PipelineConditionName(arguments.condition),
            prompt_lock=(
                load_and_validate_pipeline_prompt_lock(
                    arguments.prompt_lock.resolve()
                )
                if arguments.prompt_lock is not None
                else None
            ),
            **common,
        )
        if arguments.condition
        else agent.build_family_pipeline(
            variant=variant,
            prompt_name=arguments.prompt_name,
            **common,
        )
    )
    print(f"Pipeline ID: {build.pipeline_id}")
    print(f"Pipeline directory: {build.pipeline_dir}")
    usage = build.generation.manifest.llm_usage
    if usage:
        print(
            f"Generation: {usage.total_tokens} tokens across {usage.call_count} call(s), "
            f"estimated ${usage.estimated_cost_usd:.6f}"
        )
