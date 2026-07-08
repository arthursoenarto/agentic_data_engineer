"""ETL Pipeline Agent facade."""

from __future__ import annotations

from pathlib import Path

from backend.access_probes import AccessContext, read_access_context
from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.etl_pipeline.schemas import PipelineGenerationResult, PipelineVariant, pipeline_experiment_slug
from backend.agents.etl_pipeline.strategies import PipelineGenerationStrategy


class ETLPipelineAgent:
    """Build generated ETL pipeline artifacts using a selected strategy."""

    def __init__(self, strategies: dict[PipelineVariant, PipelineGenerationStrategy]) -> None:
        self._strategies = strategies

    def build_pipeline(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        variant: PipelineVariant = PipelineVariant.DIRECT_LLM,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        experimental: bool = False,
        output_dir: str | None = None,
        access_context: AccessContext | None = None,
    ) -> PipelineGenerationResult:
        """Generate ETL pipeline files for a dataset contract."""

        strategy = self._strategies.get(variant)
        if strategy is None:
            raise ValueError(f"No ETL pipeline strategy registered for variant: {variant}")
        resolved_output_dir = output_dir or (pipeline_experiment_slug(variant) if experimental else "pipeline")
        resolved_access_context = access_context or read_access_context(dataset_dir)
        return strategy.generate(
            contract=contract,
            dataset_dir=dataset_dir,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=resolved_output_dir,
            access_context=resolved_access_context,
        )
