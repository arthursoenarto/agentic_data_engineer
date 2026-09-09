"""Run one complete Stage A generation-to-evaluation trial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import (  # noqa: E402
    DirectLLMStrategy,
    PipelineGenerationAgent,
    PipelineVariant,
    StagedLLMStrategy,
    TemplateHybridStrategy,
    TerraioDirectStrategy,
    TerraioStagedStrategy,
)
from backend.agents.pipeline_repair import PipelineRepairAgent  # noqa: E402
from backend.llm import LLMClient  # noqa: E402
from backend.orchestration import StageAWorkflowConfig, run_stage_a_workflow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = StageAWorkflowConfig.model_validate_json(
        arguments.config.read_text(encoding="utf-8")
    )
    llm = LLMClient(timeout_seconds=config.llm_timeout_seconds)
    generation = PipelineGenerationAgent(
        {
            PipelineVariant.DIRECT_LLM: DirectLLMStrategy(llm),
            PipelineVariant.STAGED_LLM: StagedLLMStrategy(llm),
            PipelineVariant.TERRAIO_DIRECT: TerraioDirectStrategy(llm),
            PipelineVariant.TERRAIO_STAGED: TerraioStagedStrategy(llm),
            PipelineVariant.TEMPLATE_HYBRID: TemplateHybridStrategy(llm),
        },
        repair_agent=PipelineRepairAgent(llm),
    )
    report, path = run_stage_a_workflow(
        config,
        generation_agent=generation,
    )
    print(f"Stage A status: {report.status}")
    print(f"Pipeline: {report.pipeline_id}")
    print(f"Repair attempts: {report.repair_attempts_used}/{config.max_repair_attempts}")
    print(f"Objective vector complete: {report.objective_vector_complete}")
    print(f"Report: {path}")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
