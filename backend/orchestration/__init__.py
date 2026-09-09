"""Top-level workflows that connect otherwise independent subsystems."""

from backend.orchestration.stage_a import (
    StageAWorkflowConfig,
    StageAWorkflowReport,
    run_stage_a_workflow,
)
from backend.orchestration.self_improvement import run_self_improvement
from backend.orchestration.self_improvement_schemas import (
    ObjectiveVector,
    SelfImprovementConfig,
    SelfImprovementRunReport,
)

__all__ = [
    "StageAWorkflowConfig",
    "StageAWorkflowReport",
    "run_stage_a_workflow",
    "ObjectiveVector",
    "SelfImprovementConfig",
    "SelfImprovementRunReport",
    "run_self_improvement",
]
