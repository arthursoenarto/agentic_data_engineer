"""Evaluation-guided proposal agent for immutable pipeline candidates."""

from backend.agents.pipeline_improvement.agent import (
    PipelineImprovementAgent,
    ProposalCall,
    ProposalGenerationError,
)
from backend.agents.pipeline_improvement.schemas import (
    ImprovementEdit,
    ImprovementProposal,
    ImprovementProposalFailureRecord,
    ImprovementProposalRecord,
)

__all__ = [
    "ImprovementEdit",
    "ImprovementProposal",
    "ImprovementProposalFailureRecord",
    "ImprovementProposalRecord",
    "PipelineImprovementAgent",
    "ProposalCall",
    "ProposalGenerationError",
]
