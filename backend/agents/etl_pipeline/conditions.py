"""Frozen research conditions layered over pipeline orchestration strategies."""

from __future__ import annotations

from backend.agents.etl_pipeline.schemas import (
    ArtifactTargetKind,
    PipelineConditionMetadata,
    PipelineConditionName,
    PipelineVariant,
    PromptExpertise,
    ReferenceContextMode,
    ResearchRole,
    StrategyStatusMetadata,
)


PIPELINE_CONDITIONS = {
    PipelineConditionName.NAIVE_LLM: PipelineConditionMetadata(
        name=PipelineConditionName.NAIVE_LLM,
        orchestration_variant=PipelineVariant.DIRECT_LLM,
        prompt_name="naive_family_v2",
        prompt_expertise=PromptExpertise.MINIMAL,
        reference_context_mode=ReferenceContextMode.NONE,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.PRIMARY_BASELINE,
    ),
    PipelineConditionName.EXPERT_DIRECT_LLM: PipelineConditionMetadata(
        name=PipelineConditionName.EXPERT_DIRECT_LLM,
        orchestration_variant=PipelineVariant.DIRECT_LLM,
        prompt_name="expert_family_v2",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.NONE,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.PRIMARY_CONDITION,
    ),
    PipelineConditionName.TERRAIO_REFERENCED: PipelineConditionMetadata(
        name=PipelineConditionName.TERRAIO_REFERENCED,
        orchestration_variant=PipelineVariant.TERRAIO_DIRECT,
        prompt_name="expert_reference_family_v2",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.CURATED_TERRAIO,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.PRIMARY_CONTEXT_CONDITION,
    ),
    PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE: PipelineConditionMetadata(
        name=PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE,
        orchestration_variant=PipelineVariant.DIRECT_LLM,
        prompt_name="expert_family_v3",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.NONE,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
    PipelineConditionName.TERRAIO_REFERENCED_CONCISE: PipelineConditionMetadata(
        name=PipelineConditionName.TERRAIO_REFERENCED_CONCISE,
        orchestration_variant=PipelineVariant.TERRAIO_DIRECT,
        prompt_name="expert_reference_family_v3",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.CURATED_TERRAIO,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
    PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V2: PipelineConditionMetadata(
        name=PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V2,
        orchestration_variant=PipelineVariant.DIRECT_LLM,
        prompt_name="expert_family_v4",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.NONE,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
    PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V2: PipelineConditionMetadata(
        name=PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V2,
        orchestration_variant=PipelineVariant.TERRAIO_DIRECT,
        prompt_name="expert_reference_family_v4",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.CURATED_TERRAIO,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
    PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V3: PipelineConditionMetadata(
        name=PipelineConditionName.EXPERT_DIRECT_LLM_CONCISE_V3,
        orchestration_variant=PipelineVariant.DIRECT_LLM,
        prompt_name="expert_family_v5",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.NONE,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
    PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V3: PipelineConditionMetadata(
        name=PipelineConditionName.TERRAIO_REFERENCED_CONCISE_V3,
        orchestration_variant=PipelineVariant.TERRAIO_DIRECT,
        prompt_name="expert_reference_family_v5",
        prompt_expertise=PromptExpertise.EXPERT,
        reference_context_mode=ReferenceContextMode.CURATED_TERRAIO,
        artifact_target_kind=ArtifactTargetKind.STANDALONE_FAMILY_ADAPTER,
        research_role=ResearchRole.OPTIONAL_ABLATION,
    ),
}


STRATEGY_STATUS = {
    PipelineVariant.DIRECT_LLM: StrategyStatusMetadata(
        variant=PipelineVariant.DIRECT_LLM,
        primary_matrix=True,
        research_role=ResearchRole.PRIMARY_CONDITION,
        note="One-call mechanism used by matched naive and expert conditions.",
    ),
    PipelineVariant.TERRAIO_DIRECT: StrategyStatusMetadata(
        variant=PipelineVariant.TERRAIO_DIRECT,
        primary_matrix=True,
        research_role=ResearchRole.PRIMARY_CONTEXT_CONDITION,
        note="One-call mechanism with frozen curated reference context.",
    ),
    PipelineVariant.STAGED_LLM: StrategyStatusMetadata(
        variant=PipelineVariant.STAGED_LLM,
        primary_matrix=False,
        research_role=ResearchRole.LEGACY_ABLATION,
        note="Preserved staged-reasoning ablation; excluded from the primary future matrix.",
    ),
    PipelineVariant.TERRAIO_STAGED: StrategyStatusMetadata(
        variant=PipelineVariant.TERRAIO_STAGED,
        primary_matrix=False,
        research_role=ResearchRole.LEGACY_ABLATION,
        note="Preserved staged reference-context ablation; excluded from the primary matrix.",
    ),
    PipelineVariant.TEMPLATE_HYBRID: StrategyStatusMetadata(
        variant=PipelineVariant.TEMPLATE_HYBRID,
        primary_matrix=False,
        research_role=ResearchRole.OPTIONAL_ABLATION,
        note="Preserved scaffold-enforcement ablation; not a primary condition.",
    ),
}


def pipeline_condition(name: PipelineConditionName | str) -> PipelineConditionMetadata:
    """Return a defensive copy of one frozen standalone condition."""

    resolved = name if isinstance(name, PipelineConditionName) else PipelineConditionName(name)
    return PIPELINE_CONDITIONS[resolved].model_copy(deep=True)


def strategy_status(variant: PipelineVariant) -> StrategyStatusMetadata:
    """Return a defensive copy of one orchestration strategy's research status."""

    return STRATEGY_STATUS[variant].model_copy(deep=True)
