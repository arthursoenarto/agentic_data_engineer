"""Swappable ETL pipeline generation strategies."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from backend.access_probes import AccessContext
from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.etl_pipeline.reference_context import (
    ReferenceContext,
    load_terraio_reference_context,
)
from backend.agents.etl_pipeline.prompt_provenance import (
    PromptRecorder,
    recorded_complete_json,
)
from backend.agents.etl_pipeline.conditions import strategy_status
from backend.agents.etl_pipeline.schemas import (
    GeneratedFile,
    GenerationPromptOverride,
    GenerationMode,
    PipelineConditionMetadata,
    PipelineGenerationResult,
    PipelineManifest,
    PipelinePolicy,
    PipelineVariant,
    PromptProvenance,
    contract_hash,
    is_pipeline_output_dir,
    stable_json_hash,
)
from backend.agents.etl_pipeline.strategy_schemas import (
    FamilyPipelineCompletion,
    PipelineDesign,
    PipelineReview,
    TemplatePipelineCompletion,
)
from backend.llm import LLMClient, LLMUsageSummary, LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class PipelineGenerationStrategy(Protocol):
    """Strategy interface used by PipelineGenerationAgent."""

    variant: PipelineVariant

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> PipelineGenerationResult:
        ...


class DirectLLMStrategy:
    """Simple one-call baseline: DatasetContract directly to pipeline files."""

    variant = PipelineVariant.DIRECT_LLM

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> PipelineGenerationResult:
        _validate_generation_inputs(
            output_dir=output_dir,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
        )
        _validate_condition(condition, self.variant, prompt_name, generation_mode)
        usage_tracker = LLMUsageTracker()
        recorder = PromptRecorder(
            model=getattr(self._llm, "model", "unknown"),
            reasoning_effort=getattr(self._llm, "reasoning_effort", None),
        )
        if prompt_override is None:
            system_path = _prompt_path(self.variant, prompt_name, "system")
            user_path = _prompt_path(self.variant, prompt_name, "user")
            effective_prompt_name = prompt_name
        else:
            system_path, user_path = prompt_override.validated_sources()
            effective_prompt_name = prompt_override.prompt_id
        recorder.add_source(system_path)
        recorder.add_source(user_path)
        system_prompt = system_path.read_text(encoding="utf-8")
        user_prompt = _user_prompt(
            variant=self.variant,
            prompt_name=prompt_name,
            contract=contract,
            dataset_dir=dataset_dir,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            template_path=user_path,
        )
        response_model = (
            FamilyPipelineCompletion
            if condition is not None
            else PipelineGenerationResult
        )
        completion = recorded_complete_json(
            llm=self._llm,
            recorder=recorder,
            call_name="generation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_output_tokens=16384,
            usage_tracker=usage_tracker,
        )
        result = (
            PipelineGenerationResult(
                manifest=PipelineManifest(),
                files=completion.files,
                tests=completion.tests,
            )
            if isinstance(completion, FamilyPipelineCompletion)
            else completion
        )
        return _with_manifest_metadata(
            result,
            contract=contract,
            model=getattr(self._llm, "model", "unknown"),
            prompt_name=effective_prompt_name,
            variant=self.variant,
            output_dir=output_dir,
            llm_usage=usage_tracker.summary(),
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            condition=condition,
            prompt_provenance=recorder.provenance(),
            notes=(
                [
                    f"prompt_override={prompt_override.prompt_id}",
                    f"base_prompt_name={prompt_name}",
                ]
                if prompt_override is not None
                else None
            ),
        )


class StagedLLMStrategy:
    """Design, generate, review, and conditionally revise a pipeline."""

    variant = PipelineVariant.STAGED_LLM

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
    ) -> PipelineGenerationResult:
        return _run_staged_strategy(
            llm=self._llm,
            variant=self.variant,
            contract=contract,
            dataset_dir=dataset_dir,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            condition=condition,
        )


class TerraioDirectStrategy:
    """One-call generation with curated read-only TerraIO source context."""

    variant = PipelineVariant.TERRAIO_DIRECT

    def __init__(self, llm: LLMClient, terraio_root: Path | None = None) -> None:
        self._llm = llm
        self._terraio_root = terraio_root or REPOSITORY_ROOT / "terraio"

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> PipelineGenerationResult:
        _validate_generation_inputs(
            output_dir=output_dir,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
        )
        _validate_condition(condition, self.variant, prompt_name, generation_mode)
        reference_context = load_terraio_reference_context(
            self._terraio_root,
            dataset_id=inventory.dataset_id if inventory is not None else None,
            architecture_only=condition is not None,
        )
        usage_tracker = LLMUsageTracker()
        recorder = PromptRecorder(
            model=getattr(self._llm, "model", "unknown"),
            reasoning_effort=getattr(self._llm, "reasoning_effort", None),
        )
        if prompt_override is None:
            system_path = _prompt_path(self.variant, prompt_name, "system")
            user_path = _prompt_path(self.variant, prompt_name, "user")
            effective_prompt_name = prompt_name
        else:
            system_path, user_path = prompt_override.validated_sources()
            effective_prompt_name = prompt_override.prompt_id
        recorder.add_source(system_path)
        recorder.add_source(user_path)
        system_prompt = system_path.read_text(encoding="utf-8")
        user_prompt = _user_prompt(
            variant=self.variant,
            prompt_name=prompt_name,
            contract=contract,
            dataset_dir=dataset_dir,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            reference_context=reference_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            template_path=user_path,
        )
        response_model = (
            FamilyPipelineCompletion
            if condition is not None
            else PipelineGenerationResult
        )
        completion = recorded_complete_json(
            llm=self._llm,
            recorder=recorder,
            call_name="generation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_output_tokens=16384,
            usage_tracker=usage_tracker,
        )
        result = (
            PipelineGenerationResult(
                manifest=PipelineManifest(),
                files=completion.files,
                tests=completion.tests,
            )
            if isinstance(completion, FamilyPipelineCompletion)
            else completion
        )
        return _with_manifest_metadata(
            result,
            contract=contract,
            model=getattr(self._llm, "model", "unknown"),
            prompt_name=effective_prompt_name,
            variant=self.variant,
            output_dir=output_dir,
            reference_context=reference_context,
            llm_usage=usage_tracker.summary(),
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            condition=condition,
            prompt_provenance=recorder.provenance(),
            notes=(
                [
                    f"prompt_override={prompt_override.prompt_id}",
                    f"base_prompt_name={prompt_name}",
                ]
                if prompt_override is not None
                else None
            ),
        )


class TerraioStagedStrategy:
    """Staged generation whose design call receives curated TerraIO context."""

    variant = PipelineVariant.TERRAIO_STAGED

    def __init__(self, llm: LLMClient, terraio_root: Path | None = None) -> None:
        self._llm = llm
        self._terraio_root = terraio_root or REPOSITORY_ROOT / "terraio"

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
    ) -> PipelineGenerationResult:
        reference_context = load_terraio_reference_context(
            self._terraio_root,
            dataset_id=inventory.dataset_id if inventory is not None else None,
        )
        return _run_staged_strategy(
            llm=self._llm,
            variant=self.variant,
            contract=contract,
            dataset_dir=dataset_dir,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            reference_context=reference_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            condition=condition,
        )


class TemplateHybridStrategy:
    """Framework-owned executable scaffold with LLM-filled implementation slots."""

    variant = PipelineVariant.TEMPLATE_HYBRID

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
        generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
        inventory: DatasetInventory | None = None,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        condition: PipelineConditionMetadata | None = None,
    ) -> PipelineGenerationResult:
        _validate_generation_inputs(
            output_dir=output_dir,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
        )
        _validate_condition(condition, self.variant, prompt_name, generation_mode)
        usage_tracker = LLMUsageTracker()
        recorder = PromptRecorder(
            model=getattr(self._llm, "model", "unknown"),
            reasoning_effort=getattr(self._llm, "reasoning_effort", None),
        )
        system_path = _prompt_path(self.variant, prompt_name, "system")
        user_path = _prompt_path(self.variant, prompt_name, "user")
        recorder.add_source(system_path)
        recorder.add_source(user_path)
        system_prompt = system_path.read_text(encoding="utf-8")
        user_prompt = _template_user_prompt(
            variant=self.variant,
            prompt_name=prompt_name,
            contract=contract,
            dataset_dir=dataset_dir,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
        )
        completion = recorded_complete_json(
            llm=self._llm,
            recorder=recorder,
            call_name="generation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=TemplatePipelineCompletion,
            max_output_tokens=16384,
            usage_tracker=usage_tracker,
        )
        dependencies = list(dict.fromkeys(completion.runtime_dependencies))
        files = [
            *(
                [
                    GeneratedFile(
                        relative_path=f"{output_dir}/run_pipeline.py",
                        content=_template_runner(),
                    )
                ]
                if generation_mode is GenerationMode.CONTRACT_SPECIALIZED
                else []
            ),
            GeneratedFile(
                relative_path=f"{output_dir}/requirements.txt",
                content="\n".join(dependencies) + "\n",
            ),
            GeneratedFile(
                relative_path=f"{output_dir}/.gitignore",
                content=_template_gitignore(),
            ),
            GeneratedFile(
                relative_path=f"{output_dir}/README.md",
                content=completion.readme.rstrip() + "\n",
            ),
            *[
                GeneratedFile(
                    relative_path=f"{output_dir}/{file.relative_path}",
                    content=file.content,
                )
                for file in completion.files
            ],
        ]
        tests = [
            GeneratedFile(
                relative_path=f"{output_dir}/{file.relative_path}",
                content=file.content,
            )
            for file in completion.tests
        ]
        result = PipelineGenerationResult(
            manifest=PipelineManifest(),
            files=files,
            tests=tests,
        )
        return _with_manifest_metadata(
            result,
            contract=contract,
            model=getattr(self._llm, "model", "unknown"),
            prompt_name=prompt_name,
            variant=self.variant,
            output_dir=output_dir,
            llm_usage=usage_tracker.summary(),
            notes=[
                "strategy_calls=1",
                (
                    "template_scaffold=dataset_family_v1"
                    if generation_mode is GenerationMode.DATASET_FAMILY
                    else "template_scaffold=python_pipeline_v1"
                ),
            ],
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
            condition=condition,
            prompt_provenance=recorder.provenance(),
        )


def _run_staged_strategy(
    *,
    llm: LLMClient,
    variant: PipelineVariant,
    contract: DatasetContract,
    dataset_dir: Path,
    prompt_name: str,
    allow_network_probe: bool,
    output_dir: str,
    access_context: AccessContext | None,
    reference_context: ReferenceContext | None = None,
    generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
    inventory: DatasetInventory | None = None,
    policy: PipelinePolicy | None = None,
    pipeline_id: str | None = None,
    condition: PipelineConditionMetadata | None = None,
) -> PipelineGenerationResult:
    _validate_generation_inputs(
        output_dir=output_dir,
        generation_mode=generation_mode,
        inventory=inventory,
        policy=policy,
        pipeline_id=pipeline_id,
    )
    _validate_condition(condition, variant, prompt_name, generation_mode)
    usage_tracker = LLMUsageTracker()
    recorder = PromptRecorder(
        model=getattr(llm, "model", "unknown"),
        reasoning_effort=getattr(llm, "reasoning_effort", None),
    )
    prompt_kinds = [
        "system",
        "design_user",
        "implementation_user",
        "review_user",
        "revision_user",
    ]
    for kind in prompt_kinds:
        recorder.add_source(_prompt_path(variant, prompt_name, kind))
    system_prompt = _prompt_text(variant, prompt_name, "system")
    common_values = _prompt_values(
        contract=contract,
        dataset_dir=dataset_dir,
        allow_network_probe=allow_network_probe,
        output_dir=output_dir,
        access_context=access_context,
        reference_context=reference_context,
        generation_mode=generation_mode,
        inventory=inventory,
        policy=policy,
        pipeline_id=pipeline_id,
    )

    design_prompt = _render_prompt(
        variant,
        prompt_name,
        "design_user",
        **common_values,
    )
    design = recorded_complete_json(
        llm=llm,
        recorder=recorder,
        call_name="design",
        system_prompt=system_prompt,
        user_prompt=design_prompt,
        response_model=PipelineDesign,
        max_output_tokens=6000,
        usage_tracker=usage_tracker,
    )
    design_json = json.dumps(design.model_dump(mode="json"), indent=2)
    implementation_prompt = _render_prompt(
        variant,
        prompt_name,
        "implementation_user",
        **common_values,
        design_json=design_json,
    )
    draft = recorded_complete_json(
        llm=llm,
        recorder=recorder,
        call_name="implementation",
        system_prompt=system_prompt,
        user_prompt=implementation_prompt,
        response_model=PipelineGenerationResult,
        max_output_tokens=16384,
        usage_tracker=usage_tracker,
    )
    draft_json = json.dumps(draft.model_dump(mode="json"), indent=2)
    review_prompt = _render_prompt(
        variant,
        prompt_name,
        "review_user",
        **common_values,
        design_json=design_json,
        draft_json=draft_json,
    )
    review = recorded_complete_json(
        llm=llm,
        recorder=recorder,
        call_name="review",
        system_prompt=system_prompt,
        user_prompt=review_prompt,
        response_model=PipelineReview,
        max_output_tokens=6000,
        usage_tracker=usage_tracker,
    )

    result = draft
    call_count = 3
    if review.verdict == "revise":
        revision_prompt = _render_prompt(
            variant,
            prompt_name,
            "revision_user",
            **common_values,
            design_json=design_json,
            draft_json=draft_json,
            review_json=json.dumps(review.model_dump(mode="json"), indent=2),
        )
        result = recorded_complete_json(
            llm=llm,
            recorder=recorder,
            call_name="revision",
            system_prompt=system_prompt,
            user_prompt=revision_prompt,
            response_model=PipelineGenerationResult,
            max_output_tokens=16384,
            usage_tracker=usage_tracker,
        )
        call_count += 1

    return _with_manifest_metadata(
        result,
        contract=contract,
        model=getattr(llm, "model", "unknown"),
        prompt_name=prompt_name,
        variant=variant,
        output_dir=output_dir,
        reference_context=reference_context,
        llm_usage=usage_tracker.summary(),
        notes=[
            f"strategy_calls={call_count}",
            f"review_verdict={review.verdict}",
            f"review_issue_count={len(review.issues)}",
        ],
        generation_mode=generation_mode,
        inventory=inventory,
        policy=policy,
        pipeline_id=pipeline_id,
        condition=condition,
        prompt_provenance=recorder.provenance(),
    )


def _prompt_path(variant: PipelineVariant, prompt_name: str, kind: str) -> Path:
    shared_prefix = {
        PipelineVariant.DIRECT_LLM: "expert_family_",
        PipelineVariant.TERRAIO_DIRECT: "expert_reference_family_",
    }.get(variant)
    if shared_prefix is not None and prompt_name.startswith(shared_prefix):
        version = prompt_name.rsplit("_", 1)[-1]
        path = PROMPTS_DIR / "shared" / f"expert_family_{version}_{kind}.md"
    else:
        path = PROMPTS_DIR / variant.value / f"{prompt_name}_{kind}.md"
    if not path.exists():
        raise RuntimeError(f"Prompt file not found: {path}")
    return path


def _prompt_text(variant: PipelineVariant, prompt_name: str, kind: str) -> str:
    return _prompt_path(variant, prompt_name, kind).read_text(encoding="utf-8")


def _render_prompt(
    variant: PipelineVariant,
    prompt_name: str,
    kind: str,
    **values: str | int,
) -> str:
    return _prompt_text(variant, prompt_name, kind).format(**values)


def _user_prompt(
    *,
    variant: PipelineVariant,
    prompt_name: str,
    contract: DatasetContract,
    dataset_dir: Path,
    allow_network_probe: bool,
    output_dir: str,
    access_context: AccessContext | None,
    reference_context: ReferenceContext | None = None,
    generation_mode: GenerationMode,
    inventory: DatasetInventory | None,
    policy: PipelinePolicy | None,
    pipeline_id: str | None,
    template_path: Path | None = None,
) -> str:
    values = _prompt_values(
            contract=contract,
            dataset_dir=dataset_dir,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            reference_context=reference_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
    )
    values["response_schema_json"] = json.dumps(
        PipelineGenerationResult.model_json_schema(), indent=2
    )
    if template_path is not None:
        return template_path.read_text(encoding="utf-8").format(**values)
    return _render_prompt(variant, prompt_name, "user", **values)


def _template_user_prompt(
    *,
    variant: PipelineVariant,
    prompt_name: str,
    contract: DatasetContract,
    dataset_dir: Path,
    allow_network_probe: bool,
    output_dir: str,
    access_context: AccessContext | None,
    generation_mode: GenerationMode,
    inventory: DatasetInventory | None,
    policy: PipelinePolicy | None,
    pipeline_id: str | None,
) -> str:
    return _render_prompt(
        variant,
        prompt_name,
        "user",
        **_prompt_values(
            contract=contract,
            dataset_dir=dataset_dir,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=access_context,
            generation_mode=generation_mode,
            inventory=inventory,
            policy=policy,
            pipeline_id=pipeline_id,
        ),
    )


def _prompt_values(
    *,
    contract: DatasetContract,
    dataset_dir: Path,
    allow_network_probe: bool,
    output_dir: str,
    access_context: AccessContext | None,
    reference_context: ReferenceContext | None = None,
    generation_mode: GenerationMode,
    inventory: DatasetInventory | None,
    policy: PipelinePolicy | None,
    pipeline_id: str | None,
) -> dict[str, str | int]:
    interface = _execution_interface(policy)
    return {
        "contract_json": json.dumps(contract.model_dump(mode="json"), indent=2),
        "access_context_json": json.dumps(
            access_context.model_dump(mode="json") if access_context else None,
            indent=2,
        ),
        "dataset_dir": str(dataset_dir),
        "output_dir": output_dir,
        "allow_network_probe": json.dumps(allow_network_probe),
        "generation_mode": generation_mode.value,
        "inventory_json": json.dumps(
            inventory.model_dump(mode="json") if inventory else None,
            indent=2,
        ),
        "policy_json": json.dumps(
            policy.model_dump(mode="json") if policy else None,
            indent=2,
        ),
        "minimal_output_policy_json": json.dumps(
            (
                {
                    "provider": policy.provider,
                    "dataset_id": policy.dataset_id,
                    "acquisition_format": policy.acquisition_format,
                    "publication_format": policy.publication_format,
                    "data_model": policy.data_model,
                    "zarr": (
                        policy.zarr.model_dump(mode="json")
                        if policy.zarr is not None
                        else (
                            {"format_version": 3, "consolidated_metadata": True}
                            if policy.publication_format == "zarr"
                            else None
                        )
                    ),
                    "parquet": (
                        policy.parquet.model_dump(mode="json")
                        if policy.parquet is not None
                        else None
                    ),
                }
                if policy
                else None
            ),
            indent=2,
        ),
        "pipeline_id": pipeline_id or "",
        "execution_interface_json": json.dumps(interface, indent=2),
        "reference_context": reference_context.rendered_context if reference_context else "",
        "reference_context_files_json": json.dumps(
            list(reference_context.selected_file_paths) if reference_context else [],
            indent=2,
        ),
        "reference_context_size": reference_context.context_size if reference_context else 0,
        "reference_context_block": (
            "\n".join(
                [
                    "A frozen curated TerraIO reference context is provided below.",
                    "Use it for engineering lessons only; the generated adapter must remain standalone.",
                    f"Base commit: {reference_context.base_commit}",
                    f"Selected paths: {json.dumps(list(reference_context.selected_file_paths))}",
                    "",
                    reference_context.rendered_context,
                ]
            )
            if reference_context
            else "No reference source-code context is provided for this condition."
        ),
    }


def _execution_interface(policy: PipelinePolicy | None) -> dict[str, object]:
    common: dict[str, object] = {
        "implementation_callable": (
            "pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)"
        ),
        "framework_owned_paths": [
            "run_pipeline.py",
            "pipeline_contract.json",
            "manifest.json",
            "pipeline_run.json",
        ],
        "required_cli": [
            "--contract",
            "--inventory",
            "--cache-dir",
            "--output-dir",
            "--run-receipt",
        ],
        "cache_dir_semantics": (
            "cache_dir is a framework-authorized external root. A "
            "source_fixture_manifest.json at its root describes evaluator-owned raw "
            "inputs. Read those files before any provider construction or credential "
            "check. Never write into a read-only fixture cache."
        ),
        "source_fixture_manifest": {
            "schema_version": "source_fixture_manifest.v1",
            "location": "cache_dir/source_fixture_manifest.json",
            "entries": [
                {
                    "entry_id": "stable source object ID",
                    "relative_path": "path relative to cache_dir",
                    "source": "provider URL or repository-local provenance path",
                    "size_bytes": "positive integer",
                    "sha256": "lowercase SHA-256",
                }
            ],
            "rule": (
                "Read manifest.entries and resolve each relative_path beneath cache_dir; "
                "verify size_bytes and sha256 before decoding."
            ),
        },
        "canonical_field_id": (
            "Use field name alone when selectors are empty; otherwise use "
            "name[dimension=JSON-quoted-string-value,...] in contract selector order."
        ),
    }
    if policy is not None and policy.publication_format == "parquet":
        parquet = (
            policy.parquet.model_dump(mode="json")
            if policy.parquet is not None
            else {
                "dataset_class": "station_time_series",
                "compression": "zstd",
                "required_columns": [
                    "station_id",
                    "timestamp",
                    "field_id",
                    "value",
                ],
                "primary_key": ["station_id", "timestamp", "field_id"],
            }
        )
        return {
            **common,
            "schema_version": "family_pipeline_interface.v4",
            "parquet_output_policy": parquet,
            "implementation_return": {
                "cache": {
                    "hits": "nonnegative integer",
                    "misses": "nonnegative integer",
                    "acquired": "nonnegative integer",
                    "reused_keys": ["secret-safe cache key"],
                    "acquired_keys": ["secret-safe cache key"],
                },
                "dataset_artifact": {
                    "schema_version": "dataset_artifact_layout.tabular.v1",
                    "storage_format": "parquet",
                    "file_path": "relative/path/to/observations.parquet",
                    "columns": parquet["required_columns"],
                    "primary_key": parquet["primary_key"],
                },
                "warnings": ["optional secret-safe warning"],
            },
        }
    zarr = (
        policy.zarr.model_dump(mode="json")
        if policy is not None and policy.zarr is not None
        else {"format_version": 3, "consolidated_metadata": True}
    )
    return {
        **common,
        "schema_version": "family_pipeline_interface.v3",
        "zarr_output_policy": zarr,
        "implementation_return": {
            "cache": {
                "hits": "nonnegative integer",
                "misses": "nonnegative integer",
                "acquired": "nonnegative integer",
                "reused_keys": ["secret-safe cache key"],
                "acquired_keys": ["secret-safe cache key"],
            },
            "dataset_artifact": {
                "schema_version": "dataset_artifact_layout.v1",
                "storage_format": "zarr",
                "store_path": "relative/path/to/store.zarr",
                "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
                "coordinates": {
                    "sample": "time",
                    "y": "latitude",
                    "x": "longitude",
                },
                "channels": [
                    {
                        "field_id": "canonical requested field-selector ID",
                        "array_path": "relative Zarr data array path",
                        "selectors": {},
                        "selector_coordinate_paths": {},
                    }
                ],
            },
            "warnings": ["optional secret-safe warning"],
        },
    }


def _with_manifest_metadata(
    result: PipelineGenerationResult,
    *,
    contract: DatasetContract,
    model: str,
    prompt_name: str,
    variant: PipelineVariant,
    output_dir: str,
    llm_usage: LLMUsageSummary,
    reference_context: ReferenceContext | None = None,
    notes: list[str] | None = None,
    generation_mode: GenerationMode = GenerationMode.CONTRACT_SPECIALIZED,
    inventory: DatasetInventory | None = None,
    policy: PipelinePolicy | None = None,
    pipeline_id: str | None = None,
    condition: PipelineConditionMetadata | None = None,
    prompt_provenance: PromptProvenance | None = None,
) -> PipelineGenerationResult:
    result = _move_result_to_output_dir(result, output_dir)
    generated_files = [file.relative_path for file in [*result.files, *result.tests]]
    manifest = result.manifest.model_copy(
        update={
            "schema_version": (
                "etl_pipeline_manifest.v4"
                if prompt_provenance is not None or condition is not None
                else (
                    "etl_pipeline_manifest.v3"
                    if generation_mode is GenerationMode.DATASET_FAMILY
                    else "etl_pipeline_manifest.v2"
                )
            ),
            "dataset_slug": contract.dataset_slug,
            "variant": variant,
            "prompt_name": prompt_name,
            "model": model,
            "contract_hash": contract_hash(contract),
            "output_dir": output_dir,
            "generated_at": datetime.now(UTC).isoformat(),
            "generated_files": generated_files,
            "reference_context_files": (
                list(reference_context.selected_file_paths) if reference_context else []
            ),
            "reference_context_size": reference_context.context_size if reference_context else None,
            "reference_context": (
                reference_context.provenance() if reference_context else None
            ),
            "condition": condition,
            "strategy_status": strategy_status(variant),
            "prompt_provenance": prompt_provenance,
            "llm_usage": llm_usage,
            "generation_mode": generation_mode,
            "pipeline_id": pipeline_id,
            "inventory_hash": stable_json_hash(inventory) if inventory else None,
            "inventory_schema_version": inventory.schema_version if inventory else None,
            "seed_contract_hash": (
                contract_hash(contract)
                if generation_mode is GenerationMode.DATASET_FAMILY
                else None
            ),
            "fixed_policy": policy,
            "notes": [*result.manifest.notes, *(notes or [])],
        }
    )
    return result.model_copy(update={"manifest": manifest})


def _template_runner() -> str:
    return '''"""Framework-owned executable entrypoint for the generated pipeline."""

from pipeline_impl import main


if __name__ == "__main__":
    raise SystemExit(main() or 0)
'''


def _template_gitignore() -> str:
    return """.env
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
"""


def _validate_output_dir(output_dir: str) -> None:
    if not is_pipeline_output_dir(output_dir):
        raise ValueError("output_dir must be 'pipeline' or a generated pipeline_* experiment slug.")


def _validate_generation_inputs(
    *,
    output_dir: str,
    generation_mode: GenerationMode,
    inventory: DatasetInventory | None,
    policy: PipelinePolicy | None,
    pipeline_id: str | None,
) -> None:
    if generation_mode is GenerationMode.CONTRACT_SPECIALIZED:
        _validate_output_dir(output_dir)
        return
    if inventory is None or policy is None or pipeline_id is None:
        raise ValueError(
            "Dataset-family generation requires inventory, policy, and pipeline_id."
        )
    if output_dir != f"pipelines/{pipeline_id}":
        raise ValueError("Dataset-family output_dir must be pipelines/{pipeline_id}.")


def _validate_condition(
    condition: PipelineConditionMetadata | None,
    variant: PipelineVariant,
    prompt_name: str,
    generation_mode: GenerationMode,
) -> None:
    if condition is None:
        return
    if generation_mode is not GenerationMode.DATASET_FAMILY:
        raise ValueError("Controlled standalone conditions require dataset-family generation.")
    if condition.orchestration_variant is not variant:
        raise ValueError("Condition orchestration variant does not match the selected strategy.")
    if condition.prompt_name != prompt_name:
        raise ValueError("Condition prompt name does not match the selected prompt.")


def _move_result_to_output_dir(result: PipelineGenerationResult, output_dir: str) -> PipelineGenerationResult:
    files = [_move_file_to_output_dir(file, output_dir) for file in result.files]
    tests = [_move_file_to_output_dir(file, output_dir) for file in result.tests]
    return result.model_copy(update={"files": files, "tests": tests})


def _move_file_to_output_dir(file: GeneratedFile, output_dir: str) -> GeneratedFile:
    path = Path(file.relative_path)
    output_parts = Path(output_dir).parts
    if path.parts[: len(output_parts)] == output_parts:
        return file
    if path.parts and path.parts[0] == "pipelines" and len(path.parts) >= 3:
        tail = path.parts[2:]
    else:
        tail = path.parts[1:]
    moved_path = Path(output_dir, *tail).as_posix()
    return file.model_copy(update={"relative_path": moved_path})
