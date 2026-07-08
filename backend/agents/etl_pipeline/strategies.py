"""Swappable ETL pipeline generation strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from backend.access_probes import AccessContext
from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.etl_pipeline.schemas import (
    GeneratedFile,
    PipelineGenerationResult,
    PipelineManifest,
    PipelineVariant,
    contract_hash,
    is_pipeline_output_dir,
)
from backend.llm import LLMClient


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PipelineGenerationStrategy(Protocol):
    """Strategy interface used by ETLPipelineAgent."""

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
    ) -> PipelineGenerationResult:
        _validate_output_dir(output_dir)
        result = self._llm.complete_json(
            system_prompt=_prompt_text(self.variant, prompt_name, "system"),
            user_prompt=_user_prompt(
                variant=self.variant,
                prompt_name=prompt_name,
                contract=contract,
                dataset_dir=dataset_dir,
                allow_network_probe=allow_network_probe,
                output_dir=output_dir,
                access_context=access_context,
            ),
            response_model=PipelineGenerationResult,
            max_output_tokens=16384,
        )
        return _with_manifest_metadata(
            result,
            contract=contract,
            model=getattr(self._llm, "model", "unknown"),
            prompt_name=prompt_name,
            variant=self.variant,
            output_dir=output_dir,
        )


class StagedLLMStrategy:
    """Future variant: internal design, code generation, tests, and repair."""

    variant = PipelineVariant.STAGED_LLM

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
    ) -> PipelineGenerationResult:
        raise NotImplementedError("StagedLLMStrategy is not implemented yet.")


class TerraioDirectStrategy:
    """Future variant: one-call generation with curated read-only Terraio context."""

    variant = PipelineVariant.TERRAIO_DIRECT

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
    ) -> PipelineGenerationResult:
        raise NotImplementedError("TerraioDirectStrategy is not implemented yet.")


class TerraioStagedStrategy:
    """Future variant: staged generation with curated read-only Terraio context."""

    variant = PipelineVariant.TERRAIO_STAGED

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
    ) -> PipelineGenerationResult:
        raise NotImplementedError("TerraioStagedStrategy is not implemented yet.")


class TemplateHybridStrategy:
    """Future variant: deterministic template selection with LLM-filled gaps."""

    variant = PipelineVariant.TEMPLATE_HYBRID

    def generate(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        prompt_name: str = "default",
        allow_network_probe: bool = False,
        output_dir: str = "pipeline",
        access_context: AccessContext | None = None,
    ) -> PipelineGenerationResult:
        raise NotImplementedError("TemplateHybridStrategy is not implemented yet.")


def _prompt_text(variant: PipelineVariant, prompt_name: str, kind: str) -> str:
    path = PROMPTS_DIR / variant.value / f"{prompt_name}_{kind}.md"
    if not path.exists():
        raise RuntimeError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _user_prompt(
    *,
    variant: PipelineVariant,
    prompt_name: str,
    contract: DatasetContract,
    dataset_dir: Path,
    allow_network_probe: bool,
    output_dir: str,
    access_context: AccessContext | None,
) -> str:
    template = _prompt_text(variant, prompt_name, "user")
    return template.format(
        contract_json=json.dumps(contract.model_dump(mode="json"), indent=2),
        access_context_json=json.dumps(
            access_context.model_dump(mode="json") if access_context else None,
            indent=2,
        ),
        dataset_dir=str(dataset_dir),
        output_dir=output_dir,
        allow_network_probe=json.dumps(allow_network_probe),
        response_schema_json=json.dumps(PipelineGenerationResult.model_json_schema(), indent=2),
    )


def _with_manifest_metadata(
    result: PipelineGenerationResult,
    *,
    contract: DatasetContract,
    model: str,
    prompt_name: str,
    variant: PipelineVariant,
    output_dir: str,
) -> PipelineGenerationResult:
    result = _move_result_to_output_dir(result, output_dir)
    generated_files = [file.relative_path for file in [*result.files, *result.tests]]
    manifest = result.manifest.model_copy(
        update={
            "dataset_slug": contract.dataset_slug,
            "variant": variant,
            "prompt_name": prompt_name,
            "model": model,
            "contract_hash": contract_hash(contract),
            "output_dir": output_dir,
            "generated_files": generated_files,
        }
    )
    return result.model_copy(update={"manifest": manifest})


def _validate_output_dir(output_dir: str) -> None:
    if not is_pipeline_output_dir(output_dir):
        raise ValueError("output_dir must be 'pipeline' or a generated pipeline_* experiment slug.")


def _move_result_to_output_dir(result: PipelineGenerationResult, output_dir: str) -> PipelineGenerationResult:
    files = [_move_file_to_output_dir(file, output_dir) for file in result.files]
    tests = [_move_file_to_output_dir(file, output_dir) for file in result.tests]
    return result.model_copy(update={"files": files, "tests": tests})


def _move_file_to_output_dir(file: GeneratedFile, output_dir: str) -> GeneratedFile:
    path = Path(file.relative_path)
    moved_path = Path(output_dir, *path.parts[1:]).as_posix()
    return file.model_copy(update={"relative_path": moved_path})
