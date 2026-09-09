"""Pipeline Generation Agent facade."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from backend.access_probes import AccessContext, read_access_context
from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.etl_pipeline.conditions import pipeline_condition
from backend.agents.etl_pipeline.execution import (
    PipelineExecutionRun,
    execute_family_pipeline,
    validate_pipeline_bundle,
)
from backend.agents.etl_pipeline.paths import DatasetPipelinePaths
from backend.agents.etl_pipeline.prompt_lock import (
    PipelinePromptLock,
    validate_manifest_prompt_lock,
    validate_pipeline_prompt_lock,
)
from backend.agents.etl_pipeline.schemas import (
    GenerationMode,
    GenerationPromptOverride,
    OutputArtifactContract,
    PipelineContract,
    PipelineConditionMetadata,
    PipelineConditionName,
    PipelineGenerationResult,
    PipelineManifest,
    PipelinePolicy,
    PipelineVariant,
    ParquetPublicationPolicy,
    ZarrPublicationPolicy,
    is_pipeline_output_dir,
    pipeline_contract_hash,
    pipeline_experiment_slug,
    pipeline_manifest_hash,
    pipeline_version_id,
    stable_json_hash,
)
from backend.agents.etl_pipeline.strategies import PipelineGenerationStrategy
from backend.agents.pipeline_repair import (
    MAX_REPAIR_ATTEMPTS,
    PipelineExecutionResult,
    PipelineRepairAgent,
    PipelineRepairLog,
)
from backend.env import ENV_FILE


FAMILY_RUNNER_TEMPLATE = Path(__file__).resolve().parent / "templates" / "family_runner.py"
FAMILY_PROTECTED_PATHS = {
    "manifest.json",
    "pipeline_contract.json",
    "pipeline_run.json",
    "run_pipeline.py",
}


@dataclass(frozen=True)
class PipelineBuildRun:
    """Generated pipeline artifacts plus their execution/repair outcome."""

    generation: PipelineGenerationResult
    pipeline_dir: Path
    repair_log: PipelineRepairLog
    repair_log_path: Path

    @property
    def verified(self) -> bool:
        return self.repair_log.final_status in {"already_succeeded", "repaired"}


@dataclass(frozen=True)
class FamilyPipelineBuild:
    """One immutable reusable pipeline version and its framework interface."""

    generation: PipelineGenerationResult
    pipeline_contract: PipelineContract
    pipeline_dir: Path

    @property
    def pipeline_id(self) -> str:
        return self.pipeline_contract.pipeline_id


@dataclass(frozen=True)
class PipelineAgentContextPolicy:
    """Explicit context and memory boundary for generation and repair calls."""

    generation_context: tuple[str, ...] = (
        "dataset_contract",
        "dataset_inventory",
        "access_context",
        "pipeline_policy",
        "strategy_prompts",
        "reference_context",
    )
    repair_context: tuple[str, ...] = (
        "current_pipeline_artifacts",
        "execution_command",
        "execution_failure",
        "immutable_execution_inputs",
        "previous_repair_error",
    )
    persistent_memory: str = "artifact_backed"
    shared_conversational_memory: bool = False


class PipelineGenerationAgent:
    """Generate and repair pipelines through role-specialized LLM operations.

    The facade deliberately shares no conversational state between generation
    and repair. Durable artifacts and execution evidence are the only memory
    transferred across the two operations.
    """

    def __init__(
        self,
        strategies: dict[PipelineVariant, PipelineGenerationStrategy],
        repair_agent: PipelineRepairAgent | None = None,
    ) -> None:
        self._strategies = strategies
        self._repair_agent = repair_agent
        self._context_policy = PipelineAgentContextPolicy()

    @property
    def context_policy(self) -> PipelineAgentContextPolicy:
        """Describe the context and memory available to each role."""

        return self._context_policy

    def generate(
        self,
        *,
        seed_contract: DatasetContract,
        inventory: DatasetInventory,
        dataset_dir: Path,
        variant: PipelineVariant = PipelineVariant.DIRECT_LLM,
        prompt_name: str = "family_adapter_v1",
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        allow_network_probe: bool = False,
        access_context: AccessContext | None = None,
        condition_name: PipelineConditionName | str | None = None,
        prompt_lock: PipelinePromptLock | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> FamilyPipelineBuild:
        """Generate one reusable pipeline through a direct or controlled condition."""

        if condition_name is not None:
            return self.build_condition_pipeline(
                condition_name=condition_name,
                seed_contract=seed_contract,
                inventory=inventory,
                dataset_dir=dataset_dir,
                policy=policy,
                pipeline_id=pipeline_id,
                allow_network_probe=allow_network_probe,
                access_context=access_context,
                prompt_lock=prompt_lock,
                prompt_override=prompt_override,
            )
        if prompt_lock is not None:
            raise ValueError("A prompt lock requires a controlled generation condition.")
        if prompt_override is not None:
            raise ValueError("A prompt override requires a controlled generation condition.")
        return self.build_family_pipeline(
            seed_contract=seed_contract,
            inventory=inventory,
            dataset_dir=dataset_dir,
            variant=variant,
            prompt_name=prompt_name,
            policy=policy,
            pipeline_id=pipeline_id,
            allow_network_probe=allow_network_probe,
            access_context=access_context,
        )

    def repair(
        self,
        *,
        pipeline_dir: Path,
        command: Sequence[str],
        prompt_name: str = "default",
        max_attempts: int = MAX_REPAIR_ATTEMPTS,
        timeout_seconds: int = 3600,
        env_path: Path = ENV_FILE,
        env_overrides: Mapping[str, str] | None = None,
        initial_execution: PipelineExecutionResult | None = None,
        repair_root: Path | None = None,
        protected_paths: set[str] | None = None,
    ) -> tuple[PipelineRepairLog, Path]:
        """Repair from current artifacts and evidence without generation chat state."""

        if self._repair_agent is None:
            raise RuntimeError("PipelineGenerationAgent has no repair capability configured.")
        return self._repair_agent.repair_pipeline(
            pipeline_dir=pipeline_dir,
            command=command,
            prompt_name=prompt_name,
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
            env_path=env_path,
            env_overrides=env_overrides,
            initial_execution=initial_execution,
            repair_root=repair_root,
            protected_paths=protected_paths,
        )

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
        resolved_output_dir = output_dir or (
            pipeline_experiment_slug(variant) if experimental else "pipeline"
        )
        resolved_access_context = access_context or read_access_context(dataset_dir)
        return strategy.generate(
            contract=contract,
            dataset_dir=dataset_dir,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=resolved_output_dir,
            access_context=resolved_access_context,
        )

    def build_family_pipeline(
        self,
        *,
        seed_contract: DatasetContract,
        inventory: DatasetInventory,
        dataset_dir: Path,
        variant: PipelineVariant = PipelineVariant.DIRECT_LLM,
        prompt_name: str = "family_adapter_v1",
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        allow_network_probe: bool = False,
        access_context: AccessContext | None = None,
        condition: PipelineConditionMetadata | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> FamilyPipelineBuild:
        """Generate and freeze one reusable adapter for an inventory snapshot."""

        strategy = self._strategies.get(variant)
        if strategy is None:
            raise ValueError(f"No ETL pipeline strategy registered for variant: {variant}")
        if seed_contract.dataset_slug != inventory.dataset_slug:
            raise ValueError("Seed contract and inventory belong to different datasets.")

        resolved_policy = policy or PipelinePolicy(
            provider=inventory.provider,
            dataset_id=inventory.dataset_id,
            acquisition_format=str(inventory.defaults.get("data_format", "provider_native")),
            publication_format="zarr",
            data_model="xarray_dataset",
        )
        if (
            resolved_policy.publication_format == "zarr"
            and resolved_policy.zarr is None
        ):
            resolved_policy = resolved_policy.model_copy(
                update={"zarr": ZarrPublicationPolicy()}
            )
        if (
            resolved_policy.publication_format == "parquet"
            and resolved_policy.parquet is None
        ):
            resolved_policy = resolved_policy.model_copy(
                update={"parquet": ParquetPublicationPolicy()}
            )
        inventory_hash = stable_json_hash(inventory)
        resolved_pipeline_id = pipeline_id or pipeline_version_id(
            inventory.dataset_slug,
            variant,
            inventory_hash,
            label=condition.name.value if condition else None,
        )
        workspace = DatasetPipelinePaths(dataset_dir.resolve())
        pipeline_dir = workspace.pipeline(resolved_pipeline_id)
        if pipeline_dir.exists():
            raise FileExistsError(f"Immutable pipeline version already exists: {pipeline_dir}")
        output_dir = pipeline_dir.relative_to(workspace.dataset_dir).as_posix()
        resolved_access_context = access_context or read_access_context(dataset_dir)

        generation_kwargs = dict(
            contract=seed_contract,
            dataset_dir=dataset_dir,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=output_dir,
            access_context=resolved_access_context,
            generation_mode=GenerationMode.DATASET_FAMILY,
            inventory=inventory,
            policy=resolved_policy,
            pipeline_id=resolved_pipeline_id,
            condition=condition,
        )
        if prompt_override is not None:
            if variant not in {
                PipelineVariant.DIRECT_LLM,
                PipelineVariant.TERRAIO_DIRECT,
            }:
                raise ValueError(
                    "Prompt overrides currently support direct generation strategies only."
                )
            generation_kwargs["prompt_override"] = prompt_override
        generation = strategy.generate(**generation_kwargs)
        _validate_family_generated_files(generation, output_dir=output_dir)

        generated_files = [
            file.relative_path for file in [*generation.files, *generation.tests]
        ]
        generated_files.extend(
            [
                f"{output_dir}/run_pipeline.py",
                f"{output_dir}/pipeline_contract.json",
            ]
        )
        manifest = generation.manifest.model_copy(
            update={
                "schema_version": (
                    generation.manifest.schema_version
                    if generation.manifest.prompt_provenance is not None
                    else "etl_pipeline_manifest.v3"
                ),
                "generation_mode": GenerationMode.DATASET_FAMILY,
                "pipeline_id": resolved_pipeline_id,
                "inventory_hash": inventory_hash,
                "inventory_schema_version": inventory.schema_version,
                "fixed_policy": resolved_policy,
                "generated_files": generated_files,
                "pipeline_contract_hash": None,
            }
        )
        credential_names = _credential_names(resolved_access_context)
        regular_grid_zarr_interface = (
            resolved_policy.publication_format == "zarr"
            and resolved_policy.data_model == "xarray_dataset"
        )
        station_parquet_interface = (
            resolved_policy.publication_format == "parquet"
            and resolved_policy.data_model == "records"
        )
        pipeline_contract = PipelineContract(
            schema_version=(
                "etl_pipeline_contract.v3"
                if regular_grid_zarr_interface
                else (
                    "etl_pipeline_contract.v4"
                    if station_parquet_interface
                    else "etl_pipeline_contract.v1"
                )
            ),
            implementation_interface_version=(
                "family_pipeline_interface.v3"
                if regular_grid_zarr_interface
                else (
                    "family_pipeline_interface.v4"
                    if station_parquet_interface
                    else "family_pipeline_interface.v1"
                )
            ),
            pipeline_id=resolved_pipeline_id,
            dataset_slug=inventory.dataset_slug,
            generation_mode=GenerationMode.DATASET_FAMILY,
            credential_environment_variables=credential_names,
            output_artifact=OutputArtifactContract(
                artifact_id=f"{inventory.dataset_slug}_dataset",
                data_model=resolved_policy.data_model,
                storage_format=resolved_policy.publication_format,
                path_template="{output_dir}",
            ),
            policy=resolved_policy,
            generation_manifest_hash=pipeline_manifest_hash(manifest),
        )
        manifest = manifest.model_copy(
            update={"pipeline_contract_hash": pipeline_contract_hash(pipeline_contract)}
        )
        generation = generation.model_copy(update={"manifest": manifest})

        write_pipeline_artifacts(
            generation,
            dataset_dir=workspace.dataset_dir,
            pipeline_dir=pipeline_dir,
            pipeline_contract=pipeline_contract,
            framework_files={
                "run_pipeline.py": FAMILY_RUNNER_TEMPLATE.read_text(encoding="utf-8")
            },
        )
        validate_pipeline_bundle(pipeline_dir)
        return FamilyPipelineBuild(
            generation=generation,
            pipeline_contract=pipeline_contract,
            pipeline_dir=pipeline_dir,
        )

    def build_condition_pipeline(
        self,
        *,
        condition_name: PipelineConditionName | str,
        seed_contract: DatasetContract,
        inventory: DatasetInventory,
        dataset_dir: Path,
        policy: PipelinePolicy | None = None,
        pipeline_id: str | None = None,
        allow_network_probe: bool = False,
        access_context: AccessContext | None = None,
        prompt_lock: PipelinePromptLock | None = None,
        prompt_override: GenerationPromptOverride | None = None,
    ) -> FamilyPipelineBuild:
        """Build one frozen primary-condition adapter with matched infrastructure."""

        condition = pipeline_condition(condition_name)
        if prompt_lock is not None:
            validate_pipeline_prompt_lock(prompt_lock)
        if prompt_lock is not None and prompt_override is not None:
            raise ValueError("Use either a registry prompt lock or a prompt override, not both.")
        build = self.build_family_pipeline(
            seed_contract=seed_contract,
            inventory=inventory,
            dataset_dir=dataset_dir,
            variant=condition.orchestration_variant,
            prompt_name=condition.prompt_name,
            policy=policy,
            pipeline_id=pipeline_id,
            allow_network_probe=allow_network_probe,
            access_context=access_context,
            condition=condition,
            prompt_override=prompt_override,
        )
        if prompt_lock is not None:
            validate_manifest_prompt_lock(build.generation.manifest, prompt_lock)
        return build

    def execute_family_pipeline(
        self,
        *,
        dataset_dir: Path,
        pipeline_id: str,
        contract_lock_path: Path,
        inventory_path: Path | None = None,
        cache_dir: Path | None = None,
        run_id: str | None = None,
        timeout_seconds: int = 3600,
        env_path: Path = ENV_FILE,
        env_overrides: Mapping[str, str] | None = None,
        unset_environment_variables: set[str] | None = None,
        repair_run_reference: str | None = None,
        python_executable: Path | None = None,
    ) -> PipelineExecutionRun:
        """Execute a reusable pipeline through the framework-owned run boundary."""

        return execute_family_pipeline(
            dataset_dir=dataset_dir,
            pipeline_id=pipeline_id,
            contract_lock_path=contract_lock_path,
            inventory_path=inventory_path,
            cache_dir=cache_dir,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            env_path=env_path,
            env_overrides=env_overrides,
            unset_environment_variables=unset_environment_variables,
            repair_run_reference=repair_run_reference,
            python_executable=python_executable,
        )

    def build_and_verify_pipeline(
        self,
        *,
        contract: DatasetContract,
        dataset_dir: Path,
        command: Sequence[str],
        variant: PipelineVariant = PipelineVariant.DIRECT_LLM,
        prompt_name: str = "default",
        repair_prompt_name: str = "default",
        allow_network_probe: bool = False,
        experimental: bool = False,
        output_dir: str | None = None,
        access_context: AccessContext | None = None,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
        timeout_seconds: int = 3600,
        env_path: Path = ENV_FILE,
        env_overrides: Mapping[str, str] | None = None,
    ) -> PipelineBuildRun:
        """Generate, persist, execute, and repair a pipeline as one workflow."""

        if self._repair_agent is None:
            raise RuntimeError(
                "PipelineGenerationAgent requires a repair capability for "
                "build_and_verify_pipeline()."
            )
        if not command:
            raise ValueError("A non-empty pipeline verification command is required.")
        if not 1 <= max_repair_attempts <= MAX_REPAIR_ATTEMPTS:
            raise ValueError(
                f"max_repair_attempts must be between 1 and {MAX_REPAIR_ATTEMPTS}."
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        resolved_output_dir = output_dir or (
            pipeline_experiment_slug(variant) if experimental else "pipeline"
        )
        if not is_pipeline_output_dir(resolved_output_dir):
            raise ValueError(
                "output_dir must be 'pipeline' or a generated pipeline_* experiment slug."
            )
        pipeline_dir = dataset_dir.resolve() / resolved_output_dir
        if pipeline_dir.exists():
            if not pipeline_dir.is_dir():
                raise FileExistsError(f"Pipeline output path is not a directory: {pipeline_dir}")
            if any(pipeline_dir.iterdir()):
                raise FileExistsError(f"Pipeline output directory is not empty: {pipeline_dir}")

        generation = self.build_pipeline(
            contract=contract,
            dataset_dir=dataset_dir,
            variant=variant,
            prompt_name=prompt_name,
            allow_network_probe=allow_network_probe,
            output_dir=resolved_output_dir,
            access_context=access_context,
        )
        write_pipeline_artifacts(
            generation,
            dataset_dir=dataset_dir,
            pipeline_dir=pipeline_dir,
        )
        repair_log, repair_log_path = self.repair(
            pipeline_dir=pipeline_dir,
            command=command,
            prompt_name=repair_prompt_name,
            max_attempts=max_repair_attempts,
            timeout_seconds=timeout_seconds,
            env_path=env_path,
            env_overrides=env_overrides,
        )
        return PipelineBuildRun(
            generation=generation,
            pipeline_dir=pipeline_dir,
            repair_log=repair_log,
            repair_log_path=repair_log_path,
        )


# Historical public name retained for existing scripts and experiment artifacts.
ETLPipelineAgent = PipelineGenerationAgent


def write_pipeline_artifacts(
    result: PipelineGenerationResult,
    *,
    dataset_dir: Path,
    pipeline_dir: Path,
    pipeline_contract: PipelineContract | None = None,
    framework_files: Mapping[str, str] | None = None,
) -> None:
    """Persist one generation result without executing or evaluating it."""

    artifacts = [*result.files, *result.tests]
    if not artifacts:
        raise ValueError("Pipeline generation returned no files.")
    dataset_dir = dataset_dir.resolve()
    pipeline_dir = pipeline_dir.resolve()
    try:
        expected_output_dir = pipeline_dir.relative_to(dataset_dir).as_posix()
    except ValueError as error:
        raise ValueError("Pipeline directory must stay under the dataset directory.") from error
    if result.manifest.output_dir != expected_output_dir:
        raise ValueError(
            "Pipeline manifest output_dir does not match the requested output directory."
        )

    targets: dict[Path, str] = {}
    for artifact in artifacts:
        target = (dataset_dir / artifact.relative_path).resolve()
        try:
            target.relative_to(pipeline_dir)
        except ValueError as error:
            raise ValueError(
                f"Generated file is outside the requested pipeline directory: {artifact.relative_path}"
            ) from error
        relative_to_pipeline = target.relative_to(pipeline_dir).as_posix()
        if pipeline_contract is not None and relative_to_pipeline in FAMILY_PROTECTED_PATHS:
            raise ValueError(
                f"Generated files must not replace framework-owned {relative_to_pipeline}."
            )
        if target.name == "manifest.json":
            raise ValueError(
                "Generated files must not replace the framework-owned manifest.json."
            )
        if target in targets:
            raise ValueError(f"Duplicate generated file path: {artifact.relative_path}")
        targets[target] = artifact.content

    if pipeline_dir.exists():
        if not pipeline_dir.is_dir():
            raise FileExistsError(f"Pipeline output path is not a directory: {pipeline_dir}")
        if any(pipeline_dir.iterdir()):
            raise FileExistsError(f"Pipeline output directory is not empty: {pipeline_dir}")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    for target, content in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, content)
    for relative_path, content in (framework_files or {}).items():
        if relative_path not in FAMILY_PROTECTED_PATHS:
            raise ValueError(f"Unexpected framework-owned pipeline path: {relative_path}")
        _write_atomic(pipeline_dir / relative_path, content)
    if pipeline_contract is not None:
        _write_atomic(
            pipeline_dir / "pipeline_contract.json",
            _serialized_pipeline_contract(pipeline_contract),
        )
    _write_atomic(
        pipeline_dir / "manifest.json",
        _serialized_manifest(result.manifest),
    )


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _serialized_manifest(manifest: PipelineManifest) -> str:
    excluded_fields = (
        {
            "reference_context",
            "condition",
            "strategy_status",
            "prompt_provenance",
        }
        if getattr(manifest, "schema_version", "") in {
            "etl_pipeline_manifest.v2",
            "etl_pipeline_manifest.v3",
        }
        else set()
    )
    payload = manifest.model_dump(mode="json", exclude=excluded_fields)
    if (
        manifest.fixed_policy is not None
        and manifest.fixed_policy.zarr is None
        and "zarr" not in manifest.fixed_policy.model_fields_set
    ):
        payload["fixed_policy"].pop("zarr", None)
    if (
        manifest.fixed_policy is not None
        and manifest.fixed_policy.parquet is None
        and "parquet" not in manifest.fixed_policy.model_fields_set
    ):
        payload["fixed_policy"].pop("parquet", None)
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def _serialized_pipeline_contract(pipeline_contract: PipelineContract) -> str:
    excluded_fields = (
        {"implementation_interface_version"}
        if pipeline_contract.schema_version == "etl_pipeline_contract.v1"
        else set()
    )
    payload = pipeline_contract.model_dump(mode="json", exclude=excluded_fields)
    if (
        pipeline_contract.policy.zarr is None
        and "zarr" not in pipeline_contract.policy.model_fields_set
    ):
        payload["policy"].pop("zarr", None)
    if (
        pipeline_contract.policy.parquet is None
        and "parquet" not in pipeline_contract.policy.model_fields_set
    ):
        payload["policy"].pop("parquet", None)
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def _validate_family_generated_files(
    result: PipelineGenerationResult,
    *,
    output_dir: str,
) -> None:
    artifacts = [*result.files, *result.tests]
    if not artifacts:
        raise ValueError("Dataset-family generation returned no implementation files.")
    implementation_path = f"{output_dir}/pipeline_impl.py"
    paths: set[str] = set()
    for artifact in artifacts:
        path = Path(artifact.relative_path)
        expected_root = Path(output_dir)
        try:
            relative = path.relative_to(expected_root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"Family pipeline file is outside {output_dir}: {artifact.relative_path}"
            ) from error
        if relative in FAMILY_PROTECTED_PATHS:
            raise ValueError(
                f"Model output attempted to replace framework-owned {relative}."
            )
        if artifact.relative_path in paths:
            raise ValueError(f"Duplicate generated file path: {artifact.relative_path}")
        paths.add(artifact.relative_path)
    if implementation_path not in paths:
        raise ValueError("Dataset-family generation must provide root pipeline_impl.py.")


def _credential_names(access_context: AccessContext | None) -> list[str]:
    if access_context is None:
        return []
    names: list[str] = []
    for reference in access_context.credential_env_vars:
        names.append(reference.env_var)
        names.extend(reference.aliases)
    return list(dict.fromkeys(names))
