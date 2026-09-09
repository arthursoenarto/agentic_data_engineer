"""Deterministic compiler and append-only artifacts for evaluation planning."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.agents.evaluation_planning import (
    EvaluationPlanningAgent,
    EvaluationPlanningCall,
    EvaluationPlanningDraft,
)
from backend.agents.evaluation_planning.schemas import ProposedEvaluationCheckDraft
from backend.agents.contract_drafting.schemas import PipelineOptimizationRequirements
from backend.evaluation.check_library import (
    EvaluationProfile,
    ResolvedEvaluationCheckPlan,
    compile_evaluation_check_plan,
    evaluation_check_catalog,
)
from backend.evaluation.evidence import canonical_json_file_hash, logical_field_id
from backend.llm import LLMUsageSummary


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanningInputFile(_StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class QuarantinedCheckProposal(_StrictModel):
    """LLM proposal that has no authority in the trusted check catalog."""

    status: Literal["quarantined"] = "quarantined"
    promotion_requirements: list[str] = Field(min_length=1)
    proposal: ProposedEvaluationCheckDraft


class EvaluationPlanningRun(_StrictModel):
    """Immutable hybrid planning result consumed by dataset suite builders."""

    schema_version: Literal["evaluation_planning_run.v1"] = (
        "evaluation_planning_run.v1"
    )
    plan_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    dataset_slug: str
    model: str
    prompt_version: Literal["evaluation_planning_v1"] = "evaluation_planning_v1"
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory: PlanningInputFile
    contract_lock: PlanningInputFile
    engineering_enabled: bool
    extensibility_probe_enabled: bool
    llm_draft: EvaluationPlanningDraft
    objective_policy: PipelineOptimizationRequirements | None = None
    profile: EvaluationProfile
    check_plan: ResolvedEvaluationCheckPlan
    quarantined_check_proposals: list[QuarantinedCheckProposal] = Field(
        default_factory=list
    )
    suite_ready: bool
    outstanding_suite_inputs: list[str]
    planning_complete: bool = True
    llm_latency_seconds: float = Field(default=0, ge=0)
    llm_usage: LLMUsageSummary | None = None

    @model_validator(mode="after")
    def plan_matches_profile_and_settings(self) -> "EvaluationPlanningRun":
        expected = compile_evaluation_check_plan(
            self.profile,
            engineering_enabled=self.engineering_enabled,
            extensibility_probe_enabled=self.extensibility_probe_enabled,
        )
        if self.check_plan != expected:
            raise ValueError("Planning check plan must match deterministic resolution")
        if self.check_plan.profile != self.profile:
            raise ValueError("Planning profile and check-plan profile must match")
        return self


class EvaluationPlanningArtifacts(_StrictModel):
    output_dir: Path
    planning_run: Path
    evaluation_profile: Path
    llm_draft: Path
    proposed_checks: Path
    system_prompt: Path
    user_prompt: Path


def compile_evaluation_planning_run(
    *,
    plan_id: str,
    profile_id: str,
    inventory_path: Path,
    contract_lock_path: Path,
    call: EvaluationPlanningCall,
    repository_root: Path,
    engineering_enabled: bool,
    extensibility_probe_enabled: bool,
) -> EvaluationPlanningRun:
    """Validate the LLM draft against trusted inputs and compile exact checks."""

    root = repository_root.resolve()
    inventory_path = inventory_path.resolve()
    contract_lock_path = contract_lock_path.resolve()
    inventory = _read_mapping(inventory_path)
    contract_lock = _read_mapping(contract_lock_path)
    contract = contract_lock.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Contract lock does not contain a contract mapping")
    inventory_hash = canonical_json_file_hash(inventory_path)
    if contract_lock.get("inventory_sha256") != inventory_hash:
        raise ValueError("Contract lock does not match the trusted inventory")
    dataset_slug = str(inventory.get("dataset_slug") or "")
    if not dataset_slug or contract.get("dataset_slug") != dataset_slug:
        raise ValueError("Inventory and contract dataset slugs must match")

    expected_field_ids = _expected_field_ids(contract)
    drafted_field_ids = sorted(field.field_id for field in call.draft.fields)
    if drafted_field_ids != expected_field_ids:
        raise ValueError(
            "LLM field mappings must match the frozen contract exactly: "
            f"expected={expected_field_ids}, observed={drafted_field_ids}"
        )

    requirements = contract.get("pipeline_requirements")
    trusted_objectives: PipelineOptimizationRequirements | None = None
    if isinstance(requirements, Mapping):
        downstream = requirements.get("downstream_use")
        optimization = requirements.get("optimization")
        if not isinstance(downstream, Mapping) or not isinstance(optimization, Mapping):
            raise ValueError("Typed pipeline requirements are malformed")
        trusted_objectives = PipelineOptimizationRequirements.model_validate(optimization)
        if call.draft.target.workload_kind != downstream.get("workload_kind"):
            raise ValueError("Evaluation workload must match the frozen pipeline requirements")
        if call.draft.objective_policy != trusted_objectives:
            raise ValueError("Evaluation objective policy must match the frozen contract exactly")

    trusted_check_ids = {check.check_id for check in evaluation_check_catalog()}
    collisions = sorted(
        item.proposal_id
        for item in call.draft.proposed_checks
        if item.proposal_id in trusted_check_ids
    )
    if collisions:
        raise ValueError(f"Proposed checks collide with trusted catalog: {collisions}")

    dataset_identity = str(inventory.get("dataset_id") or dataset_slug)
    tags = {
        f"dataset:{_tag_value(dataset_identity)}",
        *(_domain_tag(tag) for tag in call.draft.domain_tags),
    }
    family = contract.get("dataset_family")
    if isinstance(family, str) and family.strip():
        tags.add(f"family:{_tag_value(family)}")
    provider = inventory.get("provider")
    if isinstance(provider, str) and provider.strip():
        tags.add(f"provider:{_tag_value(provider)}")

    target = call.draft.target
    profile = EvaluationProfile(
        profile_id=profile_id,
        data_class=target.data_class,
        output_format=target.output_format,
        output_format_version=target.output_format_version,
        workload_kind=target.workload_kind,
        engineering_profile=target.engineering_profile,
        dataset_tags=sorted(tags),
    )
    plan = compile_evaluation_check_plan(
        profile,
        engineering_enabled=engineering_enabled,
        extensibility_probe_enabled=extensibility_probe_enabled,
    )
    proposals = [
        QuarantinedCheckProposal(
            proposal=item,
            promotion_requirements=[
                "deterministic implementation outside candidate code",
                "independent oracle or fixture",
                "positive and mutation test coverage",
                "new versioned trusted catalog release",
            ],
        )
        for item in call.draft.proposed_checks
    ]
    outstanding = [
        "bind candidate physical paths from generation acceptance",
        "freeze an evaluator-owned independent oracle and expected shape",
        "freeze dataset-specific comparison and coordinate parameters",
    ]
    return EvaluationPlanningRun(
        plan_id=plan_id,
        dataset_slug=dataset_slug,
        model=call.model,
        system_prompt_sha256=call.system_prompt_sha256,
        user_prompt_sha256=call.user_prompt_sha256,
        inventory=PlanningInputFile(
            path=_display(inventory_path, root),
            sha256=inventory_hash,
        ),
        contract_lock=PlanningInputFile(
            path=_display(contract_lock_path, root),
            sha256=canonical_json_file_hash(contract_lock_path),
        ),
        engineering_enabled=engineering_enabled,
        extensibility_probe_enabled=extensibility_probe_enabled,
        llm_draft=call.draft,
        objective_policy=trusted_objectives,
        profile=profile,
        check_plan=plan,
        quarantined_check_proposals=proposals,
        suite_ready=False,
        outstanding_suite_inputs=outstanding,
        planning_complete=(
            call.draft.compatibility == "compatible" and not proposals
        ),
        llm_latency_seconds=call.latency_seconds,
        llm_usage=call.llm_usage,
    )


def create_evaluation_planning_artifacts(
    *,
    agent: EvaluationPlanningAgent,
    plan_id: str,
    dataset_dir: Path,
    contract_lock_path: Path,
    repository_root: Path,
    output_dir: Path | None = None,
    profile_id: str | None = None,
    engineering_enabled: bool = True,
    extensibility_probe_enabled: bool = False,
    planning_context: Mapping[str, Any] | None = None,
) -> tuple[EvaluationPlanningRun, EvaluationPlanningArtifacts]:
    """Run one LLM draft and atomically publish deterministic planning artifacts."""

    root = repository_root.resolve()
    dataset_dir = dataset_dir.resolve()
    inventory_path = dataset_dir / "dataset_inventory.json"
    contract_lock_path = contract_lock_path.resolve()
    inventory = _read_mapping(inventory_path)
    contract_lock = _read_mapping(contract_lock_path)
    contract = contract_lock.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Contract lock does not contain a contract mapping")
    expected_field_ids = _expected_field_ids(contract)
    call = agent.draft(
        inventory=inventory,
        contract_lock=contract_lock,
        expected_field_ids=expected_field_ids,
        planning_context=planning_context,
    )
    dataset_identity = str(
        inventory.get("dataset_id") or inventory.get("dataset_slug") or "dataset"
    )
    target = call.draft.target
    resolved_profile_id = profile_id or (
        f"{_tag_value(dataset_identity)}-"
        f"{_tag_value(target.data_class)}-{target.output_format}-v{target.output_format_version}"
    )
    run = compile_evaluation_planning_run(
        plan_id=plan_id,
        profile_id=resolved_profile_id,
        inventory_path=inventory_path,
        contract_lock_path=contract_lock_path,
        call=call,
        repository_root=root,
        engineering_enabled=engineering_enabled,
        extensibility_probe_enabled=extensibility_probe_enabled,
    )

    destination = (
        output_dir.resolve()
        if output_dir is not None
        else dataset_dir / "benchmarks" / "evaluation_plans" / plan_id
    )
    if destination.exists():
        raise FileExistsError(f"Evaluation planning output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        paths = _artifact_paths(temporary)
        paths.system_prompt.write_text(
            call.rendered_system_prompt + "\n", encoding="utf-8"
        )
        paths.user_prompt.write_text(
            call.rendered_user_prompt + "\n", encoding="utf-8"
        )
        paths.llm_draft.write_text(
            call.draft.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        paths.evaluation_profile.write_text(
            yaml.safe_dump(
                run.profile.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        paths.proposed_checks.write_text(
            json.dumps(
                {
                    "schema_version": "evaluation_check_proposals.v1",
                    "plan_id": run.plan_id,
                    "proposals": [
                        item.model_dump(mode="json")
                        for item in run.quarantined_check_proposals
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.planning_run.write_text(
            run.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return run, _artifact_paths(destination)


def load_evaluation_planning_run(path: Path) -> EvaluationPlanningRun:
    """Load and revalidate one frozen planning artifact."""

    return EvaluationPlanningRun.model_validate_json(path.read_text(encoding="utf-8"))


def _artifact_paths(root: Path) -> EvaluationPlanningArtifacts:
    return EvaluationPlanningArtifacts(
        output_dir=root,
        planning_run=root / "evaluation_plan.json",
        evaluation_profile=root / "evaluation_profile.yaml",
        llm_draft=root / "llm_draft.json",
        proposed_checks=root / "proposed_checks.json",
        system_prompt=root / "system_prompt.md",
        user_prompt=root / "user_prompt.md",
    )


def _expected_field_ids(contract: Mapping[str, Any]) -> list[str]:
    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Frozen contract must contain at least one field")
    result: list[str] = []
    for field in fields:
        if not isinstance(field, Mapping):
            raise ValueError("Contract fields must be mappings")
        name = field.get("name")
        selectors = field.get("selectors", [])
        if not isinstance(name, str) or not isinstance(selectors, list):
            raise ValueError("Contract field names/selectors are malformed")
        result.append(logical_field_id(name, selectors))
    if len(result) != len(set(result)):
        raise ValueError("Frozen contract contains duplicate logical fields")
    return sorted(result)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tag_value(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError(f"Cannot derive a tag from {value!r}")
    return normalized


def _domain_tag(value: str) -> str:
    label = value.split(":", 1)[1] if value.startswith("domain:") else value
    return f"domain:{_tag_value(label)}"


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
