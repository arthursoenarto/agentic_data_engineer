"""Deterministic validation and append-only publication of intention artifacts."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from backend.agents.intention_space.agent import IntentionSpaceAgent
from backend.agents.intention_space.schemas import (
    ConversationTurn,
    IntentionAgentCallRecord,
    IntentionInputFile,
    IntentionSpaceDraft,
    IntentionSpaceRun,
)
from backend.contracts import file_hash, lock_dataset_contract, write_editable_contract
from backend.llm import aggregate_llm_usage


class IntentionSpaceArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: Path
    interaction: Path
    intention: Path
    credentials: Path
    editable_contract: Path
    contract_lock: Path
    run: Path
    system_prompt: Path
    user_prompt: Path
    agent_calls: Path


def create_intention_space_artifacts(
    *,
    agent: IntentionSpaceAgent,
    run_id: str,
    inventory_path: Path,
    interaction: Sequence[ConversationTurn],
    supported_capabilities: Mapping[str, Any],
    repository_root: Path,
    output_dir: Path,
    current_state: Mapping[str, Any] | None = None,
    clarification_resolver: Callable[
        [list[str], Sequence[ConversationTurn]], Sequence[ConversationTurn]
    ]
    | None = None,
    max_clarification_rounds: int = 1,
) -> tuple[IntentionSpaceRun, IntentionSpaceArtifacts]:
    """Draft once, validate against inventory, and publish one immutable run."""

    root = repository_root.resolve()
    inventory_path = inventory_path.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Intention-space output already exists: {destination}")
    inventory = _read_mapping(inventory_path)
    active_interaction = list(interaction)
    calls = []
    for round_index in range(max_clarification_rounds + 1):
        call = agent.draft(
            inventory=inventory,
            interaction=active_interaction,
            supported_capabilities=supported_capabilities,
            current_state=current_state,
        )
        calls.append(call)
        if not call.draft.clarification_questions:
            break
        if clarification_resolver is None or round_index >= max_clarification_rounds:
            raise ValueError(
                "Unresolved clarification questions prevent specification materialization"
            )
        answers = list(
            clarification_resolver(
                call.draft.clarification_questions,
                tuple(active_interaction),
            )
        )
        if not answers:
            raise ValueError("Clarification resolver returned no answers")
        active_interaction.extend(answers)
    _validate_draft(call.draft, inventory)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        paths = _artifact_paths(temporary)
        paths.editable_contract.parent.mkdir(parents=True)
        _write_jsonl(paths.interaction, active_interaction)
        paths.intention.write_text(_render_intention(call.draft), encoding="utf-8")
        paths.credentials.write_text(
            json.dumps(
                {
                    "schema_version": "credential_references.v1",
                    "credentials": [
                        item.model_dump(mode="json")
                        for item in call.draft.credential_references
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_editable_contract(call.draft.contract, paths.editable_contract)
        lock_result = lock_dataset_contract(
            editable_contract_path=paths.editable_contract,
            inventory_path=inventory_path,
            locks_dir=paths.editable_contract.parent,
            intention_path=paths.intention,
            interaction_path=paths.interaction,
        )
        paths.system_prompt.write_text(call.rendered_system_prompt + "\n", encoding="utf-8")
        paths.user_prompt.write_text(call.rendered_user_prompt + "\n", encoding="utf-8")
        call_records = [
            IntentionAgentCallRecord(
                round_index=index,
                system_prompt=item.rendered_system_prompt,
                user_prompt=item.rendered_user_prompt,
                system_prompt_sha256=item.system_prompt_sha256,
                user_prompt_sha256=item.user_prompt_sha256,
                latency_seconds=item.latency_seconds,
                usage=item.llm_usage,
                draft=item.draft,
            )
            for index, item in enumerate(calls)
        ]
        paths.agent_calls.write_text(
            json.dumps(
                {
                    "schema_version": "intention_agent_calls.v1",
                    "calls": [item.model_dump(mode="json") for item in call_records],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        usage = aggregate_llm_usage(
            [item.llm_usage for item in calls if item.llm_usage is not None]
        )
        run = IntentionSpaceRun(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            dataset_slug=call.draft.contract.dataset_slug,
            model=call.model,
            prompt_version=call.prompt_version,
            system_prompt_sha256=call.system_prompt_sha256,
            user_prompt_sha256=call.user_prompt_sha256,
            inventory=_input_file(inventory_path, root),
            interaction=_input_file(paths.interaction, temporary),
            intention=_input_file(paths.intention, temporary),
            editable_contract=_input_file(paths.editable_contract, temporary),
            contract_lock=_input_file(lock_result.path, temporary),
            credentials=_input_file(paths.credentials, temporary),
            draft=call.draft,
            agent_calls=call_records,
            clarification_rounds=len(calls) - 1,
            clarification_turns=sum(
                turn.role == "system" for turn in active_interaction
            ),
            llm_latency_seconds=sum(item.latency_seconds for item in calls),
            llm_usage=usage,
        )
        paths.run.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_paths = _artifact_paths(destination)
    return run, final_paths.model_copy(
        update={"contract_lock": destination / "contracts" / lock_result.path.name}
    )


def _validate_draft(draft: IntentionSpaceDraft, inventory: Mapping[str, Any]) -> None:
    if draft.contract.dataset_slug != inventory.get("dataset_slug"):
        raise ValueError("Intention contract dataset_slug must match the trusted inventory")
    if draft.clarification_questions:
        raise ValueError(
            "Unresolved clarification questions prevent specification materialization"
        )


def _render_intention(draft: IntentionSpaceDraft) -> str:
    requirements = draft.contract.pipeline_requirements
    assert requirements is not None
    lines = [
        f"# {draft.contract.title}",
        "",
        "## Goal",
        "",
        draft.goal,
        "",
        "## Selected Data",
        "",
        draft.selected_data_summary,
        "",
        "## Downstream Use",
        "",
        requirements.downstream_use.description
        or requirements.downstream_use.kind.replace("_", " "),
        "",
        "## Optimisation Priorities",
        "",
    ]
    for item in sorted(
        requirements.optimization.ordered_objectives,
        key=lambda objective: objective.priority,
    ):
        lines.append(
            f"{item.priority}. {item.direction.title()} `{item.objective}`."
        )
    lines.extend(["", "## Descriptive Measurements", ""])
    lines.extend(
        f"- `{name}`" for name in requirements.optimization.descriptive_measurements
    )
    lines.extend(["", "## Decisions", ""])
    lines.extend(f"- {item}" for item in (draft.decisions or ["None recorded."]))
    lines.extend(["", "## Assumptions", ""])
    lines.extend(f"- {item}" for item in (draft.assumptions or ["None recorded."]))
    return "\n".join(lines).rstrip() + "\n"


def _write_jsonl(path: Path, turns: Sequence[ConversationTurn]) -> None:
    text = "".join(
        json.dumps(turn.model_dump(mode="json"), sort_keys=True) + "\n"
        for turn in turns
    )
    path.write_text(text, encoding="utf-8")


def _input_file(path: Path, root: Path) -> IntentionInputFile:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return IntentionInputFile(path=display, sha256=file_hash(resolved))


def _artifact_paths(root: Path) -> IntentionSpaceArtifacts:
    return IntentionSpaceArtifacts(
        output_dir=root,
        interaction=root / "interaction.jsonl",
        intention=root / "intention.md",
        credentials=root / "credentials.json",
        editable_contract=root / "contracts" / "contract.yaml",
        contract_lock=root / "contracts" / "contract_v1.lock.json",
        run=root / "intention_run.json",
        system_prompt=root / "system_prompt.md",
        user_prompt=root / "user_prompt.md",
        agent_calls=root / "agent_calls.json",
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload
