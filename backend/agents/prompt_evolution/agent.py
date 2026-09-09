"""GEPA-inspired reflective mutation of pipeline-generation prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.agents.prompt_evolution.schemas import PromptMutation, PromptMutationCall
from backend.llm import LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_TRAJECTORY_CHARACTERS = 180_000


class StructuredLLM(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> PromptMutation: ...


class PromptEvolutionAgent:
    """Produce one trace-grounded mutation while leaving evaluation fixed."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def mutate(
        self,
        *,
        experiment_id: str,
        iteration: int,
        candidate_id: str,
        parent_candidate_id: str,
        parent_system_prompt: str,
        fixed_user_prompt: str,
        task: Mapping[str, Any],
        evaluation_feedback: Mapping[str, Any],
        candidate_source: str,
        history: Sequence[Mapping[str, Any]],
    ) -> PromptMutationCall:
        system_prompt = _read_prompt("gepa_v1_system.md")
        feedback_json = json.dumps(evaluation_feedback, indent=2, sort_keys=True)
        user_prompt = _read_prompt("gepa_v1_user.md").format(
            iteration=iteration,
            task_json=json.dumps(task, indent=2, sort_keys=True),
            evaluation_json=feedback_json,
            history_json=json.dumps(list(history), indent=2, sort_keys=True),
            parent_system_prompt=parent_system_prompt,
            fixed_user_prompt=fixed_user_prompt,
            candidate_source=candidate_source,
        )
        if len(user_prompt) > MAX_TRAJECTORY_CHARACTERS:
            raise ValueError(
                f"Prompt-evolution trajectory exceeds {MAX_TRAJECTORY_CHARACTERS} characters."
            )
        usage = LLMUsageTracker()
        mutation = self._llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=PromptMutation,
            max_output_tokens=12_000,
            usage_tracker=usage,
        )
        return PromptMutationCall(
            experiment_id=experiment_id,
            iteration=iteration,
            candidate_id=candidate_id,
            parent_candidate_id=parent_candidate_id,
            model=getattr(self._llm, "model", "unknown"),
            system_prompt_sha256=_sha256(system_prompt),
            user_prompt_sha256=_sha256(user_prompt),
            parent_generation_prompt_sha256=_sha256(parent_system_prompt),
            evaluation_feedback_sha256=_sha256(feedback_json),
            mutation=mutation,
            llm_usage=usage.summary(),
        )


def _read_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
