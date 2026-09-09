"""LLM component that converts bounded interaction context into typed intent."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from backend.agents.intention_space.schemas import (
    ConversationTurn,
    IntentionSpaceDraft,
)
from backend.llm import LLMUsageSummary, LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_CONTEXT_CHARACTERS = 120_000


class StructuredLLM(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> IntentionSpaceDraft: ...


@dataclass(frozen=True)
class IntentionSpaceCall:
    draft: IntentionSpaceDraft
    model: str
    prompt_version: str
    rendered_system_prompt: str
    rendered_user_prompt: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    latency_seconds: float
    llm_usage: LLMUsageSummary | None


class IntentionSpaceAgent:
    """Interpret intent while leaving validation and publication to the harness."""

    def __init__(self, llm: StructuredLLM, *, prompt_name: str = "default") -> None:
        self._llm = llm
        self._prompt_name = prompt_name

    def draft(
        self,
        *,
        inventory: Mapping[str, Any],
        interaction: Sequence[ConversationTurn],
        supported_capabilities: Mapping[str, Any],
        current_state: Mapping[str, Any] | None = None,
    ) -> IntentionSpaceCall:
        system_prompt = self._prompt("system")
        user_prompt = self._prompt("user").format(
            interaction_json=json.dumps(
                [turn.model_dump(mode="json") for turn in interaction],
                indent=2,
            ),
            inventory_json=json.dumps(_bounded_inventory(inventory), indent=2, sort_keys=True),
            capabilities_json=json.dumps(supported_capabilities, indent=2, sort_keys=True),
            current_state_json=json.dumps(current_state or {}, indent=2, sort_keys=True),
        )
        if len(user_prompt) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"Intention-space context exceeds {MAX_CONTEXT_CHARACTERS} characters"
            )
        tracker = LLMUsageTracker()
        started = time.perf_counter()
        draft = self._llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=IntentionSpaceDraft,
            max_output_tokens=8_000,
            usage_tracker=tracker,
        )
        latency = time.perf_counter() - started
        return IntentionSpaceCall(
            draft=draft,
            model=getattr(self._llm, "model", "unknown"),
            prompt_version=f"intention_space_{self._prompt_name}_v1",
            rendered_system_prompt=system_prompt,
            rendered_user_prompt=user_prompt,
            system_prompt_sha256=_sha256(system_prompt),
            user_prompt_sha256=_sha256(user_prompt),
            latency_seconds=latency,
            llm_usage=_usage_or_none(tracker),
        )

    def _prompt(self, kind: str) -> str:
        path = PROMPTS_DIR / f"{self._prompt_name}_{kind}.md"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8").strip()


def _bounded_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Keep full selected option semantics while excluding bulky provider schemas."""

    keys = (
        "schema_version",
        "dataset_slug",
        "dataset_id",
        "title",
        "description",
        "provider",
        "source_url",
        "request_fields",
        "options",
        "option_units",
        "defaults",
        "constraints",
        "availability",
        "output_transmission",
        "warnings",
        "extractor_name",
        "extraction_method",
        "note",
    )
    return {key: inventory.get(key) for key in keys if key in inventory}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usage_or_none(tracker: LLMUsageTracker) -> LLMUsageSummary | None:
    try:
        return tracker.summary()
    except RuntimeError:
        return None
