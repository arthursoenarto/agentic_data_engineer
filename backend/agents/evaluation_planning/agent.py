"""LLM classifier for evaluation profiles and logical mapping requirements."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from backend.agents.evaluation_planning.schemas import EvaluationPlanningDraft
from backend.evaluation.check_library import evaluation_check_catalog
from backend.llm import LLMUsageSummary, LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_CONTEXT_CHARACTERS = 100_000


class StructuredLLM(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> EvaluationPlanningDraft: ...


@dataclass(frozen=True)
class EvaluationPlanningCall:
    """Validated draft with exact prompt and usage provenance."""

    draft: EvaluationPlanningDraft
    model: str
    rendered_system_prompt: str
    rendered_user_prompt: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    latency_seconds: float = 0.0
    llm_usage: LLMUsageSummary | None = None


class EvaluationPlanningAgent:
    """Draft semantic evaluator inputs without choosing trusted check IDs."""

    def __init__(self, llm: StructuredLLM, *, prompt_name: str = "default") -> None:
        self._llm = llm
        self._prompt_name = prompt_name

    def draft(
        self,
        *,
        inventory: Mapping[str, Any],
        contract_lock: Mapping[str, Any],
        expected_field_ids: list[str],
        planning_context: Mapping[str, Any] | None = None,
    ) -> EvaluationPlanningCall:
        system_prompt = self._prompt("system")
        user_template = self._prompt("user")
        user_prompt = user_template.format(
            inventory_json=json.dumps(
                _bounded_inventory(inventory), indent=2, sort_keys=True
            ),
            contract_json=json.dumps(contract_lock, indent=2, sort_keys=True),
            expected_field_ids_json=json.dumps(expected_field_ids, indent=2),
            catalog_json=json.dumps(_catalog_context(), indent=2, sort_keys=True),
            planning_context_json=json.dumps(
                planning_context or {}, indent=2, sort_keys=True
            ),
        )
        if len(user_prompt) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"Evaluation planning context exceeds {MAX_CONTEXT_CHARACTERS} characters"
            )
        tracker = LLMUsageTracker()
        started = time.perf_counter()
        draft = self._llm.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=EvaluationPlanningDraft,
            max_output_tokens=8_000,
            usage_tracker=tracker,
        )
        latency = time.perf_counter() - started
        return EvaluationPlanningCall(
            draft=draft,
            model=getattr(self._llm, "model", "unknown"),
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
    options = inventory.get("options")
    option_summary: dict[str, Any] = {}
    if isinstance(options, Mapping):
        for name, values in sorted(options.items()):
            if isinstance(values, list):
                option_summary[str(name)] = {
                    "count": len(values),
                    "examples": values[:12],
                }
            elif isinstance(values, Mapping):
                keys = list(values)[:12]
                option_summary[str(name)] = {
                    "count": len(values),
                    "examples": keys,
                }
            else:
                option_summary[str(name)] = values
    return {
        key: inventory.get(key)
        for key in (
            "schema_version",
            "dataset_slug",
            "dataset_id",
            "title",
            "description",
            "provider",
            "process_version",
            "source_url",
            "catalogue_metadata",
            "request_fields",
            "option_units",
            "defaults",
            "constraints",
            "availability",
            "outputs",
            "output_transmission",
            "warnings",
        )
    } | {"option_summary": option_summary}


def _catalog_context() -> list[dict[str, Any]]:
    return [
        {
            "check_id": check.check_id,
            "title": check.title,
            "description": check.description,
            "layer": check.layer,
            "role": check.role,
            "required_profile_tags": check.required_profile_tags,
        }
        for check in evaluation_check_catalog()
    ]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usage_or_none(tracker: LLMUsageTracker) -> LLMUsageSummary | None:
    try:
        return tracker.summary()
    except RuntimeError:
        return None
