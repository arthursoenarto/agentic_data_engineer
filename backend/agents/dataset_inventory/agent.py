"""LLM fallback agent for dataset inventory generation."""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.contract_drafting.schemas import DatasetCandidate
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.llm import LLMClient


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class DatasetInventoryAgent:
    """Creates a best-effort inventory when no deterministic extractor exists."""

    def __init__(self, llm: LLMClient, *, prompt_name: str = "default") -> None:
        self._llm = llm
        self._prompt_name = prompt_name

    def create_inventory(self, candidate: DatasetCandidate, *, dataset_slug: str) -> DatasetInventory:
        inventory = self._llm.complete_json(
            system_prompt=self._prompt_text("system"),
            user_prompt=self._user_prompt(candidate, dataset_slug),
            response_model=DatasetInventory,
            web_search=True,
            web_search_required=True,
            max_output_tokens=8192,
        )
        return inventory.model_copy(
            update={
                "dataset_slug": dataset_slug,
                "extractor_name": "generic_web_inventory_agent",
                "extraction_method": "llm",
            }
        )

    def _prompt_text(self, kind: str) -> str:
        path = PROMPTS_DIR / f"{self._prompt_name}_{kind}.md"
        if not path.exists():
            raise RuntimeError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _user_prompt(self, candidate: DatasetCandidate, dataset_slug: str) -> str:
        template = self._prompt_text("user")
        return template.format(
            dataset_slug=dataset_slug,
            candidate_json=json.dumps(candidate.model_dump(mode="json"), indent=2),
            schema_json=json.dumps(DatasetInventory.model_json_schema(), indent=2),
        )
