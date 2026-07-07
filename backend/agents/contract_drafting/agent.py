"""Dataset Contract Drafting Agent."""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.contract_drafting.schemas import DatasetCandidate, DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.llm import LLMClient


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class ContractDraftingAgent:
    """Drafts a small human-editable dataset contract from a candidate dataset."""

    def __init__(self, llm: LLMClient, *, prompt_name: str = "default") -> None:
        self._llm = llm
        self._prompt_name = prompt_name

    def draft_contract(self, candidate: DatasetCandidate, *, inventory: DatasetInventory | None = None) -> DatasetContract:
        """Create a DatasetContract draft from a user-provided candidate."""

        return self._llm.complete_json(
            system_prompt=self._prompt_text("system"),
            user_prompt=self._user_prompt(candidate, inventory),
            response_model=DatasetContract,
            web_search=True,
            web_search_required=True,
            max_output_tokens=8192,
        )

    def _prompt_text(self, kind: str) -> str:
        path = PROMPTS_DIR / f"{self._prompt_name}_{kind}.md"
        if not path.exists():
            raise RuntimeError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _user_prompt(self, candidate: DatasetCandidate, inventory: DatasetInventory | None) -> str:
        template = self._prompt_text("user")
        return template.format(
            candidate_json=json.dumps(candidate.model_dump(mode="json"), indent=2),
            inventory_json=json.dumps(inventory.model_dump(mode="json"), indent=2) if inventory else "null",
            schema_json=json.dumps(DatasetContract.model_json_schema(), indent=2),
        )
