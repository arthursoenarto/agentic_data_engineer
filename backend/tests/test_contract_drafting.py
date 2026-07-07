from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agents.contract_drafting import (
    ContractDraftingAgent,
    DatasetCandidate,
    DatasetCandidateInput,
    DatasetContract,
)
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.contract_drafting.workflow import dataset_slug, normalize_candidate_input, slugify
from backend.scripts.draft_dataset_contract import draft_contract


def example_contract(candidate: DatasetCandidate, *, summary: str = "Drafted from docs.") -> DatasetContract:
    return DatasetContract(
        dataset_slug=candidate.slug or candidate.name.lower(),
        title=candidate.name,
        source_url=candidate.url,
        intent=candidate.user_goal or "Explore whether this dataset supports the project goal.",
        summary=summary,
        dataset_family="generic_api",
        access_methods=["REST API"],
        scope={"date_range": "last_7_days"},
        human_editable_fields=["scope.date_range"],
    )


class ContractDraftingTests(unittest.TestCase):
    def test_agent_drafts_contract_with_required_web_search(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.kwargs = None

            def complete_json(self, **kwargs):  # type: ignore[no-untyped-def]
                self.kwargs = kwargs
                return DatasetContract(
                    dataset_slug="example_dataset",
                    title="Example Dataset",
                    source_url="https://example.org/data",
                    intent="Explore a documented example dataset.",
                    summary="A small contract draft.",
                    fields=[{"name": "temperature", "units": "K"}],
                    evidence=[{"url": "https://example.org/data", "title": "Example docs"}],
                )

        llm = FakeLLM()
        agent = ContractDraftingAgent(llm)  # type: ignore[arg-type]
        candidate = DatasetCandidate(name="Example Dataset", url="https://example.org/data")
        inventory = DatasetInventory(
            dataset_slug="example_dataset",
            title="Example Dataset",
            source_url="https://example.org/data",
            generated_at="2026-07-06T00:00:00+00:00",
            extractor_name="test",
            extraction_method="manual",
            options={"variable": ["temperature"]},
        )

        contract = agent.draft_contract(candidate, inventory=inventory)

        self.assertEqual(contract.title, "Example Dataset")
        self.assertIs(llm.kwargs["response_model"], DatasetContract)
        self.assertTrue(llm.kwargs["web_search"])
        self.assertTrue(llm.kwargs["web_search_required"])
        self.assertIn("DatasetCandidate JSON", llm.kwargs["user_prompt"])
        self.assertIn("DatasetInventory JSON", llm.kwargs["user_prompt"])
        self.assertIn("temperature", llm.kwargs["user_prompt"])

    def test_draft_candidate_writes_project_contract_file(self) -> None:
        class FakeAgent:
            def __init__(self, llm, *, prompt_name="default"):  # type: ignore[no-untyped-def]
                self.llm = llm
                self.prompt_name = prompt_name

            def draft_contract(self, candidate, *, inventory=None):  # type: ignore[no-untyped-def]
                self.inventory = inventory
                return example_contract(candidate)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "candidate.json"
            output_dir = root / "datasets"
            input_path.write_text(
                json.dumps({"name": "OpenAQ", "url": "https://docs.openaq.org/"}),
                encoding="utf-8",
            )

            with (
                patch("backend.scripts.draft_dataset_contract.ROOT", root),
                patch("backend.agents.contract_drafting.workflow.LLMClient", lambda **kwargs: object()),
                patch("backend.agents.contract_drafting.workflow.ContractDraftingAgent", FakeAgent),
            ):
                output_path = draft_contract(input_path=input_path, datasets_dir=output_dir)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output_path.name, "contract.json")
            self.assertEqual(output_path.parent.name, "openaq")
            self.assertEqual(payload["contract"]["title"], "OpenAQ")
            self.assertEqual(payload["schema_version"], "dataset_contract_run.v1")
            self.assertEqual(payload["candidate_file"], "candidate.json")
            self.assertEqual(payload["inventory_file"], "datasets/openaq/dataset_inventory.json")
            inventory_payload = json.loads((output_dir / "openaq/dataset_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory_payload["extraction_method"], "manual")

    def test_slugify(self) -> None:
        self.assertEqual(slugify("ERA5 Single Levels"), "era5_single_levels")

    def test_explicit_candidate_slug_wins(self) -> None:
        candidate = DatasetCandidate(name="ERA5 Single Levels", url="https://example.org/era5", slug="era5")

        self.assertEqual(dataset_slug(candidate), "era5")

    def test_normalize_candidate_input_accepts_url_only(self) -> None:
        candidate = normalize_candidate_input(
            DatasetCandidateInput(url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels")
        )

        self.assertEqual(candidate.name, "Reanalysis Era5 Single Levels")
        self.assertEqual(candidate.slug, "reanalysis_era5_single_levels")


if __name__ == "__main__":
    unittest.main()
