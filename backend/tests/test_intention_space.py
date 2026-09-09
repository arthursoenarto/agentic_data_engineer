from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.agents.contract_drafting.schemas import (
    ContractFieldSpec,
    ContractSelectorSpec,
    DatasetContract,
    PipelineDownstreamUse,
    PipelineObjectivePreference,
    PipelineOptimizationRequirements,
    PipelineRequirements,
)
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.agents.intention_space import (
    ConversationTurn,
    IntentionSpaceAgent,
    IntentionSpaceDraft,
    create_intention_space_artifacts,
)
from backend.contracts import file_hash, read_contract_lock


class _FakeLLM:
    model = "fake-intention-model"

    def __init__(self, draft: IntentionSpaceDraft) -> None:
        self._draft = draft

    def complete_json(self, **_: Any) -> IntentionSpaceDraft:
        return self._draft


class _SequentialFakeLLM:
    model = "fake-intention-model"

    def __init__(self, drafts: list[IntentionSpaceDraft]) -> None:
        self._drafts = iter(drafts)

    def complete_json(self, **_: Any) -> IntentionSpaceDraft:
        return next(self._drafts)


def _requirements() -> PipelineRequirements:
    return PipelineRequirements(
        downstream_use=PipelineDownstreamUse(
            kind="ml_training",
            workload_kind="full_field_tensor",
            description="Repeated full-field tensor reads.",
        ),
        optimization=PipelineOptimizationRequirements(
            ordered_objectives=[
                PipelineObjectivePreference(
                    objective="consumer_samples_per_second",
                    direction="maximize",
                    priority=1,
                ),
                PipelineObjectivePreference(
                    objective="output_bytes",
                    direction="minimize",
                    priority=2,
                ),
            ],
            descriptive_measurements=[
                "materialization_seconds",
                "q_engineering",
            ],
        ),
    )


class IntentionSpaceTests(unittest.TestCase):
    def _inventory(self) -> DatasetInventory:
        return DatasetInventory(
            dataset_slug="example_grid",
            title="Example grid",
            source_url="https://example.com/grid",
            generated_at="2026-09-07T00:00:00Z",
            provider="example",
            extraction_method="deterministic",
            extractor_name="fixture",
            options={
                "variable": ["temperature"],
                "level": ["500"],
            },
        )

    def _draft(self) -> IntentionSpaceDraft:
        return IntentionSpaceDraft(
            goal="Prepare a reproducible grid dataset for repeated ML training.",
            selected_data_summary="Temperature at level 500 for the selected period.",
            decisions=["Throughput is the primary optimisation objective."],
            assumptions=[],
            credential_references=[],
            contract=DatasetContract(
                schema_version="dataset_contract.v2",
                dataset_slug="example_grid",
                title="Example ML grid task",
                source_url="https://example.com/grid",
                intent="Prepare selected temperature values for ML training.",
                summary="One selected grid channel.",
                fields=[
                    ContractFieldSpec(
                        name="temperature",
                        selectors=[
                            ContractSelectorSpec(dimension="level", value="500")
                        ],
                    )
                ],
                scope={},
                pipeline_requirements=_requirements(),
            ),
        )

    def test_workflow_materializes_one_linked_secret_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = root / "project/datasets/example_grid/dataset_inventory.json"
            inventory_path.parent.mkdir(parents=True)
            inventory_path.write_text(
                self._inventory().model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            output = root / "runs/r1/specification"
            run, artifacts = create_intention_space_artifacts(
                agent=IntentionSpaceAgent(_FakeLLM(self._draft())),
                run_id="example-grid-r1",
                inventory_path=inventory_path,
                interaction=[
                    ConversationTurn(role="user", content="Prepare temperature at level 500."),
                    ConversationTurn(role="system", content="Which objective comes first?"),
                    ConversationTurn(role="user", content="Maximise throughput, then minimise bytes."),
                ],
                supported_capabilities={"workloads": ["full_field_tensor"]},
                repository_root=root,
                output_dir=output,
            )
            lock = read_contract_lock(artifacts.contract_lock)
            self.assertEqual(run.dataset_slug, "example_grid")
            self.assertEqual(lock.intention_sha256, file_hash(artifacts.intention))
            self.assertEqual(lock.interaction_sha256, file_hash(artifacts.interaction))
            interaction_lines = artifacts.interaction.read_text().splitlines()
            self.assertEqual(len(interaction_lines), 3)
            self.assertNotIn("secret", artifacts.credentials.read_text().lower())
            self.assertIn("consumer_samples_per_second", artifacts.intention.read_text())
            with self.assertRaises(FileExistsError):
                create_intention_space_artifacts(
                    agent=IntentionSpaceAgent(_FakeLLM(self._draft())),
                    run_id="example-grid-r1",
                    inventory_path=inventory_path,
                    interaction=[ConversationTurn(role="user", content="Repeat")],
                    supported_capabilities={},
                    repository_root=root,
                    output_dir=output,
                )

    def test_priority_schema_rejects_conflicts(self) -> None:
        payload = _requirements().optimization.model_dump(mode="json")
        payload["ordered_objectives"][1]["priority"] = 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            PipelineOptimizationRequirements.model_validate(payload)

    def test_one_bounded_clarification_is_recorded_and_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory_path = root / "dataset_inventory.json"
            inventory_path.write_text(
                self._inventory().model_dump_json(indent=2), encoding="utf-8"
            )
            unresolved = self._draft().model_copy(
                update={"clarification_questions": ["Confirm the selected level."]}
            )
            run, artifacts = create_intention_space_artifacts(
                agent=IntentionSpaceAgent(
                    _SequentialFakeLLM([unresolved, self._draft()])
                ),
                run_id="example-grid-clarified",
                inventory_path=inventory_path,
                interaction=[ConversationTurn(role="user", content="Prepare the grid.")],
                supported_capabilities={"workloads": ["full_field_tensor"]},
                repository_root=root,
                output_dir=root / "specification",
                clarification_resolver=lambda questions, active: [
                    ConversationTurn(role="system", content="Use level 500 exactly.")
                ],
            )
            self.assertEqual(run.clarification_rounds, 1)
            self.assertEqual(len(run.agent_calls), 2)
            interaction_lines = artifacts.interaction.read_text().splitlines()
            self.assertEqual(len(interaction_lines), 2)
            self.assertIn("Use level 500", interaction_lines[-1])


if __name__ == "__main__":
    unittest.main()
