from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.etl_pipeline import (
    ETLPipelineAgent,
    GeneratedFile,
    PipelineGenerationAgent,
    PipelineGenerationResult,
    PipelineManifest,
    PipelineVariant,
)
from backend.agents.pipeline_repair import PipelineExecutionResult, PipelineRepairLog


class FakeStrategy:
    variant = PipelineVariant.DIRECT_LLM

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: Any) -> PipelineGenerationResult:
        self.calls += 1
        output_dir = kwargs["output_dir"]
        relative_path = f"{output_dir}/pipeline.py"
        return PipelineGenerationResult(
            manifest=PipelineManifest(
                dataset_slug=kwargs["contract"].dataset_slug,
                output_dir=output_dir,
                generated_files=[relative_path],
            ),
            files=[GeneratedFile(relative_path=relative_path, content="print('ok')\n")],
        )


class FakeRepairAgent:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def repair_pipeline(self, **kwargs: Any) -> tuple[PipelineRepairLog, Path]:
        self.request = kwargs
        pipeline_dir = kwargs["pipeline_dir"]
        assert (pipeline_dir / "pipeline.py").is_file()
        assert (pipeline_dir / "manifest.json").is_file()
        now = datetime.now(UTC).isoformat()
        execution = PipelineExecutionResult(
            command=list(kwargs["command"]),
            started_at=now,
            completed_at=now,
            duration_seconds=0,
            return_code=0,
            ok=True,
        )
        log = PipelineRepairLog(
            pipeline_dir=pipeline_dir.name,
            command=list(kwargs["command"]),
            model="gpt-5.5",
            prompt_name=kwargs["prompt_name"],
            max_attempts=kwargs["max_attempts"],
            started_at=now,
            completed_at=now,
            final_status="already_succeeded",
            initial_execution=execution,
        )
        log_dir = pipeline_dir / "repairs" / "repair_test"
        log_dir.mkdir(parents=True)
        log_path = log_dir / "repair_log.json"
        log_path.write_text(log.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return log, log_path


class PipelineBuildWorkflowTests(unittest.TestCase):
    def test_canonical_agent_exposes_role_specialized_context_policy(self) -> None:
        self.assertIs(ETLPipelineAgent, PipelineGenerationAgent)
        agent = PipelineGenerationAgent({})

        self.assertEqual(agent.context_policy.persistent_memory, "artifact_backed")
        self.assertFalse(agent.context_policy.shared_conversational_memory)
        self.assertIn("strategy_prompts", agent.context_policy.generation_context)
        self.assertIn("execution_failure", agent.context_policy.repair_context)
        self.assertNotIn("strategy_prompts", agent.context_policy.repair_context)

    def test_verified_build_generates_writes_executes_and_returns_one_run(self) -> None:
        strategy = FakeStrategy()
        repair_agent = FakeRepairAgent()
        agent = ETLPipelineAgent(  # type: ignore[arg-type]
            {PipelineVariant.DIRECT_LLM: strategy},
            repair_agent=repair_agent,
        )
        contract = DatasetContract(
            dataset_slug="example",
            title="Example",
            source_url="https://example.com/data",
            intent="Retrieve a small example.",
            summary="Example dataset.",
        )

        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "datasets" / "example"
            dataset_dir.mkdir(parents=True)
            run = agent.build_and_verify_pipeline(
                contract=contract,
                dataset_dir=dataset_dir,
                command=["python", "pipeline.py"],
            )

            self.assertTrue(run.verified)
            self.assertEqual(strategy.calls, 1)
            self.assertEqual(run.pipeline_dir, (dataset_dir / "pipeline").resolve())
            self.assertEqual(
                (run.pipeline_dir / "pipeline.py").read_text(encoding="utf-8"),
                "print('ok')\n",
            )
            manifest = PipelineManifest.model_validate_json(
                (run.pipeline_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest.output_dir, "pipeline")
            self.assertEqual(repair_agent.request["max_attempts"], 3)
            self.assertEqual(repair_agent.request["command"], ["python", "pipeline.py"])
            self.assertTrue(run.repair_log_path.is_file())

    def test_verified_build_requires_a_repair_component(self) -> None:
        strategy = FakeStrategy()
        agent = ETLPipelineAgent({PipelineVariant.DIRECT_LLM: strategy})  # type: ignore[arg-type]
        contract = DatasetContract(
            dataset_slug="example",
            title="Example",
            source_url="https://example.com/data",
            intent="Retrieve a small example.",
            summary="Example dataset.",
        )

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "requires a repair capability"):
                agent.build_and_verify_pipeline(
                    contract=contract,
                    dataset_dir=Path(temporary),
                    command=["python", "pipeline.py"],
                )
        self.assertEqual(strategy.calls, 0)


if __name__ == "__main__":
    unittest.main()
