from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.agents.pipeline_repair import (
    PipelineRepairAgent,
    PipelineRepairEdit,
    PipelineExecutionResult,
    PipelineRepairLog,
    PipelineRepairProposal,
)
from backend.agents.pipeline_repair.agent import _classify_execution, _pipeline_context


class FakeLLM:
    model = "gpt-5.5"

    def __init__(self, proposals: list[PipelineRepairProposal]) -> None:
        self.proposals = proposals
        self.calls = 0
        self.user_prompts: list[str] = []

    def complete_json(self, **kwargs: Any) -> PipelineRepairProposal:
        self.calls += 1
        self.user_prompts.append(kwargs["user_prompt"])
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 120,
                },
            },
            requested_model=self.model,
        )
        return self.proposals[self.calls - 1]


def _completed(return_code: int, *, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python", "pipeline.py"],
        returncode=return_code,
        stdout="ok\n" if return_code == 0 else "",
        stderr=stderr,
    )


def _failed_execution(stderr: str, *, command: list[str] | None = None) -> PipelineExecutionResult:
    return PipelineExecutionResult(
        command=command or ["python", "pipeline.py"],
        started_at="2026-08-06T00:00:00+00:00",
        completed_at="2026-08-06T00:00:01+00:00",
        duration_seconds=1.0,
        return_code=1,
        stderr=stderr,
        ok=False,
    )


class PipelineRepairAgentTests(unittest.TestCase):
    def test_repair_context_includes_public_execution_inputs_and_read_only_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline_dir = root / "pipeline"
            pipeline_dir.mkdir()
            (pipeline_dir / "pipeline.py").write_text("raise ValueError()\n", encoding="utf-8")
            (pipeline_dir / "pipeline_contract.json").write_text(
                '{"implementation_interface_version":"family_pipeline_interface.v3"}\n',
                encoding="utf-8",
            )
            contract = root / "contract.lock.json"
            contract.write_text(
                '{"contract":{"fields":[{"name":"temperature"}]}}\n',
                encoding="utf-8",
            )

            context, selected, _ = _pipeline_context(
                pipeline_dir,
                "ValueError: no requested fields",
                protected_paths={"pipeline_contract.json"},
                execution_inputs=[("--contract", contract)],
            )

            self.assertEqual(selected[0], "execution_input:--contract:contract.lock.json")
            self.assertIn('"fields"', context)
            self.assertIn('path="pipeline_contract.json" editable="false"', context)

    def test_candidate_owned_missing_import_remains_repairable(self) -> None:
        result = _classify_execution(
            _failed_execution("ModuleNotFoundError: No module named 'candidate_codec'")
        )

        self.assertEqual(result.failure_origin, "candidate")
        self.assertEqual(result.failure_code, "CANDIDATE_EXECUTION_FAILURE")

    def test_missing_pytest_harness_is_not_sent_to_repair_llm(self) -> None:
        result = _classify_execution(
            _failed_execution(
                "No module named pytest",
                command=["/runtime/python", "-m", "pytest", "-q"],
            )
        )

        self.assertEqual(result.failure_origin, "environment")
        self.assertEqual(
            result.failure_code,
            "ENVIRONMENT_HARNESS_DEPENDENCY_MISSING",
        )

    def test_candidate_output_policy_defect_remains_repairable(self) -> None:
        result = _classify_execution(
            _failed_execution("Published output is not Zarr format 3.")
        )

        self.assertEqual(result.failure_origin, "candidate")

    def test_shared_policy_drift_is_not_sent_to_repair_llm(self) -> None:
        result = _classify_execution(
            _failed_execution(
                "Evaluation output policy differs from the generation-visible pipeline policy"
            )
        )

        self.assertEqual(result.failure_origin, "policy")
        self.assertEqual(result.failure_code, "PUBLIC_OUTPUT_POLICY_FAILURE")

    def test_repair_context_prioritizes_failure_matching_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            source = pipeline_dir / "adapter" / "native.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                'raise ValueError("selector does not match scalar coordinate")\n',
                encoding="utf-8",
            )
            (pipeline_dir / "README.md").write_text("overview\n", encoding="utf-8")
            (pipeline_dir / "manifest.json").write_text(
                '{"prompt": "' + ("x" * 179_000) + '"}\n', encoding="utf-8"
            )

            context, selected, truncated = _pipeline_context(
                pipeline_dir,
                "ValueError: selector pressure_level='500' does not match scalar coordinate 500.0",
                protected_paths={"manifest.json"},
            )

            self.assertEqual(selected[0], "adapter/native.py")
            self.assertIn("adapter/native.py", context)
            self.assertNotIn("manifest.json", selected)
            self.assertFalse(truncated)

    def test_three_repairs_are_logged_with_per_attempt_and_aggregate_usage(self) -> None:
        proposals = [
            PipelineRepairProposal(
                diagnosis="Python uses False rather than JSON false.",
                edits=[
                    PipelineRepairEdit(
                        relative_path="pipeline.py",
                        old_text="VALUE = false",
                        new_text="VALUE = False",
                        rationale="Use the Python boolean literal.",
                    )
                ],
                expected_outcome="The first NameError is removed.",
            ),
            PipelineRepairProposal(
                diagnosis="A second undefined name remains.",
                edits=[
                    PipelineRepairEdit(
                        relative_path="pipeline.py",
                        old_text="STATE = broken",
                        new_text="STATE = 'fixed'",
                        rationale="Replace the undefined name with the intended string value.",
                    )
                ],
                expected_outcome="The command completes.",
            ),
            PipelineRepairProposal(
                diagnosis="A third undefined name remains.",
                edits=[
                    PipelineRepairEdit(
                        relative_path="pipeline.py",
                        old_text="MODE = missing",
                        new_text="MODE = 'ready'",
                        rationale="Replace the final undefined name with the intended string value.",
                    )
                ],
                expected_outcome="The command completes.",
            ),
        ]
        llm = FakeLLM(proposals)

        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline_direct_llm_20260721T000000Z"
            pipeline_dir.mkdir()
            source_path = pipeline_dir / "pipeline.py"
            source_path.write_text(
                "VALUE = false\nSTATE = broken\nMODE = missing\n",
                encoding="utf-8",
            )

            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {"SERVICE_TOKEN": "secret-value"}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    side_effect=[
                        _completed(1, stderr="NameError: false; token=secret-value"),
                        _completed(1, stderr="NameError: broken"),
                        _completed(1, stderr="NameError: missing"),
                        _completed(0),
                    ],
                ) as run,
            ):
                log, log_path = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                    max_attempts=3,
                )

            self.assertEqual(run.call_count, 4)
            self.assertEqual(llm.calls, 3)
            self.assertTrue(all("secret-value" not in prompt for prompt in llm.user_prompts))
            self.assertEqual(log.final_status, "repaired")
            self.assertEqual(
                [attempt.status for attempt in log.attempts],
                ["execution_failed", "execution_failed", "succeeded"],
            )
            self.assertNotIn("secret-value", log.model_dump_json())
            self.assertIn("[REDACTED]", log.initial_execution.stderr)
            self.assertTrue(log_path.is_file())
            self.assertEqual(
                PipelineRepairLog.model_validate_json(log_path.read_text(encoding="utf-8")),
                log,
            )
            self.assertEqual(
                source_path.read_text(encoding="utf-8"),
                "VALUE = False\nSTATE = 'fixed'\nMODE = 'ready'\n",
            )

            assert log.aggregate_llm_usage is not None
            self.assertEqual(log.aggregate_llm_usage.call_count, 3)
            self.assertEqual(log.aggregate_llm_usage.total_tokens, 360)
            self.assertTrue(all(attempt.llm_usage is not None for attempt in log.attempts))
            self.assertEqual(
                [attempt.llm_usage.total_tokens for attempt in log.attempts if attempt.llm_usage],
                [120, 120, 120],
            )

    def test_successful_initial_execution_uses_no_repair_calls(self) -> None:
        llm = FakeLLM([])

        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            (pipeline_dir / "pipeline.py").write_text("print('ok')\n", encoding="utf-8")

            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    return_value=_completed(0),
                ),
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                )

            self.assertEqual(log.final_status, "already_succeeded")
            self.assertEqual(log.attempts, [])
            self.assertIsNone(log.aggregate_llm_usage)
            self.assertEqual(llm.calls, 0)

    def test_each_execution_gets_fresh_output_and_receipt_paths(self) -> None:
        proposal = PipelineRepairProposal(
            diagnosis="Fix the failing marker.",
            edits=[
                PipelineRepairEdit(
                    relative_path="pipeline.py",
                    old_text="BROKEN = True",
                    new_text="BROKEN = False",
                    rationale="Remove the deterministic test failure.",
                )
            ],
            expected_outcome="The rerun succeeds.",
        )
        llm = FakeLLM([proposal])

        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            (pipeline_dir / "pipeline.py").write_text(
                "BROKEN = True\n", encoding="utf-8"
            )
            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    side_effect=[_completed(1, stderr="broken"), _completed(0)],
                ) as run,
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=[
                        "python",
                        "pipeline.py",
                        "--cache-dir",
                        "/stable/cache",
                        "--output-dir",
                        "/stale/output",
                        "--run-receipt",
                        "/stale/receipt.json",
                    ],
                    max_attempts=1,
                )

            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][3], "/stable/cache")
            self.assertEqual(commands[1][3], "/stable/cache")
            first_output = Path(commands[0][5])
            second_output = Path(commands[1][5])
            first_receipt = Path(commands[0][7])
            second_receipt = Path(commands[1][7])
            self.assertNotEqual(first_output, second_output)
            self.assertNotEqual(first_receipt, second_receipt)
            self.assertEqual(first_output.parent.name, "initial")
            self.assertEqual(second_output.parent.name, "attempt_01")
            self.assertEqual(log.initial_execution.execution_label, "initial")
            self.assertEqual(
                log.attempts[0].execution_after.execution_label,  # type: ignore[union-attr]
                "attempt_01",
            )

    def test_manifest_is_not_repairable(self) -> None:
        proposal = PipelineRepairProposal(
            diagnosis="Attempt to alter immutable metadata.",
            edits=[
                PipelineRepairEdit(
                    relative_path="manifest.json",
                    old_text='"model": "gpt-5.5"',
                    new_text='"model": "other"',
                    rationale="This edit must be rejected.",
                )
            ],
            expected_outcome="No valid outcome.",
        )
        llm = FakeLLM([proposal])

        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            manifest = pipeline_dir / "manifest.json"
            manifest.write_text('{"model": "gpt-5.5"}\n', encoding="utf-8")

            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    return_value=_completed(1, stderr="failure"),
                ),
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                    max_attempts=1,
                )

            self.assertEqual(log.final_status, "exhausted")
            self.assertEqual(log.attempts[0].status, "patch_failed")
            self.assertIn("manifest is immutable", log.attempts[0].repair_error or "")
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"model": "gpt-5.5"}\n')

    def test_more_than_three_attempts_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "between 0 and 3"):
                PipelineRepairAgent(FakeLLM([])).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                    max_attempts=4,
                )

    def test_zero_budget_still_runs_initial_command_without_llm(self) -> None:
        llm = FakeLLM([])
        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            (pipeline_dir / "pipeline.py").write_text("print('ok')\n", encoding="utf-8")
            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    return_value=_completed(0),
                ) as run,
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                    max_attempts=0,
                )

            self.assertEqual(run.call_count, 1)
            self.assertEqual(log.final_status, "already_succeeded")
            self.assertEqual(log.max_attempts, 0)
            self.assertEqual(log.attempts, [])
            self.assertEqual(llm.calls, 0)

    def test_provider_failure_does_not_consume_repair_budget(self) -> None:
        llm = FakeLLM([])
        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            (pipeline_dir / "pipeline.py").write_text("print('ok')\n", encoding="utf-8")
            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    return_value=_completed(1, stderr="HTTP Error 502: Bad Gateway"),
                ),
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                )

            self.assertEqual(log.final_status, "non_candidate_failure")
            self.assertEqual(log.initial_execution.failure_origin, "provider")
            self.assertEqual(log.attempts, [])
            self.assertEqual(llm.calls, 0)

    def test_invalid_python_patch_is_rejected_before_source_changes(self) -> None:
        proposal = PipelineRepairProposal(
            diagnosis="Malformed proposed fix.",
            edits=[
                PipelineRepairEdit(
                    relative_path="pipeline.py",
                    old_text="BROKEN = True",
                    new_text="if (",
                    rationale="Exercise transactional syntax validation.",
                )
            ],
            expected_outcome="The malformed patch must be rejected.",
        )
        llm = FakeLLM([proposal])
        with tempfile.TemporaryDirectory() as temporary:
            pipeline_dir = Path(temporary) / "pipeline"
            pipeline_dir.mkdir()
            source = pipeline_dir / "pipeline.py"
            source.write_text("BROKEN = True\n", encoding="utf-8")
            with (
                patch("backend.agents.pipeline_repair.agent.load_env"),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "backend.agents.pipeline_repair.agent.subprocess.run",
                    return_value=_completed(1, stderr="NameError: candidate bug"),
                ),
            ):
                log, _ = PipelineRepairAgent(llm).repair_pipeline(  # type: ignore[arg-type]
                    pipeline_dir=pipeline_dir,
                    command=["python", "pipeline.py"],
                    max_attempts=1,
                )

            self.assertEqual(log.attempts[0].status, "patch_failed")
            self.assertIn("invalid", log.attempts[0].repair_error or "")
            self.assertEqual(source.read_text(encoding="utf-8"), "BROKEN = True\n")


if __name__ == "__main__":
    unittest.main()
