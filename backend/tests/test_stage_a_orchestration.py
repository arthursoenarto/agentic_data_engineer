from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agents.pipeline_repair.schemas import (
    PipelineExecutionResult,
    PipelineRepairAttempt,
    PipelineRepairLog,
)
from backend.evaluation.controlled_materialization import _resolve_command
from backend.agents.etl_pipeline.schemas import PipelineDataModel, PublicationFormat
from backend.orchestration.stage_a import (
    StageAWorkflowConfig,
    StageAWorkflowReport,
    _framework_source_sha256,
    _has_generated_tests,
    _lineage_input_files,
    _prepare_runtime,
    _validate_requirements,
    _verify_lineage,
)


class StageAOrchestrationTests(unittest.TestCase):
    def test_prompt_override_is_hashed_as_a_lineage_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in (
                "contract.json",
                "inventory.json",
                "builder.py",
                "suite.yaml",
                "system.md",
                "user.md",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                files[name] = path
            system_hash = hashlib.sha256(files["system.md"].read_bytes()).hexdigest()
            user_hash = hashlib.sha256(files["user.md"].read_bytes()).hexdigest()
            config = StageAWorkflowConfig(
                workflow_id="gepa-lineage-test",
                dataset_dir=root,
                seed_contract=files["contract.json"],
                condition="expert_direct_llm",
                pipeline_id="pipeline-gepa-lineage-test",
                inventory=files["inventory.json"],
                generation_prompt_override={
                    "prompt_id": "gepa-p001",
                    "system_prompt": files["system.md"],
                    "user_prompt": files["user.md"],
                    "system_sha256": system_hash,
                    "user_sha256": user_hash,
                },
                primary_suite_builder=files["builder.py"],
                primary_suite_spec=files["suite.yaml"],
                primary_suite_dir=root / "suites",
                evaluation_output_dir=root / "evaluations",
            )

            lineage = _lineage_input_files(config, inventory=files["inventory.json"])

        self.assertEqual(lineage["generation_system_prompt"], files["system.md"].resolve())
        self.assertEqual(lineage["generation_user_prompt"], files["user.md"].resolve())

    def test_prompt_lock_and_override_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            StageAWorkflowConfig(
                workflow_id="invalid-gepa-config",
                dataset_dir=Path("dataset"),
                seed_contract=Path("contract.json"),
                condition="expert_direct_llm",
                pipeline_id="pipeline-invalid-gepa",
                prompt_lock=Path("prompt-lock.json"),
                generation_prompt_override={
                    "prompt_id": "gepa-p001",
                    "system_prompt": Path("system.md"),
                    "user_prompt": Path("user.md"),
                    "system_sha256": "0" * 64,
                    "user_sha256": "1" * 64,
                },
                primary_suite_builder=Path("builder.py"),
                primary_suite_dir=Path("suites"),
                evaluation_output_dir=Path("evaluations"),
            )

    def test_workflow_config_accepts_explicit_station_policy_and_suite_spec(self) -> None:
        config = StageAWorkflowConfig(
            workflow_id="station-test",
            dataset_dir=Path("project/datasets/station"),
            seed_contract=Path("contract.lock.json"),
            condition="expert_direct_llm",
            pipeline_id="pipeline-station-test",
            pipeline_policy={
                "provider": "provider",
                "dataset_id": "station-source",
                "acquisition_format": "json",
                "publication_format": "parquet",
                "data_model": "records",
            },
            primary_suite_builder=Path("builder.py"),
            primary_suite_spec=Path("suite_spec.yaml"),
            primary_suite_dir=Path("suites"),
            evaluation_output_dir=Path("evaluations"),
        )

        assert config.pipeline_policy is not None
        self.assertEqual(
            config.pipeline_policy.publication_format,
            PublicationFormat.PARQUET,
        )
        self.assertEqual(config.pipeline_policy.data_model, PipelineDataModel.RECORDS)

    def test_lineage_verification_detects_input_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "workflow_config.json"
            frozen_input = root / "contract.lock.json"
            config.write_text('{"workflow_id":"test"}\n', encoding="utf-8")
            frozen_input.write_text('{"contract":"v1"}\n', encoding="utf-8")
            framework_hash = _framework_source_sha256()
            report = StageAWorkflowReport(
                workflow_id="lineage-test",
                pipeline_id="pipeline-test",
                condition="naive_llm",
                started_at="2026-08-06T00:00:00+00:00",
                workflow_config_snapshot=str(config),
                workflow_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                framework_source_sha256=framework_hash,
                input_files={"seed_contract": str(frozen_input)},
                input_sha256={
                    "seed_contract": hashlib.sha256(frozen_input.read_bytes()).hexdigest()
                },
            )

            self.assertTrue(_verify_lineage(report))
            frozen_input.write_text('{"contract":"mutated"}\n', encoding="utf-8")
            self.assertFalse(_verify_lineage(report))

    def test_candidate_requirements_allow_only_index_package_specs(self) -> None:
        _validate_requirements(
            "numpy>=2,<3\ndask[array]>=2024.8; python_version >= '3.11'\n"
        )

        for invalid in (
            "--extra-index-url https://example.invalid/simple\n",
            "package @ https://example.invalid/package.whl\n",
            "../local-package\n",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_requirements(invalid)

    def test_controlled_command_preserves_virtualenv_interpreter_path(self) -> None:
        interpreter = "/workspace/tmp/venv/bin/python"
        command = _resolve_command(
            [
                "python",
                "run_pipeline.py",
                "{contract_lock_json}",
                "{dataset_inventory_json}",
                "{cache_dir}",
                "{output_dir}",
                "{pipeline_run_json}",
            ],
            contract=Path("/tmp/contract.json"),
            inventory=Path("/tmp/inventory.json"),
            cache=Path("/tmp/cache"),
            output=Path("/tmp/output"),
            receipt=Path("/tmp/receipt.json"),
            python_executable=interpreter,
        )

        self.assertEqual(command[0], interpreter)

    def test_generated_tests_are_optional_and_detected_by_pytest_pattern(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(_has_generated_tests(root))
            (root / "tests").mkdir()
            (root / "tests" / "helper.py").write_text("", encoding="utf-8")
            self.assertFalse(_has_generated_tests(root))
            (root / "tests" / "test_pipeline.py").write_text("", encoding="utf-8")
            self.assertTrue(_has_generated_tests(root))

    def test_runtime_install_failure_returns_repair_evidence(self) -> None:
        timestamp = "2026-08-21T00:00:00+00:00"
        failure = PipelineExecutionResult(
            command=["python", "-m", "pip", "install"],
            started_at=timestamp,
            completed_at=timestamp,
            duration_seconds=0.1,
            return_code=1,
            stderr="ResolutionImpossible",
            ok=False,
            failure_origin="candidate",
            failure_code="CANDIDATE_EXECUTION_FAILURE",
        )
        repair_log = PipelineRepairLog(
            pipeline_dir="candidate",
            command=failure.command,
            model="test",
            prompt_name="default",
            max_attempts=2,
            started_at=timestamp,
            completed_at=timestamp,
            final_status="exhausted",
            initial_execution=failure,
            attempts=[
                PipelineRepairAttempt(
                    attempt_number=1,
                    started_at=timestamp,
                    completed_at=timestamp,
                    status="execution_failed",
                    failure_before=failure,
                    context_size_chars=0,
                    execution_after=failure,
                )
            ],
        )

        class FakeRepairAgent:
            max_attempts: int | None = None

            def repair(self, **kwargs):  # type: ignore[no-untyped-def]
                self.max_attempts = kwargs["max_attempts"]
                log_path = kwargs["repair_root"] / "repair" / "repair_log.json"
                return repair_log, log_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "candidate"
            pipeline.mkdir()
            (pipeline / "requirements.txt").write_text(
                "pyarrow==16.1.0\n", encoding="utf-8"
            )
            benchmark = root / "backend" / "requirements-benchmark.txt"
            benchmark.parent.mkdir()
            benchmark.write_text("pyarrow>=17,<22\n", encoding="utf-8")
            run_dir = root / "run"
            run_dir.mkdir()
            agent = FakeRepairAgent()

            def create_runtime(_builder, runtime: Path) -> None:  # type: ignore[no-untyped-def]
                (runtime / "bin").mkdir(parents=True)
                (runtime / "bin" / "python").write_text("", encoding="utf-8")

            with (
                patch("backend.orchestration.stage_a.REPOSITORY_ROOT", root),
                patch("backend.orchestration.stage_a.BENCHMARK_REQUIREMENTS", benchmark),
                patch("backend.orchestration.stage_a.venv.EnvBuilder.create", create_runtime),
            ):
                result = _prepare_runtime(
                    pipeline,
                    workflow_id="runtime-repair-test",
                    run_dir=run_dir,
                    timeout_seconds=30,
                    pipeline_agent=agent,  # type: ignore[arg-type]
                    max_repair_attempts=2,
                    repair_prompt_name="default",
                    env_path=root / ".env",
                )

        self.assertIsNone(result.freeze_path)
        self.assertEqual(result.repair_log.final_status, "exhausted")
        self.assertEqual(agent.max_attempts, 2)


if __name__ == "__main__":
    unittest.main()
