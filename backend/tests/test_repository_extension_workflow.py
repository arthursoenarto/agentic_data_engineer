from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.agents.etl_pipeline import (
    ExtensionCommand,
    ExtensionCommandKind,
    ExtensionContract,
    ExtensionFileChange,
    RepositoryChangeOperation,
    RepositoryExtensionAgent,
    RepositoryExtensionResult,
    TerraioExtensionStrategy,
    execute_repository_extension,
    validate_extension_bundle,
)


class ExtensionLLM:
    model = "gpt-5.5"

    def __init__(self, result: RepositoryExtensionResult) -> None:
        self.result = result
        self.requests: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> RepositoryExtensionResult:
        self.requests.append(kwargs)
        kwargs["usage_tracker"].record_response(
            {
                "model": self.model,
                "usage": {
                    "input_tokens": 200,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 20},
                    "total_tokens": 300,
                },
            },
            requested_model=self.model,
        )
        return self.result


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _source_repository(root: Path) -> tuple[Path, str]:
    repository = root / "terraio-source"
    (repository / "src" / "example").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "README.md").write_text("# Fixture library\n", encoding="utf-8")
    (repository / "src" / "example" / "__init__.py").write_text("", encoding="utf-8")
    (repository / "tests" / "test_existing.py").write_text(
        "import unittest\n\n"
        "class ExistingTest(unittest.TestCase):\n"
        "    def test_baseline(self):\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.email", "fixture@example.com", cwd=repository)
    _run("git", "config", "user.name", "Fixture", cwd=repository)
    _run("git", "add", ".", cwd=repository)
    _run("git", "commit", "-q", "-m", "base", cwd=repository)
    commit = _run("git", "rev-parse", "HEAD", cwd=repository).strip()
    return repository, commit


def _contract(repository: Path, commit: str) -> ExtensionContract:
    return ExtensionContract(
        extension_id="terraio_held_out_provider_v1",
        dataset_slug="held_out_provider",
        repository_url=repository.resolve().as_uri(),
        repository_path=str(repository),
        base_commit=commit,
        context_paths=[
            "README.md",
            "src/example/__init__.py",
            "tests/test_existing.py",
        ],
        allowed_paths=["src/example/**", "tests/**"],
        protected_paths=["src/example/__init__.py", ".git/**"],
        required_public_interfaces=["example.provider:Provider"],
        python_source_roots=["src"],
        architectural_invariants=["Keep provider code independent from tests."],
        baseline_commands=[
            ExtensionCommand(
                name="baseline_tests",
                kind=ExtensionCommandKind.TEST,
                argv=["python", "-m", "unittest", "discover", "-s", "tests"],
            )
        ],
        verification_commands=[
            ExtensionCommand(
                name="all_tests",
                kind=ExtensionCommandKind.TEST,
                argv=["python", "-m", "unittest", "discover", "-s", "tests"],
            )
        ],
        representative_workflow_commands=[
            ExtensionCommand(
                name="provider_smoke",
                kind=ExtensionCommandKind.WORKFLOW,
                argv=[
                    "python",
                    "-c",
                    (
                        "import sys; sys.path.insert(0, 'src'); "
                        "from example.provider import Provider; "
                        "assert Provider().name == 'held-out'"
                    ),
                ],
            )
        ],
        expected_dataset_workflow="Instantiate the held-out provider.",
        expected_artifact_contract="Provider exposes a stable name property.",
    )


def _proposal() -> RepositoryExtensionResult:
    return RepositoryExtensionResult(
        summary="Add a held-out provider implementation.",
        files=[
            ExtensionFileChange(
                relative_path="src/example/provider.py",
                operation=RepositoryChangeOperation.ADD,
                content=(
                    '"""Held-out provider fixture."""\n\n'
                    "class Provider:\n"
                    "    @property\n"
                    "    def name(self) -> str:\n"
                    "        return \"held-out\"\n"
                ),
                rationale="Provide the contracted public interface.",
            )
        ],
        tests=[
            ExtensionFileChange(
                relative_path="tests/test_provider.py",
                operation=RepositoryChangeOperation.ADD,
                content=(
                    "import sys\n"
                    "import unittest\n"
                    "from pathlib import Path\n\n"
                    "sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))\n"
                    "from example.provider import Provider\n\n"
                    "class ProviderTest(unittest.TestCase):\n"
                    "    def test_name(self):\n"
                    "        self.assertEqual(Provider().name, 'held-out')\n"
                ),
                rationale="Verify the new public interface.",
            )
        ],
        readme="# Held-out provider extension\n\nAdds and tests `Provider`.",
    )


class RepositoryExtensionWorkflowTests(unittest.TestCase):
    def test_extension_is_derived_and_checked_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, commit = _source_repository(root)
            source_status_before = _run("git", "status", "--porcelain", cwd=source)
            dataset_dir = root / "project" / "datasets" / "held_out_provider"
            dataset_dir.mkdir(parents=True)
            llm = ExtensionLLM(_proposal())
            build = RepositoryExtensionAgent(
                TerraioExtensionStrategy(llm)  # type: ignore[arg-type]
            ).build(
                contract=_contract(source, commit),
                dataset_dir=dataset_dir,
            )

            self.assertTrue((build.extension_dir / "patch.diff").is_file())
            self.assertTrue(
                (
                    build.extension_dir
                    / "proposed_files"
                    / "src"
                    / "example"
                    / "provider.py"
                ).is_file()
            )
            manifest, contract, proposal = validate_extension_bundle(build.extension_dir)
            self.assertEqual(manifest.base_commit, commit)
            self.assertEqual(contract.extension_id, build.extension_id)
            self.assertEqual(proposal.summary, _proposal().summary)
            self.assertEqual(len(manifest.prompt_provenance.calls), 1)

            run = execute_repository_extension(
                dataset_dir=dataset_dir,
                extension_id=build.extension_id,
            )

            self.assertTrue(run.succeeded)
            self.assertTrue(run.receipt.patch_applied)
            self.assertEqual(
                run.receipt.source_tree_sha256_before,
                run.receipt.source_tree_sha256_after,
            )
            self.assertEqual(
                _run("git", "status", "--porcelain", cwd=source),
                source_status_before,
            )
            self.assertFalse((source / "src" / "example" / "provider.py").exists())
            self.assertTrue((run.run_dir / "test_results.json").is_file())
            self.assertTrue((run.run_dir / "pipeline_run.json").is_file())
            self.assertEqual(
                run.run_dir.parent,
                build.extension_dir / "runs",
            )
            self.assertTrue((run.run_dir / "logs" / "stdout.log").is_file())
            self.assertTrue((run.run_dir / "logs" / "stderr.log").is_file())
            self.assertFalse(
                (build.extension_dir / "runs" / ".work" / run.receipt.run_id).exists()
            )

    def test_protected_repository_path_is_rejected_before_patch_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, commit = _source_repository(root)
            unsafe = _proposal().model_copy(
                update={
                    "files": [
                        ExtensionFileChange(
                            relative_path="src/example/__init__.py",
                            operation=RepositoryChangeOperation.MODIFY,
                            content="from example.provider import Provider\n",
                            rationale="Unsafe protected edit.",
                        )
                    ]
                }
            )
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "protected path"):
                RepositoryExtensionAgent(
                    TerraioExtensionStrategy(ExtensionLLM(unsafe))  # type: ignore[arg-type]
                ).build(
                    contract=_contract(source, commit),
                    dataset_dir=dataset_dir,
                )


if __name__ == "__main__":
    unittest.main()
