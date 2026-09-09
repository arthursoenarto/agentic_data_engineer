"""One-call generation strategy for a PR-style TerraIO repository extension."""

from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from backend.agents.etl_pipeline.extension_schemas import (
    ExtensionContract,
    RepositoryChangeOperation,
    RepositoryExtensionResult,
)
from backend.agents.etl_pipeline.prompt_provenance import (
    PromptRecorder,
    recorded_complete_json,
)
from backend.agents.etl_pipeline.reference_context import (
    ReferenceContext,
    load_repository_reference_context,
)
from backend.agents.etl_pipeline.schemas import PromptProvenance
from backend.llm import LLMClient, LLMUsageSummary, LLMUsageTracker


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "terraio_extension"


@dataclass(frozen=True)
class RepositoryExtensionGeneration:
    """Validated proposal plus exact generation provenance."""

    proposal: RepositoryExtensionResult
    reference_context: ReferenceContext
    prompt_provenance: PromptProvenance
    llm_usage: LLMUsageSummary
    model: str


class TerraioExtensionStrategy:
    """Generate an allowed repository patch against a frozen TerraIO commit."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, contract: ExtensionContract) -> RepositoryExtensionGeneration:
        """Generate and validate one complete repository-extension proposal."""

        repository_root = Path(contract.repository_path).expanduser().resolve()
        reference_context = load_repository_reference_context(
            repository_root,
            selected_file_paths=contract.context_paths,
        )
        if reference_context.base_commit != contract.base_commit:
            raise ValueError(
                "Extension contract base_commit differs from the source repository HEAD."
            )
        if reference_context.repository_url != contract.repository_url:
            raise ValueError(
                "Extension contract repository_url differs from the source repository."
            )

        system_path = PROMPTS_DIR / f"{contract.prompt_name}_system.md"
        user_path = PROMPTS_DIR / f"{contract.prompt_name}_user.md"
        for path in (system_path, user_path):
            if not path.is_file():
                raise FileNotFoundError(f"Repository extension prompt not found: {path}")

        model = getattr(self._llm, "model", "unknown")
        recorder = PromptRecorder(
            model=model,
            reasoning_effort=getattr(self._llm, "reasoning_effort", None),
        )
        recorder.add_source(system_path)
        recorder.add_source(user_path)
        system_prompt = system_path.read_text(encoding="utf-8")
        user_prompt = user_path.read_text(encoding="utf-8").format(
            extension_contract_json=json.dumps(contract.model_dump(mode="json"), indent=2),
            reference_context=reference_context.rendered_context,
        )
        usage_tracker = LLMUsageTracker()
        proposal = recorded_complete_json(
            llm=self._llm,
            recorder=recorder,
            call_name="repository_extension_generation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RepositoryExtensionResult,
            max_output_tokens=16384,
            usage_tracker=usage_tracker,
        )
        validate_extension_proposal(
            proposal,
            contract=contract,
            repository_root=repository_root,
        )
        return RepositoryExtensionGeneration(
            proposal=proposal,
            reference_context=reference_context,
            prompt_provenance=recorder.provenance(),
            llm_usage=usage_tracker.summary(),
            model=model,
        )


def validate_extension_proposal(
    proposal: RepositoryExtensionResult,
    *,
    contract: ExtensionContract,
    repository_root: Path,
) -> None:
    """Enforce path, operation, and dependency policy before applying model output."""

    changes = [*proposal.files, *proposal.tests]
    for change in changes:
        path = change.relative_path
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in contract.allowed_paths):
            raise ValueError(f"Extension change is outside allowed paths: {path}")
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in contract.protected_paths):
            raise ValueError(f"Extension change targets a protected path: {path}")
        exists = _path_exists_at_commit(repository_root, contract.base_commit, path)
        if change.operation is RepositoryChangeOperation.ADD and exists:
            raise ValueError(f"Extension ADD path already exists at the base commit: {path}")
        if change.operation is RepositoryChangeOperation.MODIFY and not exists:
            raise ValueError(f"Extension MODIFY path does not exist at the base commit: {path}")

    if contract.dependency_policy == "existing_dependencies_only":
        dependency_files = {"pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"}
        changed_dependency_files = sorted(
            path
            for path in (change.relative_path for change in changes)
            if Path(path).name in dependency_files
        )
        if changed_dependency_files:
            raise ValueError(
                "Dependency policy forbids changes to dependency declarations: "
                + ", ".join(changed_dependency_files)
            )


def _path_exists_at_commit(repository_root: Path, commit: str, relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "cat-file", "-e", f"{commit}:{relative_path}"],
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0
