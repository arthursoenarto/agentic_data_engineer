"""Build immutable repository-extension proposals without touching source checkouts."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from backend.agents.etl_pipeline.extension_schemas import (
    ExtensionContract,
    RepositoryExtensionManifest,
    RepositoryExtensionResult,
)
from backend.agents.etl_pipeline.extension_strategy import TerraioExtensionStrategy
from backend.agents.etl_pipeline.schemas import stable_json_hash


@dataclass(frozen=True)
class RepositoryExtensionBuild:
    """Paths and provenance for one immutable extension proposal."""

    extension_dir: Path
    manifest: RepositoryExtensionManifest
    proposal: RepositoryExtensionResult

    @property
    def extension_id(self) -> str:
        return self.manifest.extension_id


class RepositoryExtensionAgent:
    """Generate, validate, and persist PR-style repository extension proposals."""

    def __init__(self, strategy: TerraioExtensionStrategy) -> None:
        self._strategy = strategy

    def build(
        self,
        *,
        contract: ExtensionContract,
        dataset_dir: Path,
    ) -> RepositoryExtensionBuild:
        """Generate an extension and derive its patch in an independent clone."""

        dataset_dir = dataset_dir.resolve()
        extension_dir = dataset_dir / "pipelines" / contract.extension_id
        if extension_dir.exists():
            raise FileExistsError(f"Immutable extension already exists: {extension_dir}")

        generation = self._strategy.generate(contract)
        source_repository = Path(contract.repository_path).expanduser().resolve()
        patch, changed_paths = _derive_patch(
            source_repository=source_repository,
            contract=contract,
            proposal=generation.proposal,
        )
        proposal_payload = generation.proposal.model_dump(mode="json")
        contract_payload = contract.model_dump(mode="json")
        generated_artifacts = [
            "manifest.json",
            "extension_contract.json",
            "proposal.json",
            "patch.diff",
            "README.md",
            *[
                f"proposed_files/{change.relative_path}"
                for change in generation.proposal.files
            ],
            *[
                f"proposed_tests/{change.relative_path}"
                for change in generation.proposal.tests
            ],
        ]
        manifest = RepositoryExtensionManifest(
            extension_id=contract.extension_id,
            dataset_slug=contract.dataset_slug,
            model=generation.model,
            prompt_name=contract.prompt_name,
            base_commit=contract.base_commit,
            extension_contract_sha256=stable_json_hash(contract_payload),
            proposal_sha256=stable_json_hash(proposal_payload),
            patch_sha256=_text_sha256(patch),
            changed_paths=changed_paths,
            generated_artifacts=generated_artifacts,
            prompt_provenance=generation.prompt_provenance,
            reference_context=generation.reference_context.provenance(),
            llm_usage=generation.llm_usage,
        )

        try:
            extension_dir.mkdir(parents=True, exist_ok=False)
            _write_new(
                extension_dir / "extension_contract.json",
                contract.model_dump_json(indent=2) + "\n",
            )
            _write_new(
                extension_dir / "proposal.json",
                generation.proposal.model_dump_json(indent=2) + "\n",
            )
            _write_new(extension_dir / "patch.diff", patch)
            _write_new(extension_dir / "README.md", generation.proposal.readme.rstrip() + "\n")
            for change in generation.proposal.files:
                _write_new(
                    extension_dir / "proposed_files" / change.relative_path,
                    change.content,
                )
            for change in generation.proposal.tests:
                _write_new(
                    extension_dir / "proposed_tests" / change.relative_path,
                    change.content,
                )
            _write_new(
                extension_dir / "manifest.json",
                manifest.model_dump_json(indent=2) + "\n",
            )
        except Exception:
            shutil.rmtree(extension_dir, ignore_errors=True)
            raise

        return RepositoryExtensionBuild(
            extension_dir=extension_dir,
            manifest=manifest,
            proposal=generation.proposal,
        )


def validate_extension_bundle(
    extension_dir: Path,
) -> tuple[RepositoryExtensionManifest, ExtensionContract, RepositoryExtensionResult]:
    """Validate hashes and paths before an extension proposal is executed."""

    extension_dir = extension_dir.resolve()
    manifest = RepositoryExtensionManifest.model_validate_json(
        (extension_dir / "manifest.json").read_text(encoding="utf-8")
    )
    contract_payload = json.loads(
        (extension_dir / "extension_contract.json").read_text(encoding="utf-8")
    )
    proposal_payload = json.loads(
        (extension_dir / "proposal.json").read_text(encoding="utf-8")
    )
    contract = ExtensionContract.model_validate(contract_payload)
    proposal = RepositoryExtensionResult.model_validate(proposal_payload)
    patch = (extension_dir / "patch.diff").read_text(encoding="utf-8")
    if manifest.extension_id != contract.extension_id:
        raise ValueError("Extension manifest and contract IDs differ.")
    if manifest.extension_contract_sha256 != stable_json_hash(contract_payload):
        raise ValueError("Extension contract hash differs from the manifest.")
    if manifest.proposal_sha256 != stable_json_hash(proposal_payload):
        raise ValueError("Extension proposal hash differs from the manifest.")
    if manifest.patch_sha256 != _text_sha256(patch):
        raise ValueError("Extension patch hash differs from the manifest.")
    expected_paths = sorted(
        change.relative_path for change in [*proposal.files, *proposal.tests]
    )
    if sorted(manifest.changed_paths) != expected_paths:
        raise ValueError("Extension manifest changed paths differ from the proposal.")
    return manifest, contract, proposal


def _derive_patch(
    *,
    source_repository: Path,
    contract: ExtensionContract,
    proposal: RepositoryExtensionResult,
) -> tuple[str, list[str]]:
    with tempfile.TemporaryDirectory(prefix="repository_extension_build_") as temporary:
        clone = Path(temporary) / "repository"
        _git_clone(source_repository, clone)
        _git(clone, "checkout", "--detach", contract.base_commit)
        if _git(clone, "rev-parse", "HEAD").strip() != contract.base_commit:
            raise ValueError("Independent clone did not resolve the contracted base commit.")

        for change in [*proposal.files, *proposal.tests]:
            target = clone / change.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change.content, encoding="utf-8")

        _git(clone, "add", "--all")
        changed_paths = sorted(
            path
            for path in _git(clone, "diff", "--cached", "--name-only", "--").splitlines()
            if path
        )
        expected_paths = sorted(
            change.relative_path for change in [*proposal.files, *proposal.tests]
        )
        if changed_paths != expected_paths:
            raise ValueError(
                "Derived patch paths differ from the proposal; unchanged or unexpected files: "
                f"expected {expected_paths}, got {changed_paths}."
            )
        patch = _git(
            clone,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--",
        )
        if not patch.strip():
            raise ValueError("Repository extension produced an empty patch.")
        return patch, changed_paths


def _git_clone(source: Path, target: Path) -> None:
    completed = subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(source), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Independent repository clone failed: {completed.stderr.strip()}")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
