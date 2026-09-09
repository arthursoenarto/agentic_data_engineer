"""Curated read-only source context for pipeline generation strategies."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.agents.etl_pipeline.schemas import ReferenceContextProvenance


TERRAIO_COMMON_REFERENCE_FILES = (
    "README.md",
    "terraio/core/storage/store.py",
    "terraio/core/storage/catalog.py",
    "terraio/core/coverage_resolver.py",
    "terraio/core/ml/terraio_dataset.py",
    "terraio/core/ml/terraio_dataloader.py",
    "terraio/cds/backend/fetcher.py",
    "tests/core/ml/test_terraio_dataset_smoke.py",
)

# The matched TerraIOReferenced condition uses only cross-dataset architectural
# abstractions. Dataset-specific modules remain available to legacy studies.
TERRAIO_ARCHITECTURE_REFERENCE_FILES = TERRAIO_COMMON_REFERENCE_FILES

TERRAIO_DATASET_REFERENCE_FILES = {
    "reanalysis-era5-pressure-levels": (
        "terraio/cds/era5/fetcher.py",
        "terraio/cds/era5/dataset.py",
        "terraio/cds/era5/variable_registry.py",
    ),
    "cams-global-reanalysis-eac4": (
        "terraio/cds/eac4/README.md",
        "terraio/cds/eac4/fetcher.py",
        "terraio/cds/eac4/dataset.py",
        "terraio/cds/eac4/variable_registry.py",
    ),
    "MCD19A2": (
        "terraio/earthaccess/mcd19a2/README.md",
        "terraio/earthaccess/mcd19a2/fetcher.py",
        "terraio/earthaccess/mcd19a2/dataset.py",
        "terraio/earthaccess/mcd19a2/normalizer.py",
        "terraio/earthaccess/mcd19a2/variable_registry.py",
    ),
}

# Backward-compatible default for callers without a frozen dataset identity.
TERRAIO_REFERENCE_FILES = (
    *TERRAIO_COMMON_REFERENCE_FILES,
    *TERRAIO_DATASET_REFERENCE_FILES["reanalysis-era5-pressure-levels"],
)


@dataclass(frozen=True)
class ReferenceContext:
    """Rendered source context read from one exact Git commit."""

    rendered_context: str
    selected_file_paths: tuple[str, ...]
    context_size: int
    repository_url: str
    base_commit: str
    context_sha256: str

    def provenance(self) -> ReferenceContextProvenance:
        """Return serializable provenance for a pipeline generation manifest."""

        return ReferenceContextProvenance(
            repository_url=self.repository_url,
            base_commit=self.base_commit,
            selected_file_paths=list(self.selected_file_paths),
            rendered_context_size=self.context_size,
            rendered_context_sha256=self.context_sha256,
        )


def load_terraio_reference_context(
    terraio_root: Path,
    *,
    dataset_id: str | None = None,
    architecture_only: bool = False,
) -> ReferenceContext:
    """Load common architecture plus the frozen dataset's TerraIO module."""

    selected = (
        TERRAIO_ARCHITECTURE_REFERENCE_FILES
        if architecture_only
        else TERRAIO_REFERENCE_FILES
    )
    if dataset_id is not None and not architecture_only:
        dataset_files = TERRAIO_DATASET_REFERENCE_FILES.get(dataset_id)
        if dataset_files is None:
            selected = TERRAIO_COMMON_REFERENCE_FILES
        else:
            selected = (*TERRAIO_COMMON_REFERENCE_FILES, *dataset_files)

    return load_repository_reference_context(
        terraio_root,
        selected_file_paths=selected,
        display_prefix="terraio",
    )


def load_repository_reference_context(
    repository_root: Path,
    *,
    selected_file_paths: tuple[str, ...] | list[str],
    display_prefix: str | None = None,
) -> ReferenceContext:
    """Render selected tracked text files from the repository's exact HEAD commit."""

    repository_root = repository_root.resolve()
    if not repository_root.is_dir():
        raise RuntimeError(f"Reference repository not found: {repository_root}")

    base_commit = _git(repository_root, "rev-parse", "HEAD").strip()
    try:
        repository_url = _git(repository_root, "remote", "get-url", "origin").strip()
    except RuntimeError:
        repository_url = repository_root.as_uri()

    sections: list[str] = []
    selected_paths: list[str] = []

    for relative_path in selected_file_paths:
        safe_path = PurePosixPath(relative_path)
        if safe_path.is_absolute() or ".." in safe_path.parts or not safe_path.parts:
            raise ValueError(f"Unsafe reference context path: {relative_path!r}")
        content = _git(repository_root, "show", f"{base_commit}:{safe_path.as_posix()}")
        display_path = (
            (PurePosixPath(display_prefix) / safe_path).as_posix()
            if display_prefix
            else safe_path.as_posix()
        )
        selected_paths.append(display_path)
        sections.append(
            f'<reference_file path="{display_path}">\n'
            f"{content}\n"
            "</reference_file>"
        )

    rendered_context = "\n\n".join(sections)
    return ReferenceContext(
        rendered_context=rendered_context,
        selected_file_paths=tuple(selected_paths),
        context_size=len(rendered_context),
        repository_url=repository_url,
        base_commit=base_commit,
        context_sha256=hashlib.sha256(rendered_context.encode("utf-8")).hexdigest(),
    )


def _git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Git command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout
