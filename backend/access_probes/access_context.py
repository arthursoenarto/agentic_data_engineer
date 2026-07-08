"""Read and write dataset-local access context artifacts."""

from __future__ import annotations

from pathlib import Path

from backend.access_probes.schemas import AccessContext

ACCESS_CONTEXT_FILENAME = "access_context.json"


def access_context_path(dataset_dir: Path) -> Path:
    """Return the dataset-local access context path."""

    return dataset_dir / ACCESS_CONTEXT_FILENAME


def read_access_context(dataset_dir: Path) -> AccessContext | None:
    """Read dataset-local access context if it exists."""

    path = access_context_path(dataset_dir)
    if not path.exists():
        return None
    return AccessContext.model_validate_json(path.read_text(encoding="utf-8"))


def write_access_context(dataset_dir: Path, context: AccessContext) -> Path:
    """Persist dataset-local access context without secret values."""

    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = access_context_path(dataset_dir)
    path.write_text(context.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
