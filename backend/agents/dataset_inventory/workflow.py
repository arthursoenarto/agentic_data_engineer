"""Project artifact workflow for dataset inventory generation.

Pseudocode:

DatasetCandidate
  -> DatasetInventory workflow
      -> choose best available inventory extractor
          -> deterministic extractor if provider has machine-readable metadata
          -> LLM/web-search inventory agent if no reliable extractor exists and explicitly enabled
          -> manual/minimal fallback if neither is enough
  -> project/datasets/{dataset_slug}/dataset_inventory.json
  -> ContractDraftingAgent
  -> project/datasets/{dataset_slug}/contract.json

`dataset_inventory.json` stores provider option space. `contract.json` stores
the selected data slice for the current project workflow.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from backend.agents.contract_drafting.schemas import DatasetCandidate
from backend.agents.dataset_inventory.agent import DatasetInventoryAgent
from backend.agents.dataset_inventory.extractors import CDSProcessMetadataExtractor, InventoryExtractor
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.llm import LLMClient


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS_DIR = ROOT / "project/datasets"
DEFAULT_EXTRACTORS: tuple[InventoryExtractor, ...] = (CDSProcessMetadataExtractor(),)


def slugify(value: str) -> str:
    """Create a stable filesystem slug for project artifacts."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def dataset_slug(candidate: DatasetCandidate) -> str:
    """Return the stable slug for a candidate."""

    return candidate.slug or slugify(candidate.name)


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def choose_inventory_extractor(
    candidate: DatasetCandidate,
    *,
    extractors: tuple[InventoryExtractor, ...] = DEFAULT_EXTRACTORS,
) -> InventoryExtractor | None:
    """Choose the first deterministic extractor that can inspect this candidate."""

    for extractor in extractors:
        if extractor.can_handle(candidate):
            return extractor
    return None


def minimal_inventory(candidate: DatasetCandidate, *, slug: str) -> DatasetInventory:
    """Create a minimal inventory when no extractor is available."""

    return DatasetInventory(
        dataset_slug=slug,
        title=candidate.name,
        source_url=candidate.url,
        generated_at=datetime.now(UTC).isoformat(),
        extractor_name="minimal_manual_inventory",
        extraction_method="manual",
        note="No deterministic inventory extractor was available. This inventory only records the candidate source.",
        warnings=[
            "Provider option space was not extracted. Add a deterministic extractor or enable the LLM fallback for richer inventory."
        ],
    )


def build_dataset_inventory(
    candidate: DatasetCandidate,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    root: Path = ROOT,
    allow_llm_fallback: bool = False,
    timeout_seconds: int = 180,
    prompt_name: str = "default",
) -> tuple[DatasetInventory, Path]:
    """Build and persist dataset_inventory.json for a dataset candidate."""

    slug = dataset_slug(candidate)
    extractor = choose_inventory_extractor(candidate)
    if extractor:
        inventory = extractor.extract(candidate, dataset_slug=slug)
    elif allow_llm_fallback:
        inventory = DatasetInventoryAgent(
            LLMClient(timeout_seconds=timeout_seconds),
            prompt_name=prompt_name,
        ).create_inventory(candidate, dataset_slug=slug)
    else:
        inventory = minimal_inventory(candidate, slug=slug)

    path = write_inventory_artifact(inventory, datasets_dir=datasets_dir, root=root)
    return inventory, path


def build_candidate_file_inventory(
    input_path: Path,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    root: Path = ROOT,
    allow_llm_fallback: bool = False,
    timeout_seconds: int = 180,
    prompt_name: str = "default",
) -> tuple[DatasetInventory, Path]:
    """Build dataset_inventory.json from a candidate JSON file."""

    candidate = DatasetCandidate.model_validate(json.loads(input_path.read_text(encoding="utf-8")))
    return build_dataset_inventory(
        candidate,
        datasets_dir=datasets_dir,
        root=root,
        allow_llm_fallback=allow_llm_fallback,
        timeout_seconds=timeout_seconds,
        prompt_name=prompt_name,
    )


def write_inventory_artifact(
    inventory: DatasetInventory,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    root: Path = ROOT,
) -> Path:
    """Persist a dataset inventory under the project workspace."""

    dataset_dir = datasets_dir / inventory.dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = dataset_dir / "dataset_inventory.json"
    inventory_path.write_text(json.dumps(inventory.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return inventory_path


def read_dataset_inventory(
    dataset_slug: str,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
) -> DatasetInventory | None:
    """Read a dataset inventory artifact if it exists."""

    path = datasets_dir / dataset_slug / "dataset_inventory.json"
    if not path.exists():
        return None
    return DatasetInventory.model_validate(json.loads(path.read_text(encoding="utf-8")))


def relative_inventory_path(
    inventory_path: Path,
    *,
    root: Path = ROOT,
) -> str:
    """Return a repo-relative inventory path for run metadata."""

    return _relative(inventory_path, root)
