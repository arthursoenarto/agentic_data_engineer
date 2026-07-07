"""Project artifact workflow for dataset contract drafting."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel

from backend.agents.contract_drafting.agent import ContractDraftingAgent
from backend.agents.contract_drafting.schemas import DatasetCandidate, DatasetCandidateInput, DatasetContract
from backend.agents.dataset_inventory import build_dataset_inventory, relative_inventory_path
from backend.llm import LLMClient


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS_DIR = ROOT / "project/datasets"


class DatasetContractRun(BaseModel):
    """File-backed result from a dataset contract drafting run."""

    schema_version: str = "dataset_contract_run.v1"
    candidate_file: str
    contract_file: str
    inventory_file: str | None = None
    generated_at: str
    contract: DatasetContract
    candidate: DatasetCandidate | None = None


class DatasetContractList(BaseModel):
    """Existing dataset contracts available to the frontend."""

    schema_version: str = "dataset_contract_list.v1"
    contracts: list[DatasetContractRun]


def slugify(value: str) -> str:
    """Create a stable filesystem slug for project artifacts."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "dataset"


def dataset_slug(candidate: DatasetCandidate) -> str:
    """Return the stable slug for a candidate."""

    return candidate.slug or slugify(candidate.name)


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        return path_parts[-1].replace("-", " ").replace("_", " ").strip().title()
    if parsed.netloc:
        return parsed.netloc.removeprefix("www.").split(".")[0].replace("-", " ").title()
    return "Dataset"


def normalize_candidate_input(candidate_input: DatasetCandidateInput) -> DatasetCandidate:
    """Normalize flexible human input into the strict candidate."""

    if candidate_input.url is None:
        raise ValueError("A dataset URL is required for contract drafting.")

    url = str(candidate_input.url)
    name = candidate_input.name.strip() if candidate_input.name else _name_from_url(url)
    return DatasetCandidate(
        name=name,
        url=candidate_input.url,
        slug=candidate_input.slug or slugify(name),
        description=candidate_input.description,
        user_goal=candidate_input.user_goal,
    )


def _relative(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_candidate_artifact(
    candidate: DatasetCandidate,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
) -> Path:
    """Persist a user-provided candidate under the project workspace."""

    dataset_dir = datasets_dir / dataset_slug(candidate)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = dataset_dir / "candidate.json"
    candidate_path.write_text(json.dumps(candidate.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return candidate_path


def write_contract_artifact(
    *,
    candidate: DatasetCandidate,
    contract: DatasetContract,
    candidate_path: Path,
    inventory_path: Path | None = None,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    root: Path = ROOT,
) -> DatasetContractRun:
    """Persist a contract draft under the project workspace."""

    dataset_dir = datasets_dir / dataset_slug(candidate)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    contract_path = dataset_dir / "contract.json"
    normalized_contract = contract.model_copy(update={"dataset_slug": dataset_slug(candidate)})
    run = DatasetContractRun(
        candidate_file=_relative(candidate_path, root),
        contract_file=_relative(contract_path, root),
        inventory_file=_relative(inventory_path, root) if inventory_path else None,
        generated_at=datetime.now(UTC).isoformat(),
        contract=normalized_contract,
        candidate=candidate,
    )
    contract_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return run


def draft_dataset_contract(
    candidate: DatasetCandidate,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    timeout_seconds: int = 180,
    prompt_name: str = "default",
    allow_llm_inventory_fallback: bool = False,
    root: Path = ROOT,
) -> DatasetContractRun:
    """Draft a contract and write the candidate/contract artifacts."""

    candidate_path = write_candidate_artifact(candidate, datasets_dir=datasets_dir)
    inventory, inventory_path = build_dataset_inventory(
        candidate,
        datasets_dir=datasets_dir,
        root=root,
        allow_llm_fallback=allow_llm_inventory_fallback,
        timeout_seconds=timeout_seconds,
    )
    contract = ContractDraftingAgent(
        LLMClient(timeout_seconds=timeout_seconds),
        prompt_name=prompt_name,
    ).draft_contract(candidate, inventory=inventory)
    return write_contract_artifact(
        candidate=candidate,
        contract=contract,
        candidate_path=candidate_path,
        inventory_path=inventory_path,
        datasets_dir=datasets_dir,
        root=root,
    )


def draft_candidate_file(
    input_path: Path,
    *,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    timeout_seconds: int = 180,
    prompt_name: str = "default",
    allow_llm_inventory_fallback: bool = False,
    root: Path = ROOT,
) -> DatasetContractRun:
    """Draft a contract from a candidate JSON file without rewriting the input file."""

    candidate = DatasetCandidate.model_validate(json.loads(input_path.read_text(encoding="utf-8")))
    inventory, inventory_path = build_dataset_inventory(
        candidate,
        datasets_dir=datasets_dir,
        root=root,
        allow_llm_fallback=allow_llm_inventory_fallback,
        timeout_seconds=timeout_seconds,
    )
    contract = ContractDraftingAgent(
        LLMClient(timeout_seconds=timeout_seconds),
        prompt_name=prompt_name,
    ).draft_contract(candidate, inventory=inventory)
    return write_contract_artifact(
        candidate=candidate,
        contract=contract,
        candidate_path=input_path,
        inventory_path=inventory_path,
        datasets_dir=datasets_dir,
        root=root,
    )


def _read_contract_file(path: Path, *, root: Path = ROOT) -> DatasetContractRun | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "dataset_contract_run.v1":
        run = DatasetContractRun.model_validate(payload)
        if run.candidate is None and run.candidate_file:
            candidate_path = root / run.candidate_file
            if candidate_path.exists():
                run.candidate = DatasetCandidate.model_validate(json.loads(candidate_path.read_text(encoding="utf-8")))
        if run.inventory_file is None:
            inventory_path = path.parent / "dataset_inventory.json"
            if inventory_path.exists():
                run.inventory_file = relative_inventory_path(inventory_path, root=root)
        return run

    if "contract" in payload:
        return DatasetContractRun(
            candidate_file=payload.get("candidate_file", ""),
            contract_file=_relative(path, root),
            generated_at=payload.get("generated_at", ""),
            contract=DatasetContract.model_validate(payload["contract"]),
        )

    return None


def read_contract_runs(
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    *,
    root: Path = ROOT,
) -> DatasetContractList:
    """Read generated dataset contracts from dataset folders."""

    runs: list[DatasetContractRun] = []
    if datasets_dir.exists():
        for path in sorted(datasets_dir.glob("*/contract.json")):
            run = _read_contract_file(path, root=root)
            if run:
                runs.append(run)

    runs.sort(key=lambda run: run.generated_at or "", reverse=True)
    return DatasetContractList(contracts=runs)
