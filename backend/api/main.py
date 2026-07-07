"""FastAPI surface for project dataset contract drafting."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.contract_drafting import DatasetCandidateInput, DatasetContract
from backend.agents.contract_drafting.workflow import (
    DEFAULT_DATASETS_DIR,
    DatasetContractList,
    DatasetContractRun,
    draft_dataset_contract,
    normalize_candidate_input,
    read_contract_runs,
)
from backend.agents.dataset_inventory import DatasetInventory, read_dataset_inventory
from backend.credentials import (
    CredentialStatus,
    CredentialUpdate,
    AccessProbeResult,
    ENV_FILE,
    credential_status,
    run_access_probe,
    save_credentials,
)


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = DEFAULT_DATASETS_DIR
ENV_PATH = ENV_FILE


def _allowed_origins() -> list[str]:
    raw = os.environ.get("FRONTEND_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Agentic Data Engineer API",
    version="0.1.0",
    description="Local API for reusable agentic data engineering workflows.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "agentic-data-engineer-api"}


@app.get("/api/dataset-contracts", response_model=DatasetContractList)
def list_dataset_contracts() -> DatasetContractList:
    return read_contract_runs(DATASETS_DIR, root=ROOT)


@app.post("/api/dataset-contracts", response_model=DatasetContractRun)
def create_dataset_contract(candidate_input: DatasetCandidateInput) -> DatasetContractRun:
    try:
        candidate = normalize_candidate_input(candidate_input)
        return draft_dataset_contract(
            candidate,
            datasets_dir=DATASETS_DIR,
            root=ROOT,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not write project artifacts: {error}") from error


def _dataset_run(dataset_slug: str) -> tuple[DatasetContractRun, Path]:
    dataset_dir = DATASETS_DIR / dataset_slug
    contract_path = dataset_dir / "contract.json"
    if not contract_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset contract not found for slug: {dataset_slug}")

    matching = [
        run
        for run in read_contract_runs(DATASETS_DIR, root=ROOT).contracts
        if run.contract_file == str(contract_path.relative_to(ROOT))
    ]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Dataset contract not readable for slug: {dataset_slug}")
    return matching[0], dataset_dir


@app.get("/api/datasets/{dataset_slug}/contract", response_model=DatasetContractRun)
def get_dataset_contract(dataset_slug: str) -> DatasetContractRun:
    run, _dataset_dir = _dataset_run(dataset_slug)
    return run


@app.put("/api/datasets/{dataset_slug}/contract", response_model=DatasetContractRun)
def put_dataset_contract(dataset_slug: str, contract: DatasetContract) -> DatasetContractRun:
    run, dataset_dir = _dataset_run(dataset_slug)
    contract_path = dataset_dir / "contract.json"
    updated_contract = contract.model_copy(update={"dataset_slug": dataset_slug})
    updated_run = run.model_copy(
        update={
            "generated_at": datetime.now(UTC).isoformat(),
            "contract": updated_contract,
        }
    )

    try:
        contract_path.write_text(json.dumps(updated_run.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not save dataset contract: {error}") from error

    return updated_run


@app.get("/api/datasets/{dataset_slug}/inventory", response_model=DatasetInventory)
def get_dataset_inventory(dataset_slug: str) -> DatasetInventory:
    inventory = read_dataset_inventory(dataset_slug, datasets_dir=DATASETS_DIR)
    if inventory is None:
        raise HTTPException(status_code=404, detail=f"Dataset inventory not found for slug: {dataset_slug}")
    return inventory


@app.get("/api/datasets/{dataset_slug}/credentials", response_model=CredentialStatus)
def get_dataset_credentials(dataset_slug: str) -> CredentialStatus:
    run, dataset_dir = _dataset_run(dataset_slug)
    return credential_status(dataset_slug=dataset_slug, run=run, dataset_dir=dataset_dir, env_path=ENV_PATH)


@app.put("/api/datasets/{dataset_slug}/credentials", response_model=CredentialStatus)
def put_dataset_credentials(dataset_slug: str, update: CredentialUpdate) -> CredentialStatus:
    run, dataset_dir = _dataset_run(dataset_slug)
    try:
        save_credentials(update, dataset_dir=dataset_dir)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not save credential references: {error}") from error
    return credential_status(dataset_slug=dataset_slug, run=run, dataset_dir=dataset_dir, env_path=ENV_PATH)


@app.post("/api/datasets/{dataset_slug}/access-probe", response_model=AccessProbeResult)
def post_dataset_access_probe(dataset_slug: str) -> AccessProbeResult:
    run, dataset_dir = _dataset_run(dataset_slug)
    return run_access_probe(dataset_slug=dataset_slug, run=run, dataset_dir=dataset_dir, env_path=ENV_PATH)
