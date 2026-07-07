# Agentic Data Engineer

## Run Locally

Backend:

```bash
python3 -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`.

## Core Principle

Build a reusable agentic data engineering framework that helps a human move from a dataset idea to deterministic, inspectable data pipelines.

Keep framework code separate from project state:

```text
backend/  = reusable agentic data engineering framework
project/  = current user's project artifacts, generated code, data, and runs
frontend/ = human UI over project artifacts
```

Agents coordinate, draft, search, plan, and generate code. Deterministic code performs API calls, downloads, transformations, storage writes, validation checks, and pipeline execution.

## Human-AI Collaboration

The product should minimize human decision load. Agents draft sensible defaults, then humans edit or confirm only decisions that materially affect their goal: intent, dataset scope, selected fields, time range, geography, and output needs.

Default views should stay simple. Provider/API/storage/pipeline mechanics belong in agent output, advanced controls, or later planning steps unless the human explicitly needs them.

## Current Flow

The active implemented flow is contract-first:

```text
project/datasets/{dataset_slug}/candidate.json
  -> project/datasets/{dataset_slug}/dataset_inventory.json
  -> ContractDraftingAgent
  -> project/datasets/{dataset_slug}/contract.json
  -> credential references
  -> AccessProbeGenerationAgent
  -> project/datasets/{dataset_slug}/access_probe.py
  -> project/datasets/{dataset_slug}/access_probe.json
```

Run the contract drafter from the repository root:

```bash
python3 backend/scripts/generate_dataset_inventory.py \
  --input project/datasets/reanalysis_era5_pressure_levels/candidate.json \
  --datasets-dir project/datasets

python3 backend/scripts/draft_dataset_contract.py \
  --input project/datasets/openaq/candidate.json \
  --datasets-dir project/datasets
```

The first step assumes the human provides a dataset candidate. Automated dataset discovery can be added later as an upstream agent that writes candidates.

`dataset_inventory.json` is provider option space, such as all available variables, levels, bands, tables, or formats. The backend chooses the best available inventory extractor: deterministic official metadata first, optional LLM/web fallback when enabled, then a minimal manual fallback. `contract.json` should stay small and contain only the selected fields and scope for the current workflow.

Inventory, UI, and contract have separate responsibilities:

```text
dataset_inventory.json stores provider-native options.
The UI may derive human-friendly controls from those options.
contract.json stores explicit provider-native selected fields and scope values.
```

For datasets with selectors such as pressure level, band, depth, station, or lead time, each exact field-selector combination is one selected field. Feature/target roles are intentionally excluded from `contract.json`; they belong to a later ML-view contract when the user is preparing model-ready tensors.

## Local API And Frontend

Install backend requirements:

```bash
python3 -m pip install -r backend/requirements.txt
```

The FastAPI backend exposes:

```text
GET  /api/health
GET  /api/dataset-contracts
POST /api/dataset-contracts
GET  /api/datasets/{slug}/contract
PUT  /api/datasets/{slug}/contract
GET  /api/datasets/{slug}/inventory
GET  /api/datasets/{slug}/credentials
PUT  /api/datasets/{slug}/credentials
POST /api/datasets/{slug}/access-probe
```

The frontend submits flexible candidate input to `POST /api/dataset-contracts`. A URL is required for now; name, slug, description, and user goal are optional. The backend normalizes the input, builds an inventory, runs the reusable contract drafting agent, and writes:

```text
project/datasets/{dataset_slug}/candidate.json
project/datasets/{dataset_slug}/dataset_inventory.json
project/datasets/{dataset_slug}/contract.json
```

Frontend routes:

```text
/             = dataset workspace with collapsible dataset rail and tabs
/datasets/new = dataset candidate input and contract drafting trigger
```

## Project Workspace

`project/` is the current user's workspace:

```text
project/
├── intake/
├── datasets/
│   └── {dataset_slug}/
│       ├── candidate.json
│       ├── dataset_inventory.json
│       ├── contract.json
│       ├── credentials.json
│       ├── access_probe.py
│       ├── access_probe.metadata.json
│       ├── access_probe.json
│       ├── pipeline_plan.json
│       ├── human_plan.md
│       ├── pipeline/
│       ├── data/
│       ├── runs/
│       └── reports/
└── integration/
```

Dataset folder names come from `DatasetCandidate.slug` when provided. If no slug is provided, the backend derives one from the candidate name or URL.

## Credentials And Probes

The frontend must never receive secret values. Credential setup stores references only, such as `OPENAQ_API_KEY`, `CDSAPI_KEY`, or `~/.cdsapirc`.

Dataset-specific access probe code lives under `project/datasets/{dataset_slug}/access_probe.py`. If it is missing, the backend asks the access-probe generation agent to create a small deterministic script from the dataset contract and credential requirements.

## Research And Experiments

This is also an MSc thesis system. It should support controlled experiments such as agent-design comparisons, prompt variants, model/tool variations, ablation studies, and evaluation runs.

Each agent should keep switchable prompt variants under its own `prompts/` folder, for example:

```text
backend/agents/contract_drafting/prompts/default_system.md
backend/agents/contract_drafting/prompts/default_user.md
```

Keep the main product path stable. Isolate experiments, preserve baselines, and record the prompt/model/tool configuration used for each evaluation.

## Design Rules

- Keep agents as control-plane components.
- Keep deterministic code responsible for execution.
- Store raw data before transformation.
- Preserve provenance and source metadata.
- Store credential names, never secret values.
- Keep generated code inspectable and runnable without an LLM.
- Prefer small first contracts over broad dataset inventories.
- Keep validation and reporting as explicit workflow stages.
