# Project Workspace

This folder contains the current user's data engineering work. `backend/` is reusable framework code; `project/` is the project state that agents and humans inspect together.

## Structure

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

## Folders

- `intake/` holds project-level goals, constraints, and success criteria. It is not dataset-specific.
- `datasets/` holds one folder per dataset. Keep each dataset's candidate, optional source inventory, contract, credential references, access probe, plan, pipeline, data, runs, and reports together.
- `integration/` is reserved for work that combines multiple datasets, such as cross-dataset joins, alignment, and reports.

Dataset folder names use the candidate `slug` when provided. If the human only provides a URL, the backend derives a name and slug from the URL.

## Current Flow

```text
datasets/{dataset_slug}/candidate.json
  -> datasets/{dataset_slug}/dataset_inventory.json
  -> backend ContractDraftingAgent
  -> datasets/{dataset_slug}/contract.json
```

`dataset_inventory.json` stores the provider option space, such as available variables, pressure levels, bands, tables, or formats. `contract.json` stores the selected fields and scope for the current workflow.

For datasets with selectors such as pressure level, band, depth, station, or lead time, each exact field-selector combination is one selected field. Feature/target roles belong to a later ML-view artifact, not the acquisition contract.

Reusable inventory extractor code lives in `backend/agents/dataset_inventory/`. Project folders store only the generated inventory artifact for that dataset.

Run from the repository root:

```bash
python3 backend/scripts/generate_dataset_inventory.py
python3 backend/scripts/draft_dataset_contract.py
```

Do not store secret values in project artifacts.
