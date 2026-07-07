# AGENTS.md

Operational instructions for coding agents working in this repository. Agents must read it and suggest edits if needed, but must NOT edit it unless explicitly asked by the humans. 

Note: keep this page about 100 lines or less.

## First Files To Read

1. `README.md`
2. `project/README.md`
3. Relevant agent or frontend files for the task

## Core Architecture Rule

Keep the reusable framework separate from the current user's project workspace.

```text
backend/  = reusable agentic data engineering framework
project/  = current user's project artifacts, generated code, data, and runs
frontend/ = human UI over project artifacts
```

Do not store project-specific artifacts in `backend/` unless they are test fixtures.

## Read-Only Reference Code

`terraio/` is a human-written reference implementation for scientific data engineering patterns, including ERA5 pressure-level retrieval and ML-ready dataset assembly. Agents may read it for architecture and product context, but must not edit, format, move, delete, or generate files under `terraio/`.

## Human-AI Collaboration / UX Rule

Minimize human decision load. Agents should draft defaults and ask humans only for irreducible project choices: intent, dataset scope, selected fields, date/time range, geography, and output goal.

Prefer small default contracts, progressive disclosure, and advanced/raw controls as escape hatches. Avoid blank forms, large questionnaires, and exposing provider/API/storage/pipeline internals by default.

Inventory stores provider-native options. UI may derive human-friendly controls. Contracts store explicit provider-native selected fields and scope values. For datasets with selectors such as pressure level, band, depth, station, or lead time, each exact field-selector combination is one selected field; feature/target roles belong to a later ML-view contract, not the acquisition contract.

## Agent Role

Agents do not replace deterministic data engineering code.

Agents may:

- interpret goals
- discover dataset candidates
- draft dataset contracts
- plan pipelines
- generate deterministic code
- inspect outputs
- validate and document results

Deterministic code should perform API calls, downloads, transformations, storage writes, validation checks, and pipeline execution.

## Prompt Variants

Store agent prompts under the owning agent, for example:

```text
backend/agents/{agent_name}/prompts/{variant}_system.md
backend/agents/{agent_name}/prompts/{variant}_user.md
```

Prompt variants are part of the MSc research surface. Make variants selectable, keep the default stable, and record which variant/model/tools were used for experiment outputs.

## File Ownership

Use `backend/` for:

- agent implementations
- LLM client code
- reusable schemas
- reusable access-probe generation and runner logic
- reusable dataset inventory extractors under `backend/agents/dataset_inventory/extractors/`
- orchestration scripts
- tests

Use `project/` for:

- project request/intake artifacts
- dataset folders under `project/datasets/{dataset_slug}/`
- dataset-local candidates, optional source inventories, contracts, credential references, generated access probe scripts, probe results, plans, pipelines, data, runs, and reports
- cross-dataset integration artifacts under `project/integration/` when needed

Use `frontend/` for:

- UI code
- current product surface over project artifacts
- frontend documentation

## Research Experiment Rule

Keep the main product path stable. Isolate experiments from default behavior, preserve baselines, avoid overwriting prior outputs, and document the goal, hypothesis, configuration, model, prompts, tools, dataset/task, and result.

## Development Priorities

Prefer:

- small, inspectable changes
- typed interfaces
- structured agent outputs
- explicit assumptions
- provenance-preserving artifacts
- tests for deterministic behavior
- generated code that can run without an LLM

Avoid:

- hidden agent behavior
- one-off scripts with no reusable boundary
- project artifacts written into `backend/`
- transformations without validation
- storing secret values in code, JSON, HTML, logs, or generated artifacts
- broad rewrites unrelated to the current task

## Update Protocol

After non-trivial changes, update relevant code, tests, and docs. Explain what changed, what was verified, and any remaining risks in the final response.
