# Agentic Data Engineer

## Setup

The thesis artifact is a library and CLI workflow; the historical HTTP/frontend
surface is out of scope. Install reusable and benchmark dependencies from the
repository root:

```bash
python3 -m pip install -r backend/requirements-benchmark.txt
```

Run the complete generation-to-evaluation workflow with a versioned config:

```bash
python3 backend/scripts/run_stage_a_workflow.py \
  --config project/datasets/{dataset_slug}/benchmarks/experiments/{experiment}/{condition}.json
```

Run a bounded evaluation-guided search from an optimization-ready Stage A
candidate:

```bash
python3 backend/scripts/run_self_improvement.py \
  --config project/datasets/{dataset_slug}/benchmarks/experiments/{experiment}/{search}.json
```

## Core Principle

Build a reusable agentic data engineering framework that helps a human move from a dataset idea to deterministic, inspectable data pipelines.

Keep framework code separate from project state:

```text
backend/  = reusable agentic data engineering framework
project/  = ignored local workspace for generated code, data, and runs
thesis/   = ignored local thesis source, notes, results, and figures
```

Agents coordinate, draft, search, plan, and generate code. Deterministic code performs API calls, downloads, transformations, storage writes, validation checks, and pipeline execution.

## Human-AI Collaboration

The product should minimize human decision load. Agents draft sensible defaults, then humans edit or confirm only decisions that materially affect their goal: intent, dataset scope, selected fields, time range, geography, and output needs.

Default views should stay simple. Provider/API/storage/pipeline mechanics belong in agent output, advanced controls, or later planning steps unless the human explicitly needs them.

## Current Flow

The proposed product path now generates one reusable adapter for a frozen
dataset inventory:

```text
DatasetCandidate -> DatasetInventory -> DatasetContract draft
  -> contracts/contract.yaml
  -> deterministic validation and locking
  -> contracts/contract_vN.lock.json
  -> PipelineGenerationAgent.generate(...)
  -> pipelines/{pipeline_id}/
  -> PipelineGenerationAgent.execute_family_pipeline(...)
  -> pipelines/{pipeline_id}/runs/{pipeline_run_id}/
       output/ + pipeline_run.json + logs/ + repairs/
  -> shared data/cache/
  -> immutable generation acceptance
  -> independent evaluation
  -> complete constrained objective vector F(p)
```

`backend/orchestration/stage_a.py` implements this path as one deterministic
state machine around public subsystem interfaces. Each run records generation,
runtime dependency freeze, advisory candidate-test evidence, every bounded
repair, live preparation, acceptance, alternate-contract diagnostics, suite
construction, and evaluation in
`runs/stage_a/{workflow_id}/stage_a_run.json`. It also hashes the workflow
configuration, declared inputs, active backend source, runtime, and terminal
artifacts. Alternate-contract failure affects extensibility evidence but does
not suppress a correct primary objective vector. Candidate dependency
installation and primary execution share one global repair budget; acceptance
and independent evaluation remain non-repairing gates.

`contract_specialized` generation and historical `contract.json` /
`pipeline_*` artifacts remain available as research baselines. They are not
rewritten or migrated destructively.

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

`dataset_inventory.json` is provider option space, such as all available variables, levels, bands, tables, or formats. The backend chooses the best available inventory extractor: deterministic official metadata first, optional LLM/web fallback when enabled, then a minimal manual fallback. New editable contracts use `contracts/contract.yaml`; generated pipelines consume only immutable `contract_vN.lock.json` snapshots.

Inventory, presentation, and contract have separate responsibilities:

```text
dataset_inventory.json stores provider-native options.
A presentation layer may derive human-friendly controls from those options.
contract.yaml stores explicit provider-native selected fields and scope values.
contract_vN.lock.json freezes the validated, defaults-resolved execution request.
```

For datasets with selectors such as pressure level, band, depth, station, or lead time, each exact field-selector combination is one selected field. Feature/target roles are intentionally excluded from dataset contracts; they belong to a later ML-view contract when the user is preparing model-ready tensors.

## Thesis Scope

The active artifact covers inventory and contract preparation, pipeline
generation, bounded repair, deterministic execution, generation acceptance,
independent evaluation, and the development self-improvement controller.
Frontend and HTTP API development are intentionally excluded.

## Project Workspace

`project/` is the current user's local workspace. Its contents are excluded
from version control:

```text
project/
├── intake/
├── datasets/
│   └── {dataset_slug}/
│       ├── candidate.json
│       ├── dataset_inventory.json
│       ├── credentials.json
│       ├── access_probe.py
│       ├── access_probe.metadata.json
│       ├── access_probe.json
│       ├── contracts/
│       │   ├── contract.yaml
│       │   └── contract_vN.lock.json
│       ├── pipelines/{pipeline_id}/
│       │   ├── generated source, tests, and generation metadata
│       │   └── runs/{pipeline_run_id}/
│       │       ├── output/
│       │       ├── pipeline_run.json
│       │       ├── contract.lock.json
│       │       ├── logs/
│       │       └── repairs/
│       ├── data/cache/
│       ├── runs/evaluations/{evaluation_run_id}/
│       ├── benchmarks/
│       └── reports/
└── integration/
```

Dataset folder names come from `DatasetCandidate.slug` when provided. If no slug is provided, the backend derives one from the candidate name or URL.

## Credentials And Probes

Project artifacts must never contain secret values. Credential setup stores references only, such as `OPENAQ_API_KEY`, `CDSAPI_KEY`, or `~/.cdsapirc`.

Dataset-specific access probe code lives under `project/datasets/{dataset_slug}/access_probe.py`. If it is missing, the backend asks the access-probe generation agent to create a small deterministic script from the dataset contract and credential requirements.

## Research And Experiments

This is also an MSc thesis system. It should support controlled experiments such as agent-design comparisons, prompt variants, model/tool variations, ablation studies, and evaluation runs.

Each agent should keep switchable prompt variants under its own `prompts/` folder, for example:

```text
backend/agents/contract_drafting/prompts/default_system.md
backend/agents/contract_drafting/prompts/default_user.md
```

Keep the main product path stable. Isolate experiments, preserve baselines, and record the prompt/model/tool configuration used for each evaluation.

Orchestration and experimental treatment are separate:

| Mechanism | Research status |
| --- | --- |
| `direct_llm` | Primary one-call mechanism. |
| `terraio_direct` | Primary one-call mechanism with frozen reference context. |
| `staged_llm`, `terraio_staged` | Preserved legacy staged-reasoning ablations. |
| `template_hybrid` | Preserved optional scaffold-enforcement ablation. |

The primary standalone conditions are `naive_llm`, `expert_direct_llm`, and
`terraio_referenced`. The first two are frozen prompt variants over the same
`direct_llm` mechanism. Expert Direct and TerraIO Referenced use the same shared
expert prompt source; the intended difference is curated TerraIO context only.

Optional versioned concise conditions preserve the frozen primary prompts while
testing trace-driven prompt revisions. Their development evidence is recorded in
`project/benchmarks/prompt_concision_20260903_v1/`.

New manifests record condition metadata, exact rendered prompts and hashes,
prompt-source hashes, model/request budgets, aggregate token usage, verified
pricing, and reference commit/context provenance. Historical prompts, strategies,
artifacts, and evidence remain unchanged.

Matched experiments additionally freeze all three prompt-template source hashes
with `freeze_pipeline_prompts.py`. Generation validates that lock before any LLM
call and against the resulting manifest. Trial records report `success@1` and
success after at most three repairs over a balanced condition/repetition matrix;
pipeline directory suffixes are never treated as attempt counts.

## Pipeline Generation And Repair

`PipelineGenerationAgent` exposes `generate(...)` and `repair(...)` as two
role-specialized capabilities of one public agent. Generation receives the
specification, inventory, policy, prompts, and optional reference context.
Repair receives only the materialized candidate, deterministic execution
evidence, immutable inputs, and previous repair error. Durable artifacts are
the shared memory; no hidden conversational state is carried between calls.

The repair capability sends concrete failure output and current pipeline source
to the LLM, applies exact minimal text replacements, and reruns the command in a
fresh execution workspace. `--output-dir` and `--run-receipt` are isolated per
execution while cache and frozen inputs remain stable. It stops immediately on
success and never makes more than three repair calls. `ETLPipelineAgent` remains
an import-compatible alias, and `build_pipeline(...)` remains the generation-only
baseline for research ablations. Generated pipelines are deterministic and have
no runtime LLM dependency.

For diagnostics against an already generated pipeline, the low-level repair CLI accepts the direct executable and arguments after `--`:

```bash
python3 backend/scripts/repair_pipeline.py \
  --pipeline-dir project/datasets/{dataset_slug}/pipeline \
  -- python3 pipeline.py
```

Repair logs are isolated from the immutable generation manifest. Historical
specialized runs keep their pipeline-local layout; family verification may
place repair evidence under the immutable failed pipeline run:

```text
project/datasets/{dataset_slug}/pipelines/{pipeline_id}/runs/{pipeline_run_id}/
└── repairs/repair_{timestamp}/repair_log.json
```

Each log records the initial execution, every proposed patch and rerun, per-attempt token/cost usage, aggregate repair-only usage, prompt name, model, and final status. Secret values are redacted. A pipeline that succeeds initially produces a zero-attempt log and uses no repair tokens.

## Reusable Pipeline Interface

All five strategies use a versioned family interface. Historical and non-Zarr
families remain readable on `family_pipeline_interface.v1`/`v2`; newly generated
regular-grid Zarr families use `family_pipeline_interface.v3`.
The framework owns `run_pipeline.py`, `pipeline_contract.json`, and
`manifest.json`; model output supplies `pipeline_impl.py`, dependencies, and
diagnostic tests. The common execution boundary is:

```bash
python run_pipeline.py \
  --contract contracts/contract_vN.lock.json \
  --inventory dataset_inventory.json \
  --cache-dir data/cache \
  --output-dir pipelines/{pipeline_id}/runs/{pipeline_run_id}/output \
  --run-receipt pipelines/{pipeline_id}/runs/{pipeline_run_id}/pipeline_run.json
```

`pipeline_contract.json` declares accepted schemas, credential environment
variable names, command placeholders, output artifact semantics, and fixed
acquisition/output policy. `manifest.json` records generation provenance,
inventory/seed hashes, condition and strategy status, exact prompt/context
provenance, model request budgets, token usage, and estimated cost.
`pipeline_run.json` records one execution, exact input hashes, cache evidence,
output fingerprints, logs, any repair reference, and for v2/v3 a neutral
`dataset_artifact_layout.v1` describing Zarr stores, arrays, dimensions,
coordinates, canonical field IDs, and selectors. It contains no evaluation
scores.

The v3 interface fixes publication to Zarr format 3 with consolidated metadata.
Generated dependencies must require `zarr>=3.1,<4`, and the framework verifies
the root `zarr.json`, complete consolidated node metadata, and consolidated
reopen. Zarr-Python currently warns that v3 consolidated metadata is experimental;
the thesis policy accepts that portability limitation to keep footprint
comparisons structurally fair.

Use the framework CLIs from the repository root:

```bash
python3 backend/scripts/lock_dataset_contract.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --initialize-from project/datasets/{dataset_slug}/contract.json

python3 backend/scripts/generate_family_pipeline.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --seed-contract project/datasets/{dataset_slug}/contracts/contract_v1.lock.json \
  --condition expert_direct_llm

python3 backend/scripts/run_family_pipeline.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --pipeline-id {pipeline_id} \
  --contract-lock project/datasets/{dataset_slug}/contracts/contract_v1.lock.json \
  --cache-dir project/datasets/{dataset_slug}/data/cache

python3 backend/scripts/accept_family_pipeline.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --pipeline-id {pipeline_id} \
  --contract-lock project/datasets/{dataset_slug}/contracts/contract_v1.lock.json \
  --prepared-cache project/datasets/{dataset_slug}/data/cache
```

Generation-owned acceptance never edits candidate source. It runs generated
tests as advisory diagnostics, then independently requires two credential-free
executions against an immutable external cache, genuine readable Zarr output,
the fixed Zarr v3 consolidated policy, canonical artifact metadata, exact rerun fingerprints, cache hits without
acquisition, and unchanged source/cache hashes. Only a passed acceptance report
should be registered with evaluation.

## Repository Extension Case Study

`terraio_extension` is a separate PR-style case study, not a standalone pipeline
condition. It generates typed allowed file changes against a pinned commit,
derives `patch.diff` in an independent clone, and runs frozen baseline,
interface-import, test, lint, typecheck, documentation, and representative
workflow commands. It never writes to the authoritative `terraio/` checkout.

```bash
python3 backend/scripts/generate_terraio_extension.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --extension-contract project/datasets/{dataset_slug}/extension_requests/{request}.json

python3 backend/scripts/run_terraio_extension.py \
  --dataset-dir project/datasets/{dataset_slug} \
  --extension-id {terraio_extension_id}
```

Extension proposals and their immutable clone-run evidence live together under
`pipelines/{terraio_extension_id}/runs/{run_id}/`.

## Subsystem Boundaries And Orchestration

Pipeline generation and pipeline evaluation are independent components connected through artifacts, not each other's implementation details:

```text
DatasetContract
      |
      v
Generation subsystem -> pipeline + data artifacts
                               |
                      neutral target mapping
                               |
                               v
Evaluation subsystem -> immutable measurements
```

Generation must not import or depend on evaluation code. Evaluation must not import generation strategies, prompts, repair logic, or manifests, and it must be able to assess external pipelines. Pipeline repair belongs to generation-time executable verification; evaluation happens after the candidate artifact is frozen and never edits it.

The Stage A runner oversees generation through evaluation and associates every
immutable run identifier. The implemented self-improvement controller consumes
complete Stage A objective evidence, proposes one bounded source patch, creates
a new source snapshot, and reevaluates it externally. Deterministic state
transitions and Pareto decisions remain in the runner rather than the agent.

The current regular-grid-to-consolidated-Zarr-v3 Stage A gate is passed by fresh
matched Naive, Expert Direct, and TerraIO Referenced candidates. All three
complete `F(p)` without manual candidate edits under one backend/input lineage;
see `thesis/stage-a-exit-criteria.md` for exact evidence and limitations.

A preliminary six-dataset transfer matrix now covers regular-grid/Zarr and
station-time-series/Parquet outputs. Expert Direct and architecture-only TerraIO
Referenced were feasible on 6/6 datasets; Naive was feasible on 4/6. These are
single-repetition development results, not thesis-mode evidence; see
`project/benchmarks/strategy_matrix_20260821_v1/REPORT.md`.

Self-improvement candidates live under
`runs/self_improvement/{run_id}/nodes/{node_id}/candidate/`; evaluated parents
are never mutated and exploratory candidates never enter the general
`pipelines/` directory. `greedy_search.py` and `adaptive_search.py` implement
separate policies over the same proposer, frozen-suite evaluator adapter, and
noise-aware Pareto archive. Adaptive search starts greedy and forks diverse
historical candidates only after the configured number of valid non-improving
children.

Development search uses one complete objective-v3/MERODA evaluation plus two
additional deterministic repetitions, taking the median of operational
objectives without repeating the qualitative judge. The Pareto archive is a
set of promotion candidates, not an automatic promotion decision; final
promotion still requires separately frozen held-out/thesis evidence.
The frozen `proposal_history_mode` setting supports matched `full` versus
`none` trajectory-memory ablations without changing the evaluator or policy.

## Evaluation Benchmarks

The reusable benchmark harness lives under `backend/evaluation/`; dataset-specific mappings live under each dataset's `benchmarks/` folder, and immutable results are written under `runs/evaluations/`.

Evaluation is layered so the framework can scale without pretending that one
test suite understands every dataset:

```text
typed evaluation profile
  -> trusted tagged check catalog and deterministic resolver
  -> deterministic adapters (regular grid, Zarr v3, tensor workload, ...)
  -> declarative dataset mapping, independent oracle, and parameters
  -> deterministic suite validation and frozen benchmark version
  -> candidate evaluation and machine-readable report
  -> future high-level meta agent
```

The implemented `EvaluationPlanningAgent` translates a trusted inventory and
frozen contract into a typed profile, logical mapping requirements, independent
oracle requirements, and optional quarantined check-gap proposals. Mandatory
checks are resolved deterministically; the agent cannot omit them or tailor a
suite after seeing candidate implementations or scores. Dataset suite builders
still bind accepted physical outputs and evaluator-owned oracle artifacts. See
`backend/evaluation/CHECK_LIBRARY.md`.

See `backend/evaluation/README.md` for the component boundary, current
implementation, result contract, Linux requirements, and evaluation-focused
handoff plan.

Install the optional benchmark dependencies:

```bash
python3 -m pip install -r backend/requirements-benchmark.txt
```

Run a frozen constrained v3 suite:

```bash
python3 backend/scripts/run_constrained_evaluation.py \
  --config project/datasets/{dataset_slug}/benchmarks/{constrained_suite}.json
```

Run generation through a complete F(p) in one command:

```bash
python3 backend/scripts/run_stage_a_workflow.py \
  --config project/datasets/{dataset_slug}/benchmarks/experiments/{experiment}/{condition}.json
```

The core objective vector is consumer samples/second, isolated materialization
seconds, native output bytes, and the secondary `q_engineering` score. The four
hard constraints are eligibility gates. Executable infeasible candidates still
receive explicit diagnostic measurements and MERODA feedback, but no official
`F(p)`. Generated pipelines continue
to produce normal scientific artifacts and do not depend on PyTorch. Patch/chunk
matrices remain optional ablations rather than additional core objectives.

Cold-cache preparation uses `POSIX_FADV_DONTNEED` where supported. Results always record the effective cache mode; unsupported hosts report `uncontrolled` rather than claiming a cold-cache measurement.

## Design Rules

- Keep agents as control-plane components.
- Keep deterministic code responsible for execution.
- Store raw data before transformation.
- Preserve provenance and source metadata.
- Store credential names, never secret values.
- Keep generated code inspectable and runnable without an LLM.
- Prefer small first contracts over broad dataset inventories.
- Keep validation and reporting as explicit workflow stages.
