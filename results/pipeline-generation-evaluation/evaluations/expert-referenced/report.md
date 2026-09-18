# Evaluation Report

## Decision

- Candidate: `pipeline-expert-referenced-workload-v9-matched-20260906`
- Run: `expert-referenced`
- Executable under evaluator control: `true`
- Hard-gate feasible: `true`
- Official objective vector eligible: `true`
- Engineering quality assessed: `true`
- Optimization ready: `true`
- Thesis evidence ready: `false`

## Hard Constraints

| Constraint | Status |
|---|---|
| `contract_correctness` | **pass** |
| `semantic_equivalence` | **pass** |
| `rerun_safety` | **pass** |
| `provenance_security` | **pass** |

## Diagnostic Measurements

These observations are diagnostic. They are not optimizer-facing unless all hard gates pass.

| Measurement | Direction | Observed |
|---|---|---:|
| `materialization_seconds` | minimize | 2.33856 |
| `consumer_samples_per_second` | maximize | 92.3206 |
| `output_bytes` | minimize | 175785701 |

Artifact diagnostics: mapped Zarr readable=`true`, directory bytes=`175785701`, files=`272`.

Native Zarr storage diagnostics: chunk bytes=`175754688`, metadata bytes=`31013`, objects=`272`, median local open latency seconds=`0.00164487`.

## Engineering Quality

Diagnostic `q_engineering`: **2.2 / 4**

| MERODA | Applicability | Median | Range | Improvement |
|---|---|---:|---:|---|
| M | applicable | 3 | 3-3 | Refactor the framework entrypoint into smaller reusable validation, execution, and receipt modules, and remove or isolate unused tabular-interface code from this Zarr-only pipeline. |
| E | applicable | 1 | 1-1 | Add an evaluator-verifiable alternate-contract test path and reduce hard-coded policy assumptions where the inventory and contract can safely drive behavior. |
| R | applicable | 3 | 3-3 | Document and validate external GRIB/eccodes runtime prerequisites at startup, and add tests for failed publication rollback and corrupt partial-output recovery. |
| O | applicable | 2 | 2-2 | Add structured step-level logs or receipt events for contract validation, fixture verification, source decoding, dataset construction, Zarr write, and validation, including per-step durations. |
| D | applicable | 2 | 2-2 | Expand the README with install/setup commands, the exact public CLI invocation pattern, expected output and receipt examples, dependency notes for GRIB/eccodes, and documented limitations. |
| A | not_applicable | N/A | N/A | No architecture-adaptation assessment is required for this profile. |

## Official Objective Vector

`F(p) = (materialization_seconds=2.33856, consumer_samples_per_second=92.3206, output_bytes=175785701, q_engineering=2.2)`

## Executions

| Scenario | Exit | Time (s) | Output | Receipt |
|---|---:|---:|---|---|
| initial | 0 | 2.338558 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-referenced/materializations/initial/initial_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-referenced/materializations/initial/initial_writable/pipeline_run.json` |
| rerun | 0 | 2.280911 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-referenced/materializations/rerun/rerun_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-referenced/materializations/rerun/rerun_writable/pipeline_run.json` |

## Repair Signals

| Constraint | Code | Status | What failed |
|---|---|---|---|
| all | `NONE` | pass | All hard-gate checks passed. |

## Next Iteration

1. Candidate is eligible for Pareto or lexicographic comparison using the official vector.

## Provenance

- Suite: `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/suites/pipeline-expert-referenced-workload-v9-matched-20260906.json`
- Machine-readable result: `evaluation.json`
- Evaluator source bundle: `8857aba7526d6072b2905240e0c6d616ff7447b22c55694cbda06a765329a272`
