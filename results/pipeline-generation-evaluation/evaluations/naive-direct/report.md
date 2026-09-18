# Evaluation Report

## Decision

- Candidate: `pipeline-naive-direct-workload-v9-matched-20260906`
- Run: `naive-direct`
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
| `materialization_seconds` | minimize | 2.0198 |
| `consumer_samples_per_second` | maximize | 50.8181 |
| `output_bytes` | minimize | 175177300 |

Artifact diagnostics: mapped Zarr readable=`true`, directory bytes=`175177823`, files=`95`.

Native Zarr storage diagnostics: chunk bytes=`175164390`, metadata bytes=`12910`, objects=`94`, median local open latency seconds=`0.00117663`.

## Engineering Quality

Diagnostic `q_engineering`: **2 / 4**

| MERODA | Applicability | Median | Range | Improvement |
|---|---|---:|---:|---|
| M | applicable | 2 | 2-2 | Move dataset-specific fields, store names, requested times, and aliases into validated contract-driven configuration and add focused unit tests for fixture validation, coordinate normalization, and artifact declaration validation. |
| E | applicable | 1 | 1-1 | Implement contract-derived field, selector, date/time, area, and output naming behavior and validate it with an evaluator-owned alternate-contract probe. |
| R | applicable | 3 | 3-3 | Declare and validate the required Python/package/system dependencies explicitly, especially xarray, zarr, cfgrib, and ecCodes, and add dependency/version information to receipts. |
| O | applicable | 3 | 3-3 | Add structured stage-level logs or receipt events for input validation, fixture expansion, dataset opening, field extraction, Zarr writing, and publication, including elapsed time and counts for each stage. |
| D | applicable | 1 | 1-1 | Add a concise README covering environment dependencies, exact run command, required cache manifest/files, expected outputs and receipt fields, rerun behavior, and known dataset-specific limitations. |
| A | not_applicable | N/A | N/A | No architecture-adaptation improvement is applicable for this profile. |

## Official Objective Vector

`F(p) = (materialization_seconds=2.0198, consumer_samples_per_second=50.8181, output_bytes=175177300, q_engineering=2)`

## Executions

| Scenario | Exit | Time (s) | Output | Receipt |
|---|---:|---:|---|---|
| initial | 0 | 2.019795 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/naive-direct/materializations/initial/initial_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/naive-direct/materializations/initial/initial_writable/pipeline_run.json` |
| rerun | 0 | 2.061456 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/naive-direct/materializations/rerun/rerun_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/naive-direct/materializations/rerun/rerun_writable/pipeline_run.json` |

## Repair Signals

| Constraint | Code | Status | What failed |
|---|---|---|---|
| all | `NONE` | pass | All hard-gate checks passed. |

## Next Iteration

1. Candidate is eligible for Pareto or lexicographic comparison using the official vector.

## Provenance

- Suite: `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/suites/pipeline-naive-direct-workload-v9-matched-20260906.json`
- Machine-readable result: `evaluation.json`
- Evaluator source bundle: `8857aba7526d6072b2905240e0c6d616ff7447b22c55694cbda06a765329a272`
