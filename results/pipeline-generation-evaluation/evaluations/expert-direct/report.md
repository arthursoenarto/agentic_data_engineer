# Evaluation Report

## Decision

- Candidate: `pipeline-expert-direct-workload-v9-matched-20260906`
- Run: `expert-direct`
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
| `materialization_seconds` | minimize | 2.58956 |
| `consumer_samples_per_second` | maximize | 88.4189 |
| `output_bytes` | minimize | 175786324 |

Artifact diagnostics: mapped Zarr readable=`true`, directory bytes=`175786324`, files=`272`.

Native Zarr storage diagnostics: chunk bytes=`175754688`, metadata bytes=`31636`, objects=`272`, median local open latency seconds=`0.00193546`.

## Engineering Quality

Diagnostic `q_engineering`: **2.6 / 4**

| MERODA | Applicability | Median | Range | Improvement |
|---|---|---:|---:|---|
| M | applicable | 3 | 3-3 | Separate the ERA5 pressure-level policy constants and field/selector rules into a small declarative policy layer so that the core publication flow can be analyzed and modified independently. |
| E | applicable | 1 | 1-1 | Add and pass an evaluator-recognized alternate-contract probe, and move dataset IDs, variables, selector dimensions, units, and acquisition format constraints into validated configuration rather than fixed code paths. |
| R | applicable | 4 | 4-4 | Add explicit regression tests for failure during final-store replacement and for partial temporary-store cleanup to further harden crash-recovery guarantees. |
| O | applicable | 3 | 3-3 | Emit structured phase logs for fixture verification, source opening, channel extraction, Zarr write, validation, and final publication, including record counts and array shapes. |
| D | applicable | 2 | 2-2 | Expand the README with installation steps, the exact run_pipeline.py command pattern, expected output and receipt files, how to run tests, and known dataset/policy limitations. |
| A | not_applicable | N/A | N/A | Not applicable under the general_pipeline profile. |

## Official Objective Vector

`F(p) = (materialization_seconds=2.58956, consumer_samples_per_second=88.4189, output_bytes=175786324, q_engineering=2.6)`

## Executions

| Scenario | Exit | Time (s) | Output | Receipt |
|---|---:|---:|---|---|
| initial | 0 | 2.589557 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-direct/materializations/initial/initial_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-direct/materializations/initial/initial_writable/pipeline_run.json` |
| rerun | 0 | 2.459188 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-direct/materializations/rerun/rerun_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/evaluations/expert-direct/materializations/rerun/rerun_writable/pipeline_run.json` |

## Repair Signals

| Constraint | Code | Status | What failed |
|---|---|---|---|
| all | `NONE` | pass | All hard-gate checks passed. |

## Next Iteration

1. Candidate is eligible for Pareto or lexicographic comparison using the official vector.

## Provenance

- Suite: `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/workload_aware_expert_prompt_pilot_20260906_v1/followups/expert_family_v9_matched/suites/pipeline-expert-direct-workload-v9-matched-20260906.json`
- Machine-readable result: `evaluation.json`
- Evaluator source bundle: `8857aba7526d6072b2905240e0c6d616ff7447b22c55694cbda06a765329a272`
