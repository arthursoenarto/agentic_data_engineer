# Evaluation Report

## Decision

- Candidate: `pipeline-gepa_memory_pareto_tensor_20260908_v1-p000-baseline-r2`
- Run: `p000-baseline-r2`
- Executable under evaluator control: `true`
- Hard-gate feasible: `false`
- Official objective vector eligible: `false`
- Engineering quality assessed: `false`
- Optimization ready: `false`
- Thesis evidence ready: `false`

## Hard Constraints

| Constraint | Status |
|---|---|
| `contract_correctness` | **fail** |
| `semantic_equivalence` | **not_assessed** |
| `rerun_safety` | **fail** |
| `provenance_security` | **not_assessed** |

## Diagnostic Measurements

These observations are diagnostic. They are not optimizer-facing unless all hard gates pass.

| Measurement | Direction | Observed |
|---|---|---:|
| `materialization_seconds` | minimize | N/A |
| `consumer_samples_per_second` | maximize | N/A |
| `output_bytes` | minimize | N/A |

Artifact diagnostics: mapped Zarr readable=`false`, directory bytes=`182310700`, files=`356`.

## Engineering Quality

Diagnostic `q_engineering`: **N/A / 4**

| MERODA | Applicability | Median | Range | Improvement |
|---|---|---:|---:|---|
| M/E/R/O/D/A | not assessed | N/A | N/A | Engineering-quality assessment is disabled by the frozen suite. |

## Official Objective Vector

Not eligible. Repair failed hard gates before comparing or optimizing `F(p)`.

## Executions

| Scenario | Exit | Time (s) | Output | Receipt |
|---|---:|---:|---|---|
| initial | 0 | 2.852814 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p000-baseline-r2/materializations/initial/initial_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p000-baseline-r2/materializations/initial/initial_writable/pipeline_run.json` |
| rerun | 0 | 2.772197 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p000-baseline-r2/materializations/rerun/rerun_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p000-baseline-r2/materializations/rerun/rerun_writable/pipeline_run.json` |

## Repair Signals

| Constraint | Code | Status | What failed |
|---|---|---|---|
| `contract_correctness` | `ZARR_INTEGRITY_FAILED` | fail | Candidate output is outside the supported class or failed Zarr validation. |
| `contract_correctness` | `ZARR_OUTPUT_POLICY_BLOCKED` | not_assessed | The Zarr output policy requires a valid mapped candidate store. |
| `semantic_equivalence` | `SEMANTIC_BLOCKED_BY_STRUCTURE` | not_assessed | Semantic comparison requires valid candidate and reference grids. |
| `rerun_safety` | `RERUN_OUTPUT_INVALID` | fail | Both outputs must be valid Zarr grids before fingerprint comparison. |
| `provenance_security` | `PROVENANCE_IDENTITIES_MISSING` | not_assessed | Required provenance identities could not all be established. |

## Next Iteration

1. Preserve executability and address the failed hard-gate feedback codes above.
2. Do not optimize diagnostic measurements yet.

## Provenance

- Suite: `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/suites/pipeline-gepa_memory_pareto_tensor_20260908_v1-p000-baseline-r2.json`
- Machine-readable result: `evaluation.json`
- Evaluator source bundle: `41aac1d65c3740f89714d2bfc56e1a6bf2e93c8940dfddc68d0005164676dc74`
