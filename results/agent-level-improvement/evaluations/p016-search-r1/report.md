# Evaluation Report

## Decision

- Candidate: `pipeline-gepa_memory_pareto_tensor_20260908_v1-p016-search-r1`
- Run: `p016-search-r1`
- Executable under evaluator control: `true`
- Hard-gate feasible: `true`
- Official objective vector eligible: `false`
- Engineering quality assessed: `false`
- Optimization ready: `false`
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
| `materialization_seconds` | minimize | 4.59539 |
| `consumer_samples_per_second` | maximize | 178.871 |
| `output_bytes` | minimize | 348891177 |

Artifact diagnostics: mapped Zarr readable=`true`, directory bytes=`348891177`, files=`101`.

Native Zarr storage diagnostics: chunk bytes=`348866400`, metadata bytes=`24777`, objects=`101`, median local open latency seconds=`0.00128813`.

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
| initial | 0 | 4.595392 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p016-search-r1/materializations/initial/initial_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p016-search-r1/materializations/initial/initial_writable/pipeline_run.json` |
| rerun | 0 | 4.273064 | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p016-search-r1/materializations/rerun/rerun_writable/output` | `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/evaluations/p016-search-r1/materializations/rerun/rerun_writable/pipeline_run.json` |

## Repair Signals

| Constraint | Code | Status | What failed |
|---|---|---|---|
| all | `NONE` | pass | All hard-gate checks passed. |

## Next Iteration

1. Complete the MERODA assessment before optimizer comparison.

## Provenance

- Suite: `project/datasets/reanalysis_era5_pressure_levels/benchmarks/experiments/gepa_memory_pareto_tensor_20260908_v1/suites/pipeline-gepa_memory_pareto_tensor_20260908_v1-p016-search-r1.json`
- Machine-readable result: `evaluation.json`
- Evaluator source bundle: `41aac1d65c3740f89714d2bfc56e1a6bf2e93c8940dfddc68d0005164676dc74`
