# ERA5 workload-aware expert prompt pilot

## Goal

Prepare an ML-ready ERA5 pressure-levels pipeline contract for repeated ML training reads with correctness mandatory and optimization priorities set by the user.

## Selected Data

ERA5 pressure-level reanalysis for 2024-01-01 through 2024-01-07 inclusive, global area, UTC times 00:00/06:00/12:00/18:00, with temperature at 500 hPa, temperature at 850 hPa, and geopotential at 500 hPa.

## Downstream Use

The data will be read repeatedly during ML training.

## Optimisation Priorities

1. Maximize `consumer_samples_per_second`.
2. Minimize `output_bytes`.

## Descriptive Measurements

- `materialization_seconds`
- `q_engineering`

## Decisions

- Use the selected ERA5 pressure-levels dataset and scope exactly as supplied.
- Required output target is Zarr version 3.
- Downstream use is repeated ML training reads.
- Use workload_kind full_field_tensor from the supported regular_rectilinear_grid capability.
- Correctness is mandatory.
- Optimization priority 1 is to maximize consumer_samples_per_second.
- Optimization priority 2 is to minimize output_bytes.
- materialization_seconds and q_engineering are descriptive measurements only and must not override the ordered objectives.

## Assumptions

- Credentials are supplied externally through environment-variable references only.
- No aggregation, unit conversion, feature engineering, or physical storage layout is requested.
