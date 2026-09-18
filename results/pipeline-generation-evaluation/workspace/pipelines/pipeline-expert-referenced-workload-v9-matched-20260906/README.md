# ERA5 pressure-level family adapter

Standalone deterministic adapter for `reanalysis_era5_pressure_levels` under the fixed regular-grid Zarr v3 publication policy.

## Entry point

```python
import pipeline_impl
result = pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter reads the selected `dataset_contract.v1` from a lock envelope, validates it against the frozen inventory, verifies `cache_dir/source_fixture_manifest.json`, decodes the listed local source files, filters the exact requested variables, pressure levels, times, and area, and publishes one consolidated Zarr v3 root dataset.

## Fixture-first execution

`cache_dir` is treated as an external read-only-capable source/cache root. A complete fixture requires no credentials and no network access. Every manifest entry is checked for path containment, byte size, and SHA-256 before any source decoding.

## Output layout

The public store is `era5_pressure_levels.zarr` under `output_dir`. It contains shared `time`, `latitude`, and `longitude` coordinates and one data array per exact field-selector channel. Each channel keeps a length-one `pressure_level` selector dimension; per-channel selector coordinate paths are reported in the artifact mapping.

Data variable dimension order is optimized for shuffled per-sample full-field reads:

`time`, `pressure_level`, `latitude`, `longitude`

Time and singleton selector chunks are size 1. Longitude chunks are full-width and latitude chunks target approximately 1-2 MiB uncompressed for float32.
