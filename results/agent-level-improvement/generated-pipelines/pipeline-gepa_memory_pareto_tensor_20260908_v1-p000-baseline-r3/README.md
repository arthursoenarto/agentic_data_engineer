# ERA5 pressure-level offline adapter

This directory contains a standalone dataset-family adapter for the frozen `reanalysis_era5_pressure_levels` inventory and the fixed policy that publishes regular-grid data as consolidated Zarr v3.

Runtime entry point:

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter deliberately performs no network access. `cache_dir` must contain `source_fixture_manifest.json` plus the raw fixture entries listed by that manifest. Every entry is resolved beneath `cache_dir`, size-checked, SHA-256 verified, and opened before any provider-related logic. Complete fixtures therefore run read-only and credential-free.

Supported fixture inputs are CF NetCDF, Zarr, and GRIB files readable by `cfgrib`. CF decoding is enabled with scale/offset and fill-value masking. Source encodings are cleared before public Zarr publication to avoid silent repacking.

The runtime contract is selected from either a direct `dataset_contract.v1` document or a lock envelope containing one. Validation covers dataset slug, inventory identity, ERA5 options, pressure-level selectors, UTC selected times, date bounds, product type, area bounds, and fixed GRIB acquisition policy.

Publication writes `output_dir/dataset.zarr` through a temporary store, reopens it, validates requested dimensions and coordinates, and returns a framework-compatible artifact layout. The pressure-level selector is preserved as a real Zarr dimension, including single-level selections.
