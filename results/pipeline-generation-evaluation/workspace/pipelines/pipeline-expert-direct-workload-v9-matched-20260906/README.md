# ERA5 pressure-level workload-aware adapter

Standalone adapter for `reanalysis-era5-pressure-levels` locks under the fixed regular-grid Zarr v3 publication policy.

Runtime entry point:

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter verifies `cache_dir/source_fixture_manifest.json` before any provider or credential activity. A complete fixture is therefore sufficient for read-only, credential-free execution. It decodes source packing via xarray/cfgrib or NetCDF decoding, validates lock fields/selectors/scope against the supplied inventory, preserves selected coordinates and singleton selector dimensions, then publishes one consolidated Zarr v3 root dataset at `dataset.zarr`.

Each exact field-selector combination becomes one public data array. Shared sample and spatial coordinates are `time`, `latitude`, and `longitude`; selector coordinates are channel-specific length-one dimensions to avoid conflicts between channels that select different pressure levels.
