# ERA5 pressure-level family adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and regular-grid Zarr v3 publication policy.

Key behavior:

- Reads and validates `cache_dir/source_fixture_manifest.json` before any provider or credential activity.
- Requires a complete local fixture; this implementation intentionally performs no network acquisition.
- Validates contract dataset identity, fields, pressure-level selectors, date/time scope, geography, advanced options, and inventory options.
- Decodes source data with xarray mask/scale semantics, filters by the runtime lock, and publishes only requested field-selector channels.
- Writes a genuine consolidated Zarr v3 store under `dataset.zarr` with actual time, latitude, longitude, and per-channel pressure-level selector dimensions.
- Uses throughput-focused uncompressed, sample-aligned data chunks: one time sample, one pressure-level selector plane, and the full selected spatial extent.

The source fixture may contain NetCDF/Zarr files readable by xarray for tests and local execution. GRIB files are opened through cfgrib when available. No absolute source paths, credentials, or tokens are returned in artifacts or cache evidence.
