# ERA5 pressure-level fixture adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and regular-grid Zarr publication policy.

Key behavior:

- Reads `cache_dir/source_fixture_manifest.json` first, verifies every listed raw file by size and SHA-256, and runs fully offline without credentials.
- Rejects missing fixtures instead of probing the network.
- Validates the runtime contract against the frozen inventory, including dataset slug, acquisition options, date range, selected UTC times, area order, variables, pressure-level selectors, and field-selector combinations.
- Decodes local GRIB through `cfgrib` or local NetCDF/Zarr fixtures through xarray with CF mask/scale enabled.
- Publishes genuine consolidated Zarr v3 under `output_dir/dataset.zarr`.
- Preserves `time`, `pressure_level`, `latitude`, and `longitude` as real coordinates/dimensions; pressure level is not scalarized for single-level selections.
- Uses throughput-oriented full-field chunks: `(time=1, pressure_level=1, latitude=full, longitude=full)` for data variables.
- Publishes data chunks without a data compressor to reduce full-field tensor read overhead while preserving decoded values exactly.

The implementation writes only beneath the caller-provided output directory and returns the framework-declared artifact layout. It never invokes an LLM and never emits secrets in cache evidence.
