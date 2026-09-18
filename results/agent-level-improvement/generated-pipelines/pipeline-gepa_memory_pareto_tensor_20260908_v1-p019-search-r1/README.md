# ERA5 pressure-level local-fixture adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and fixed regular-grid Zarr policy.

Key behavior:

- Requires `cache_dir/source_fixture_manifest.json`; no provider construction, credential lookup, or network fallback is performed.
- Verifies each fixture entry by path containment, byte size, and SHA-256 before opening data.
- Validates the runtime contract against the supplied inventory, including variables, pressure-level selectors, UTC selected times, inclusive date-only bounds, geography, advanced options, and product type.
- Decodes CF/GRIB data through xarray/cfgrib or xarray NetCDF readers with mask/scale enabled.
- Publishes genuine consolidated Zarr v3 at `dataset.zarr` under `output_dir`.
- Uses one logical data array per requested field-selector channel.  Each pressure-level channel keeps a real selector dimension of cardinality one.
- Uses sample-access-aligned chunks: `(time=1, selector=1, latitude=full_selected_latitude_count, longitude=full_selected_longitude_count)`.
- Attempts explicit uncompressed Zarr v3 data chunks for throughput; if the host stack rejects uncompressed chunks, it falls back only to lossless read-speed-biased Blosc-LZ4 level 1 with shuffle.

The return value follows the framework `dataset_artifact_layout.v1` structure and includes secret-safe fixture cache evidence only.
