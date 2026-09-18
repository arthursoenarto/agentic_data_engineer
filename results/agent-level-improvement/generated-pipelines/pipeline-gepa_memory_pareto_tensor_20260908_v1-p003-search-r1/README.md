# ERA5 pressure-level dataset-family adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and the fixed regular-grid Zarr publication policy.

Key properties:

- Requires a complete evaluator-owned local fixture at `cache_dir/source_fixture_manifest.json` and verifies every listed raw file by size and SHA-256 before opening data.
- Performs no network fallback and does not construct provider clients or inspect credentials.
- Reads the selected `dataset_contract.v1` from a full lock envelope and validates fields, selectors, temporal scope, geography, product type, and inventory options.
- Decodes CF NetCDF packing/missing conventions through xarray (`decode_cf=True`, `mask_and_scale=True`) and supports GRIB through cfgrib when the host has ecCodes available.
- Publishes genuine consolidated Zarr v3 under `output_dir/dataset.zarr` atomically.
- Preserves requested singleton pressure-level selector dimensions as real variable-specific dimensions.
- Uses deterministic full-field sample-aligned chunks: `(time=1, pressure_level=1, latitude=full, longitude=full)`.
- Uses explicit lossless Blosc-LZ4 level-1 shuffle compression for data arrays, prioritizing read/decode throughput for full-field tensor samples.

The returned artifact declares one channel per requested field-selector combination using the framework canonical field-id convention. The adapter is deterministic and safe for read-only fixture caches.
