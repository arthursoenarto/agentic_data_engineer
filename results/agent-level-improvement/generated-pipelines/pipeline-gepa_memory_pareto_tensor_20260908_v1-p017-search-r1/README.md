# ERA5 pressure-level regular-grid adapter

This standalone adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and regular-grid Zarr policy.

Key properties:

- Reads the selected `dataset_contract.v1` from a full lock envelope and validates variables, pressure-level selectors, times, area, product type, and advanced options against the supplied inventory.
- Uses only evaluator-owned local fixtures. If `cache_dir/source_fixture_manifest.json` is present, every listed file is size/SHA-256 verified before decoding. Network fallback is intentionally not implemented.
- Decodes CF packing/missing values through xarray before publication.
- Publishes consolidated Zarr v3 under `era5_pressure_levels.zarr`.
- Uses one data array per requested field-selector channel. Singleton pressure-level selectors remain real dimensions using variable-specific selector coordinate arrays.
- Uses sample-access-aligned chunks for throughput: one time sample, one selector plane, and the full selected latitude/longitude extent.
- Uses explicit uncompressed float32 data chunks for this throughput-only probe, while refusing to publish if a decoded floating source would be changed by float32 conversion.
- Reopens and validates the store before returning the dataset artifact layout.

The implementation is deterministic and never writes into `cache_dir`; all publication occurs atomically under `output_dir`.
