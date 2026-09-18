# ERA5 pressure-level offline adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen `reanalysis_era5_pressure_levels` inventory and the fixed regular-grid Zarr policy.

Key properties:

- Requires a complete `cache_dir/source_fixture_manifest.json`; there is no CDS/network fallback.
- Verifies every listed raw source file by path containment, byte size, and SHA-256 before decoding.
- Validates runtime contract fields, pressure-level selectors, time/date bounds, geography, product type, and advanced options against the supplied inventory.
- Decodes CF packing/missing conventions through xarray before publication and clears source encodings to prevent silent scale/offset repacking.
- Publishes genuine consolidated Zarr v3 under `output_dir/dataset.zarr`.
- Uses sample-access-aligned chunks for tensor loading: one time, one selected pressure-level plane, and full selected latitude/longitude extents.
- Groups channels sharing the same native variable into one data array with a real selected selector dimension. For the seed request this yields `temperature(time, temperature_pressure_level, latitude, longitude)` for 500/850 hPa and `geopotential(time, geopotential_pressure_level, latitude, longitude)` for 500 hPa. Selector coordinate paths in the returned artifact disambiguate each channel.
- Tries explicit uncompressed Zarr v3 chunks first. If the installed xarray/zarr stack cannot write uncompressed v3 chunks deterministically, it falls back only to lossless Blosc-LZ4 clevel 1 with byte shuffle.

The adapter writes no framework-owned files and does not write into the fixture cache.
