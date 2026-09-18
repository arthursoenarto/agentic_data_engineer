# ERA5 pressure-level family adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and the fixed regular-grid Zarr publication policy.

Key properties:

- consumes `cache_dir/source_fixture_manifest.json` before any acquisition logic;
- verifies fixture file size and SHA-256, then runs entirely credential-free and offline;
- validates runtime contract fields, pressure-level selectors, times, date range, area, product type, and advanced options against the supplied inventory;
- decodes CF NetCDF packing/missing conventions through xarray and clears source packing encodings before publication;
- supports GRIB fixtures through `cfgrib`/ecCodes and NetCDF fixtures for offline tests;
- publishes genuine consolidated Zarr v3;
- preserves time, latitude, longitude, selected pressure-level coordinates, missingness, variable attributes, and units where available;
- avoids unrequested pressure planes by using per-variable pressure selector dimensions when requested level sets differ;
- writes data chunks aligned to full-field tensor reads: one time sample, one pressure selector plane, and the full selected spatial extent;
- uses deterministic lossless Blosc/Zstandard compression with bitshuffle and `clevel=3` for a balanced throughput/footprint point.

The function performs deterministic atomic output publication to `era5_pressure_levels.zarr` under the framework-provided output directory and returns a `dataset_artifact_layout.v1` regular-grid artifact declaration.
