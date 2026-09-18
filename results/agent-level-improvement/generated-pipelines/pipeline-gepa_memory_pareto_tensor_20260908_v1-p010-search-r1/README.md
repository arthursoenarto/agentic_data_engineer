# ERA5 pressure-level regular-grid adapter

Standalone implementation for `pipeline-gepa_memory_pareto_tensor_20260908_v1-p010-search-r1`.

The runtime entry point is exactly:

`pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)`

It validates a selected `dataset_contract.v1` against the frozen ERA5 pressure-level inventory and the fixed Zarr v3 publication policy, verifies `cache_dir/source_fixture_manifest.json`, decodes the listed local source files before any possible acquisition path, filters by the lock-derived date/time, geography, field, and pressure-level selectors, and writes a consolidated Zarr v3 store at `data.zarr` under the supplied output directory.

Network acquisition is intentionally not implemented. A complete verified local fixture is required. NetCDF, Zarr, and GRIB fixture files are supported; tests use NetCDF so they run offline without CDS credentials.

Footprint-specific storage policy: data variables are written with deterministic sample/selector-aligned chunks and lossless Blosc Zstd bitshuffle at compression level 9. Coordinates remain Zarr arrays, and selected pressure levels are preserved as real one-element per-channel selector dimensions rather than scalarized attributes.
