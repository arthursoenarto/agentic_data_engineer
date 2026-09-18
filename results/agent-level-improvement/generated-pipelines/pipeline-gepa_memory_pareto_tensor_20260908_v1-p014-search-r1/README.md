# ERA5 pressure-level fixture adapter

This standalone adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen `reanalysis_era5_pressure_levels` inventory and the fixed regular-grid Zarr v3 publication policy.

Runtime behavior is deterministic and offline-first. If `cache_dir/source_fixture_manifest.json` is present, every listed raw file is size- and SHA-256-verified before any decoding. A complete fixture is mandatory; this adapter intentionally does not probe or acquire data from the network.

The writer publishes one grouped Zarr data array per requested ERA5 native source variable. Pressure-level selector dimensions remain real Zarr dimensions with per-variable coordinates such as `pressure_level__temperature`. Requested artifact channels map to planes in those grouped arrays via `selector_coordinate_paths`. Data chunks are `(time=1, pressure_level=1, latitude=full selection, longitude=full selection)` and use explicit lossless Blosc-Zstd bitshuffle at compression level 9 for Zarr v3.

The adapter validates dataset identity, fields, pressure-level selectors, scope dates/times, geography, advanced options, optional fixed policy metadata, fixture integrity, grouped layout, consolidated Zarr v3 metadata, compression metadata, and channel read-back equivalence.

No framework-owned files, manifests, receipts, caches, credentials, or generated outputs are included in this source package.
