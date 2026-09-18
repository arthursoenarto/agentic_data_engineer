# ERA5 pressure-level fixture adapter

This standalone adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and regular-grid Zarr policy.

Runtime behavior is offline-only for this pilot family. The adapter requires `cache_dir/source_fixture_manifest.json`, verifies every listed raw file by path containment, byte size, and SHA-256, and never falls back to network or credentials.

The selected contract is read from the full lock envelope. The implementation validates dataset identity, product type, variables, pressure-level selectors, selected UTC times, inclusive date-only bounds, geography, advanced options, and inventory option membership. It decodes CF packing/missing-value conventions through xarray before publication.

Output is a consolidated Zarr v3 store named `era5_pressure_levels.zarr` under the supplied output directory. Requested field-selector channels are published as separate data arrays. Singleton pressure-level selector dimensions remain real per-channel dimensions with selector coordinate paths declared in the returned artifact. Data chunks are aligned for full-field tensor reads: one time sample, one pressure-level selector plane, and the full selected latitude-longitude extent. The adapter first attempts explicit uncompressed Zarr v3 chunks for the throughput trial; if the host API does not support that deterministically, it falls back to lossless Blosc-LZ4 with clevel 1 and byte shuffle.

The adapter does not create framework-owned files, receipts, manifests, or cache artifacts.
