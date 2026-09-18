# ERA5 pressure-level fixture adapter

This adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the ERA5 pressure-level regular-grid family.

Key properties:

- offline-only execution from `cache_dir/source_fixture_manifest.json`;
- manifest path-containment, size, and SHA-256 verification before source decoding;
- no credential checks, provider construction, or network fallback;
- decoded CF/xarray source publication to consolidated Zarr v3;
- one Zarr group per requested native variable, allowing `temperature` to retain pressure levels `500,850` while `geopotential` retains only `500` as a real `pressure_level` dimension;
- sample-access-aligned chunks `(time=1, pressure_level=1, full_latitude, full_longitude)`;
- explicit uncompressed data chunks where supported, with a deterministic lossless Blosc-LZ4 clevel=1 byte-shuffle fallback only if the host Zarr v3 stack rejects uncompressed publication.

The returned `dataset_artifact.channels` declares every requested field-selector channel separately.  Channels sharing a grouped array identify the selected plane through `selectors` and `selector_coordinate_paths`.

The adapter supports other reanalysis locks within the frozen inventory for different dates, times, variables, pressure levels, and rectangular areas, provided the verified local fixture contains the requested decoded source data.  Ensemble product types are intentionally rejected because the fixed artifact policy has no ensemble/member dimension.
