# ERA5 pressure-level fixture adapter

Standalone implementation for `reanalysis_era5_pressure_levels` under the fixed regular-grid Zarr policy.

## Runtime entrypoint

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter is intentionally offline-first. It requires `cache_dir/source_fixture_manifest.json`, verifies every listed raw file by containment, byte size, and SHA-256, and never probes CDS or any other network service. Fixture files are opened as decoded xarray datasets; CF mask/scale conventions are applied before publication and source packing encodings are not propagated to the public Zarr.

## Publication

Outputs are atomically published as consolidated Zarr v3 at `dataset.zarr` beneath `output_dir`. Requested ERA5 pressure-level channels are grouped by native source variable where this remains unambiguous. Each requested channel is still declared separately in the returned `dataset_artifact.channels` with selector coordinates identifying the selected pressure-level plane.

Data chunks are deterministic full-field tensor chunks: one time sample, one selected pressure-level plane, and the full selected latitude/longitude extents. Data variables are written with an explicit lossless footprint-oriented Zstd/Blosc-Zstd codec, targeting `cname='zstd'`, `clevel=9`, and bitshuffle. The adapter reopens the output and validates dimensions, coordinates, exact values, chunks, consolidated metadata, channel declarations, and codec metadata before returning.
