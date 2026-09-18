# ERA5 pressure-level fixture adapter

This standalone adapter materializes verified local ERA5 pressure-level fixtures as consolidated Zarr v3. It implements:

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

Key properties:

- No runtime LLM calls and no dependency on the repository harness.
- Fixture-first execution: `cache_dir/source_fixture_manifest.json` is verified before any source decoding. Missing or invalid fixtures fail safely; network acquisition is not attempted.
- Runtime contract and fixed-policy validation against the frozen ERA5 pressure-level inventory.
- Inclusive date-only end bounds are treated as the complete final UTC day before exact selected-time filtering.
- Decoded source values are published without source scale/offset repacking.
- Requested pressure-level channels are grouped by native requested variable into one Zarr data array per variable, with a real per-variable pressure-level selector dimension.
- Data chunks are deterministic full-field tensor chunks: `(time=1, selected_pressure_level=1, latitude=full, longitude=full)`.
- Data chunks use lossless Zarr v3 Blosc-Zstd compression at `clevel=9` with bitshuffle.
- Artifact channels map each requested field-selector ID exactly once to its data array and selector coordinate path.

The published store is `era5_pressure_levels.zarr` relative to the caller-provided output directory. Metadata and cache evidence contain only secret-safe fixture entry IDs, not absolute paths or credentials.
