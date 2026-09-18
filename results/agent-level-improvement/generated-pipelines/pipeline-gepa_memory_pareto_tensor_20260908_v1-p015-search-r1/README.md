# ERA5 pressure-level regular-grid fixture adapter

Standalone adapter for the frozen `reanalysis_era5_pressure_levels` inventory and regular-grid Zarr v3 output policy.

Runtime entry point:

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter is deterministic and fixture-first. If `cache_dir/source_fixture_manifest.json` exists, all listed raw files are verified by size and SHA-256 before any decoding. Complete local fixtures run without credentials and without network access. This implementation does not acquire remote data; if no valid fixture manifest is present it fails safely.

Publication behavior:

- validates dataset identity, inventory options, fields, pressure-level selectors, scope, UTC selected times, geography, advanced options, and any fixed policy embedded in the lock envelope;
- decodes CF packing and missing values via xarray before publication;
- filters from the runtime contract only;
- preserves selected sample/y/x dimensions and one real pressure-level selector dimension per requested channel;
- writes genuine consolidated Zarr v3 under `dataset.zarr`;
- writes data arrays with explicit sample-aligned chunks `(time=1, selector=1, full latitude, full longitude)`;
- writes data chunks uncompressed to optimize warm full-field tensor reads;
- atomically replaces the public output store after reopen/readback/storage validation.

Returned artifact paths are relative and secret-safe. The adapter never emits local source paths, credentials, tokens, or framework-owned files.
