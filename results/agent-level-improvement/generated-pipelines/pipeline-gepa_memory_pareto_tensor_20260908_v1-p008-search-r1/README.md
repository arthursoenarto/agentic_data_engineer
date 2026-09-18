# ERA5 pressure-level regular-grid adapter

Pipeline ID: `pipeline-gepa_memory_pareto_tensor_20260908_v1-p008-search-r1`

This is a standalone family adapter for the frozen ERA5 pressure-level inventory.  It exposes only the framework callable:

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The implementation is offline-first and deterministic:

- requires `cache_dir/source_fixture_manifest.json`;
- verifies every listed source file by size and SHA-256 before reading it;
- performs no network access and no credential checks;
- validates runtime contract fields, pressure-level selectors, dates, times, product type, area, and advanced options against the supplied inventory;
- decodes CF packing/missing conventions through xarray before publication;
- publishes a genuine consolidated Zarr v3 store at `dataset.zarr`;
- keeps one public data array per requested field-selector channel;
- retains each selected pressure level as a real singleton dimension with a channel-specific coordinate path;
- uses full-field tensor chunks: `(time=1, selector=1, latitude=full, longitude=full)`;
- uses explicit lossless Zarr v3 codec metadata, preferring Blosc/LZ4 `clevel=1` with bitshuffle.

The returned artifact layout follows the framework regular-grid policy, with sample/y/x mapped to `time`, `latitude`, and `longitude`.  Selector dimensions are declared per channel through `selector_coordinate_paths`.

## Local fixture format

`cache_dir/source_fixture_manifest.json` must use schema `source_fixture_manifest.v1` and list raw GRIB or NetCDF files relative to `cache_dir`.  Tests use NetCDF fixtures; production pilot fixtures may be GRIB and are read via `cfgrib`.

## Running tests

```bash
python -m pip install -r requirements.txt
pytest tests
```
