# ERA5 pressure-level family adapter

This directory contains a standalone deterministic adapter for the frozen ERA5 pressure-level inventory and fixed regular-grid Zarr publication policy.

## Entry point

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter reads the selected `dataset_contract.v1` from a lock envelope, validates it against the supplied `dataset_inventory.v1`, verifies a local source fixture, decodes source data with xarray, filters by the runtime lock, and publishes `dataset.zarr` beneath `output_dir`.

## Source fixture behavior

If `cache_dir/source_fixture_manifest.json` exists, it is authoritative. Every manifest entry is resolved beneath `cache_dir` and verified by byte size and lowercase SHA-256 before decoding. The implementation performs no CDS client construction, credential checks, or network activity before fixture verification. This pilot has no network fallback, so a complete fixture is required.

Supported fixture inputs are CF NetCDF and GRIB files readable by xarray. NetCDF inputs are opened with `mask_and_scale=True` and `decode_times=True`; decoded values are then written to public Zarr without propagating source scale/offset packing encodings.

## Publication

The output is an atomically published consolidated Zarr v3 store:

```text
output_dir/dataset.zarr
```

The public dataset preserves explicit dimensions:

- `time`
- `pressure_level`
- `latitude`
- `longitude`

The `pressure_level` selector remains a true dimension even when only one level is selected.

## Cache discipline

`cache_dir` is treated as external framework-owned input/cache. The adapter does not write into the fixture cache. Returned cache evidence contains only secret-safe fixture entry IDs and SHA-256 prefixes.

## Running tests

```bash
python -m pip install -e '.[test]'
pytest
```
