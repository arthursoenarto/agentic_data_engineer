# ERA5 pressure-level family adapter

This directory contains a standalone deterministic adapter for `reanalysis_era5_pressure_levels` under the fixed regular-grid Zarr publication policy.

## Entry point

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The adapter reads the selected `dataset_contract.v1` from either a direct contract object or a lock envelope, validates it against the supplied frozen `dataset_inventory.v1`, verifies all source fixture files listed in `cache_dir/source_fixture_manifest.json`, filters the decoded source data, and publishes `output_dir/dataset.zarr`.

## Fixture-first execution

`cache_dir` is external framework-owned storage. If `source_fixture_manifest.json` is present, every listed file is resolved under `cache_dir`, size-checked, SHA-256-checked, and only then decoded. This implementation intentionally performs no network acquisition, no credential lookup, and no writes into `cache_dir`; a missing or incomplete fixture is a deterministic error.

## Publication

Regular grids are published as consolidated Zarr v3. Source CF packing and missing-value conventions are decoded by xarray (`decode_cf=True`, `mask_and_scale=True`), then public variable encodings are cleared to prevent accidental repacking. Data variables are transposed to canonical `time, pressure_level, latitude, longitude` order when those dimensions are present. For full-field tensor loading, data-variable chunks are explicitly aligned to one time sample, one pressure-level selector plane, and the full selected spatial extent.

## Returned artifact layout

The return value follows the framework layout:

- `cache`: secret-safe fixture evidence only.
- `dataset_artifact.storage_format`: `zarr`.
- `dataset_artifact.store_path`: `dataset.zarr` relative to `output_dir`.
- `dataset_artifact.channels`: one entry for each requested contract field-selector using the canonical field-id rule.

## Scope supported

The adapter supports valid locks inside the supplied ERA5 pressure-level inventory: inventory variables, pressure levels, UTC selected times, date ranges, product type options, and global or rectangular CDS areas. Inclusive date-only end bounds are treated as the complete final UTC calendar day before exact selected times are applied.
