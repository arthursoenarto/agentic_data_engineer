# ERA5 pressure-level family adapter

This directory contains a standalone implementation of `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and fixed regular-grid Zarr policy.

## Runtime behavior

- Reads the selected `dataset_contract.v1` from a full lock envelope.
- Validates slug, dataset id, runtime fields, pressure-level selectors, product/data/download options, UTC time selectors, date range, and geography against the supplied inventory.
- Treats `cache_dir` as an external source/cache root. If `cache_dir/source_fixture_manifest.json` exists, every listed file is resolved under `cache_dir`, size checked, SHA-256 checked, and consumed before any provider/acquisition path. This adapter intentionally requires a complete local fixture for deterministic pilot execution.
- Opens CF NetCDF fixtures with decoded scale/offset and fill-value masking. GRIB fixtures are opened through `cfgrib` when present.
- Filters only from the runtime lock. Inclusive date-only end dates cover the full final UTC calendar day before applying the selected HH:MM times.
- Publishes consolidated Zarr v3 to `output_dir/dataset.zarr` using temporary-store publication and validates the reopened store.
- Clears source encodings before publication to avoid silent public Zarr repacking.

## Output

The return value follows the framework interface:

- `cache`: secret-safe fixture evidence.
- `dataset_artifact`: regular-grid Zarr layout with `sample=time`, `y=latitude`, and `x=longitude`.
- `channels`: one channel per requested field-selector combination using canonical field IDs such as `temperature[pressure_level="500"]`.

Pressure-level selector coordinates are preserved as real one-element dimensions on each channel array, not scalar coordinates.

## Fixture manifest

Example:

```json
{
  "schema_version": "source_fixture_manifest.v1",
  "entries": [
    {
      "entry_id": "era5-fixture-netcdf",
      "relative_path": "raw/era5_fixture.nc",
      "source": "local test fixture",
      "size_bytes": 12345,
      "sha256": "...64 lowercase hex..."
    }
  ]
}
```

No secrets are required or read for complete fixture execution.
