# ERA5 pressure-level family adapter

Implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and fixed regular-grid Zarr policy.

Key properties:

- Uses only evaluator-provided local source fixtures described by `cache_dir/source_fixture_manifest.json`; it performs no credential checks and no network fallback.
- Verifies fixture size and SHA-256 before opening any provider/backend.
- Validates runtime contract fields, date/time scope, geography, pressure-level selectors, product type, and fixed GRIB acquisition policy against the supplied inventory.
- Decodes CF inputs with xarray `mask_and_scale=True`; GRIB fixtures are opened through cfgrib when present.
- Publishes genuine consolidated Zarr v3 under `dataset.zarr` with one data array per requested field-selector channel. Selector dimensions are retained as real one-element dimensions rather than scalar coordinates.
- Uses deterministic full-field tensor chunks: `(time=1, pressure_level=1, latitude=full, longitude=full)` for pressure-level variables.
- Uses explicit lossless Blosc/Zstandard compression at compression level 7 with bitshuffle for data variables and coordinates. It does not quantize, round, downcast, repack decoded values, or alter missingness.
- Publishes via a temporary store and atomically renames into the requested output directory, then reopens and validates exact semantic identity.

Run tests with:

```bash
pip install -r requirements.txt
pytest tests
```
