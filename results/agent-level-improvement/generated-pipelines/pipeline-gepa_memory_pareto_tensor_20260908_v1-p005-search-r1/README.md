# ERA5 pressure-level regular-grid adapter

This standalone adapter implements `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen `reanalysis_era5_pressure_levels` inventory and the fixed regular-grid Zarr policy.

Key properties:

- Fixture-first and credential-free: `cache_dir/source_fixture_manifest.json` is verified before any source decoding. No network acquisition is implemented.
- Runtime-driven filtering: requested variables, pressure-level selectors, dates, UTC selected times, and geographic area are derived from the supplied contract lock.
- Date-only inclusive end bounds include the complete final UTC calendar day before exact selected times are applied.
- CF decoding uses xarray `decode_cf=True` and `mask_and_scale=True`; source scale/offset/fill encodings are stripped before publication to avoid repacking decoded values.
- Publication is atomic to `dataset.zarr` under `output_dir`.
- Output is genuine consolidated Zarr v3 with real `time`, `pressure_level`, `latitude`, and `longitude` coordinate arrays when present.
- Throughput-oriented data-variable chunks are explicit: `(time=1, pressure_level=1, latitude=full, longitude=full)` for pressure-level arrays, and `(time=1, latitude=full, longitude=full)` otherwise.
- Data variables use an explicit lossless Blosc-LZ4 low-compression codec when supported by the host Zarr path.

The adapter returns the framework-declared `dataset_artifact_layout.v1` with channel IDs using the canonical field-selector format, e.g. `temperature[pressure_level="500"]`.
