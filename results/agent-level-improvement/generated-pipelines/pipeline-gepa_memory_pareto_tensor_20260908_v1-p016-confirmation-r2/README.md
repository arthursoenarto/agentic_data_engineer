# ERA5 pressure-level fixture adapter

This directory contains a standalone implementation of
`pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for
`reanalysis_era5_pressure_levels` under the fixed regular-grid Zarr policy.

Key properties:

- Offline-only. A complete `cache_dir/source_fixture_manifest.json` is required;
  every listed raw file is verified by path containment, byte size, and SHA-256
  before decoding. The implementation never probes network services or
  credentials.
- Runtime contract driven. Fields, pressure-level selectors, UTC date/time
  bounds, product type, geography, and advanced options are validated against the
  frozen inventory and then used for all filtering.
- CF decoding. Xarray opens NetCDF/GRIB fixtures with `decode_cf=True` and
  `mask_and_scale=True`; source encodings are cleared before publication so
  decoded values are not silently repacked.
- Consolidated Zarr v3 output. Data variables use sample-access-aligned chunks:
  one time sample, one selected pressure-level plane, and the full selected
  latitude/longitude extent. Same-native-variable pressure-level channels are
  grouped into one array while preserving a real selector coordinate dimension.
- Joint objective policy. The preferred publication writes uncompressed data
  chunks explicitly. If the installed Zarr v3 stack cannot do that cleanly, the
  only fallback is lossless, read-speed-biased Blosc-LZ4 with `clevel=1` and
  byte shuffle.

The returned artifact declares every requested field-selector channel separately.
For grouped arrays, channels share `array_path` and disambiguate planes through
`selectors` plus `selector_coordinate_paths`.
