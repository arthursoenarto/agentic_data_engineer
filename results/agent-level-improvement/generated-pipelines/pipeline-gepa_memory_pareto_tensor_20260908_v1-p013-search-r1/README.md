# ERA5 pressure-level family adapter

Pipeline ID: `pipeline-gepa_memory_pareto_tensor_20260908_v1-p013-search-r1`

This standalone adapter materializes verified local ERA5 pressure-level source fixtures as consolidated Zarr v3 regular-grid datasets. It performs no network probing, does not construct provider clients, and requires `cache_dir/source_fixture_manifest.json` to describe all raw fixture files before decoding.

## Runtime interface

```python
pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)
```

The function validates the selected `dataset_contract.v1` against the frozen `dataset_inventory.v1`, verifies fixture file containment, byte size, and SHA-256, decodes with xarray, filters from the runtime lock, writes `dataset.zarr` atomically under `output_dir`, reopens the store with consolidated metadata, validates exact semantic equivalence, and returns the fixed artifact layout plus secret-safe cache evidence.

## Throughput layout

For regular-grid tensor-loading workloads, data variables are written with sample-aligned full-spatial chunks:

- `time=1`
- requested pressure-selector plane `=1`
- full selected latitude extent
- full selected longitude extent

Requested pressure levels for the same native source variable are grouped in one data array when unambiguous. For example, temperature at 500 and 850 hPa is published as `temperature(time, temperature_pressure_level, latitude, longitude)`, while geopotential at 500 hPa is published as `geopotential(time, geopotential_pressure_level, latitude, longitude)`. Channel declarations identify each field-selector ID, array path, selector value, and selector coordinate path.

The primary publication attempt uses explicit uncompressed Zarr v3 data chunks. If a maintained host stack cannot express that deterministically, the code falls back only to read-speed-biased lossless Blosc-LZ4, compression level 1, byte shuffle.

## Fixture manifest

`cache_dir/source_fixture_manifest.json` must follow `source_fixture_manifest.v1` with entries containing `entry_id`, `relative_path`, `size_bytes`, and lowercase `sha256`. Paths must resolve beneath `cache_dir`.

Supported offline fixture file types are NetCDF (`.nc`, `.nc4`, `.netcdf`), GRIB (`.grib`, `.grib2`, `.grb`, `.grb2`, via cfgrib), and source Zarr stores.
