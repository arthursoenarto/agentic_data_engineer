# ERA5 pressure-level family adapter

This directory contains a standalone deterministic implementation of `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and the fixed regular-grid Zarr publication policy.

The adapter validates the selected `dataset_contract.v1` from a full lock envelope, checks requested variables, pressure-level selectors, product type, date/time selections, area bounds, and inventory enumerations, then materializes a consolidated Zarr v3 store at `dataset.zarr` under the supplied output directory.

`cache_dir` is treated as an external framework-owned cache/input root. If `cache_dir/source_fixture_manifest.json` exists, every listed file is verified by size and SHA-256 before source decoding. A complete fixture runs read-only and credential-free; this implementation deliberately performs no provider construction or network acquisition when no verified fixture is present.

Source fixtures may be GRIB, CF NetCDF, or Zarr. CF decoding uses xarray with mask-and-scale enabled; source encodings are cleared before publication so decoded values are not silently repacked in the public Zarr. Each requested field-selector pair is published as its own data array, and the selected `pressure_level` is preserved as a real length-one selector dimension rather than converted to a scalar coordinate.
