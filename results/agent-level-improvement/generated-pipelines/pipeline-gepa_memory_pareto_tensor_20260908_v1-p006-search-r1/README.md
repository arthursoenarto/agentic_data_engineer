# ERA5 pressure-level family adapter

This directory contains a standalone implementation of `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` for the frozen ERA5 pressure-level inventory and the fixed regular-grid Zarr publication policy.

Key behavior:

- Reads and validates the selected `dataset_contract.v1` from a lock envelope.
- Validates requested variables, pressure-level selectors, UTC times, date range, product type, area, and acquisition options against the supplied inventory.
- Uses `cache_dir/source_fixture_manifest.json` as the authoritative offline source. Every listed raw file is size/SHA-256 verified before xarray decoding. No provider client, credential check, or network fallback is used.
- Decodes CF packing/missing conventions through xarray, filters from the runtime lock, preserves native coordinate values, missingness, attributes, timestamps, and selector dimensions.
- Publishes one data array per requested field-selector channel. Singleton pressure-level selections remain real Zarr dimensions using channel-specific selector coordinate paths to avoid coordinate conflicts between channels.
- Writes consolidated Zarr v3 with deterministic sample-access-aligned chunks: `time=1`, selector plane `=1`, full selected latitude, and full selected longitude.
- Uses explicit lossless footprint-oriented Blosc-Zstd compression (`clevel=7`, bitshuffle when available, otherwise byte shuffle) for Zarr arrays.
- Stages output in a temporary store, reopens and validates it, then exposes `dataset.zarr`.

The implementation never imports framework-owned files and does not create `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, or `pipeline_run.json`.
