# Workload-aware expert dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory.
The seed contract is a development case, not the only supported request.

Requirements:
- Return structured JSON matching the supplied response schema and source only
  under the requested output directory. Include root `pipeline_impl.py`, exact
  dependencies, concise documentation, and realistic offline tests. Declare
  maintained Python 3.13-compatible dependency ranges.
- Do not generate framework-owned files. Define
  `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
- Read the selected contract from the full lock envelope and validate runtime
  fields, selectors, scope, and combinations against the inventory.
- Treat `cache_dir` as an external input/cache root. If it contains
  `source_fixture_manifest.json`, verify and consume its listed raw files before
  provider construction, credential checks, or network activity. A complete
  local fixture must support read-only, credential-free execution.
- Derive all filtering from the runtime lock. Do not hardcode the seed scope as
  the only valid request. Preserve native values, missingness, timestamps,
  units, coordinates, quality flags, and every selected grid dimension,
  including selector dimensions of cardinality one. Treat an inclusive
  date-only end bound as the complete final UTC calendar day.
- Decode provider packing and missing-value conventions before publication.
  For CF NetCDF inputs, apply scale/offset and fill-value masking, then prevent
  source encodings from silently repacking decoded values in the public Zarr.
- Do not globally combine raw datasets whose fields legitimately have different
  selector coverage. Decode and select each requested field-selector channel
  first, normalize its dimensions and coordinates, and then assemble the public
  dataset. This avoids treating unequal pressure levels, bands, depths, or
  similar selectors as conflicting global coordinates.
- Apply the fixed publication policy exactly. For regular grids, publish one
  genuine consolidated Zarr v3 root dataset with shared spatial and sample
  coordinates and a unique root data array for each exact requested
  field-selector channel. For station time series, publish canonical long-form
  Parquet with Zstandard compression and the declared primary key.
- Return the regular-grid `dataset_artifact` interface literally. Values in
  `dimensions` are actual dimension names on the Zarr arrays; values in
  `coordinates` are Zarr array paths. For example, if a time coordinate array
  at path `time` has dimension `sample`, report
  `dimensions.sample="sample"` and `coordinates.sample="time"`. Never put a
  coordinate path in `dimensions`. Every channel `array_path` and selector
  coordinate path must exist in the same declared store, and the mapped arrays
  must use the declared dimensions. Test the returned mapping by resolving and
  opening every path, not only by checking that files exist.
- Treat declared consumer workload and objective priorities in the pipeline
  policy as requirements. Optimize storage layout without changing selected
  data, decoded values, metadata, or artifact contracts. Do not infer hidden
  evaluator behavior.
- For shuffled per-sample full-field reads from regular-grid Zarr, align array
  order with reads: sample/time, selector dimensions, y/latitude, x/longitude.
  Chunk sample/time and cardinality-one selector dimensions as one. Prefer
  chunks spanning x while tiling y into row-contiguous strips of roughly 1-2
  MiB uncompressed for float32 data. Derive chunk shapes from runtime sizes and
  dtype. Avoid tiny chunks and chunks spanning multiple samples.
- Reference source is architectural evidence, not a storage-layout mandate.
  Reuse clear interfaces and separation of responsibilities, but do not copy
  reference layout or source-specific assumptions when they conflict with the
  contract, policy, workload, or standalone artifact target.
- Publish atomically, reopen and validate outputs, and return secret-safe cache
  evidence. Never invoke an LLM or depend on this repository at runtime.
- Use a compact modular design. Include offline tests for contract validation,
  source fixture reuse, filtering, exact reruns, publication, artifact mapping,
  chunk geometry, and read-back.
