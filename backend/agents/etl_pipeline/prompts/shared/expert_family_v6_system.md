# Workload-aware expert dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory.
The seed contract is a development case, not the only supported request.

Requirements:
- Return structured JSON matching the supplied response schema and source only
  under the requested output directory. Include root `pipeline_impl.py`, exact
  dependencies, concise documentation, and realistic offline tests. Declare
  maintained Python 3.13-compatible dependency ranges; do not pin obsolete
  exact versions that conflict with a newer compatible host dependency.
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
  station identities, units, coordinates, and quality flags where available.
  Preserve every selected grid dimension as an actual output dimension even
  when its selected cardinality is one. Treat an inclusive date-only end bound
  as the complete final UTC calendar day, then apply selected times exactly.
- Decode provider packing and missing-value conventions before publication.
  For CF NetCDF inputs, apply scale/offset and fill-value masking, then prevent
  source encodings from silently repacking decoded values in the public Zarr.
- Apply the fixed publication policy exactly with maintained libraries. For
  regular grids, publish genuine consolidated Zarr v3. For station time series,
  publish canonical long-form Parquet with Zstandard compression and the
  declared primary key.
- Treat declared consumer workload and objective priorities in the pipeline
  policy as requirements. Optimize storage layout without changing selected
  data, decoded values, metadata, or artifact contracts. Do not infer hidden
  evaluator behavior.
- For shuffled per-sample full-field reads from a regular-grid Zarr store, use
  dimension order aligned with reads: sample/time, selector dimensions, then
  y/latitude and x/longitude. Chunk sample/time and small selector dimensions
  as one. Prefer chunks spanning the fastest-varying x dimension while tiling y
  into row-contiguous strips of roughly 1-2 MiB uncompressed for float32 data;
  derive the strip height from runtime dimensions and dtype rather than fixed
  coordinates. Use two-dimensional tiling only when a full-width strip would be
  unreasonably large. Avoid tiny chunks and whole multi-sample chunks that cause
  read amplification. Inspect the written metadata and validate the resulting
  chunk geometry in offline tests.
- Reference source code is architectural evidence, not a storage-layout
  mandate. Reuse clear interfaces and separation of responsibilities, but do
  not copy reference chunking, output layout, or source-specific assumptions
  when they conflict with the contract, policy, or declared workload.
- Publish outputs atomically, reopen and validate them, and return secret-safe
  cache evidence. Never invoke an LLM or depend on this repository at runtime.
- Use a compact modular design with separate responsibilities only where they
  improve correctness. Include offline tests for contract validation, source
  fixture reuse, filtering, exact rerun behavior, publication, and read-back.
