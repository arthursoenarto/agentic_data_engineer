# Workload-aware expert dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory.
The seed contract is a development case, not the only supported request.

Requirements:
- Return structured JSON matching the supplied response schema and source only
  under the requested output directory. Include root `pipeline_impl.py`, only
  direct runtime dependencies with broad compatible ranges, concise
  documentation, useful modules, and focused offline tests. Do not exact-pin a
  library already constrained by the fixed policy, embed fixture data, return
  generated caches, or generate framework-owned files.
- Define
  `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
  Read the selected contract from the full lock envelope and validate runtime
  fields, selectors, scope, and combinations against the inventory.
- Treat `cache_dir` as an external input/cache root. If it contains
  `source_fixture_manifest.json`, verify and consume its listed raw files before
  provider construction, credential checks, or network activity. A complete
  fixture must support read-only, credential-free execution.
- Derive filtering from the runtime lock. Preserve decoded values, missingness,
  timestamps, units, coordinates, quality flags, and every selected dimension.
  Date-only end bounds include the complete final UTC day before selected times
  are applied.
- Treat each exact field-selector combination as one output channel. Provider
  filtering may collapse a selected singleton coordinate to a scalar. In that
  case, validate the scalar identity and explicitly reconstruct a length-one
  selector dimension; do not index a scalar as a vector or silently omit the
  contract-declared dimension. When an axis remains present, select without
  dropping it. Normalize channels before assembling the public dataset.
- Decode provider packing and missing-value conventions before publication and
  clear source encodings that could silently repack decoded public values.
- Apply the fixed publication policy exactly. For regular grids, publish one
  genuine consolidated Zarr v3 root dataset with shared sample and spatial
  coordinates and one uniquely named data array per requested channel. Return
  the declared artifact mapping literally: `dimensions` contains actual
  dimension names and `coordinates` contains array paths. Resolve and reopen
  every returned path in an offline test.
- Optimize only the declared workload and objective priority without changing
  semantics or the artifact contract. For shuffled per-sample full-field reads,
  use dimension order `sample/time`, selector dimensions, `y/latitude`, then
  `x/longitude`. Chunk sample/time and singleton selectors as one. Prefer
  full-width x chunks and cache-friendly y strips around 1-2 MiB uncompressed
  for float32, deriving shapes from runtime dimensions and dtype. Avoid tiny
  chunks and chunks spanning multiple samples. Use only stable codec or sharding
  APIs that are available in the maintained runtime and verified by read-back;
  otherwise retain compatible defaults.
- Use reference source only for architecture and interface lessons. Do not copy
  its output layout, chunking, or dataset assumptions when they conflict with
  the current contract, policy, workload, or standalone target.
- Publish atomically, reopen and validate outputs, and return secret-safe cache
  evidence. Never invoke an LLM or depend on this repository at runtime.
- Keep the design compact and modular. Test contract validation, fixture reuse,
  singleton selectors, filtering, exact reruns, publication, artifact mapping,
  chunk geometry, and read-back offline.
