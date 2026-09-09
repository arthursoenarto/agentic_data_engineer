# Compact workload-aware expert dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory.
The seed contract is a development case, not the only supported request.

Return a compact structured bundle that fits the response budget: use at most
six files, keep implementation and tests focused, avoid embedded fixture data,
and do not repeat documentation in source comments.

Requirements:
- Source only under the requested output directory. Include root
  `pipeline_impl.py`, exact maintained Python 3.13-compatible dependencies,
  concise documentation, and focused offline tests. Do not generate
  framework-owned files. Define
  `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
- Read the full lock envelope. Validate runtime fields, selectors, scope, and
  combinations against the inventory; do not hardcode the seed request.
- Prefer a verified `source_fixture_manifest.json` in `cache_dir` before any
  provider construction, credential check, or network action. A complete local
  fixture must run read-only and without credentials.
- Preserve decoded values, missingness, timestamps, units, coordinates, quality
  flags, and selected dimensions. Treat an inclusive date-only end as the full
  final UTC day. Decode provider scale/offset and fill conventions and prevent
  source encodings from repacking public values.
- Do not globally combine raw datasets with unequal selector coverage. Decode
  and select each requested field-selector channel first, normalize it, and
  then assemble the public dataset. This applies to pressure, band, depth, and
  other selector dimensions.
- Follow the fixed publication policy. For regular grids, publish one genuine
  consolidated Zarr v3 root dataset with shared sample and spatial coordinates
  and a unique root data array per exact requested field-selector channel.
  Preserve cardinality-one selector dimensions and their coordinate arrays.
- Return the `dataset_artifact` mapping literally. `dimensions` values are
  actual dimension names; `coordinates` values are Zarr array paths. If array
  `time` has dimension `sample`, report `dimensions.sample="sample"` and
  `coordinates.sample="time"`. Never put a coordinate path in `dimensions`.
  Resolve and open every returned coordinate, channel, and selector path in a
  focused offline test.
- Treat declared workload and objective priorities as requirements without
  changing data semantics. For shuffled per-sample full-field Zarr reads, order
  arrays as sample/time, selectors, y/latitude, x/longitude. Chunk sample/time
  and cardinality-one selectors as one. Prefer full-width x chunks and derive y
  strips near 1-2 MiB uncompressed for float32 from runtime sizes and dtype.
  Avoid tiny chunks and chunks spanning multiple samples.
- Use reference code only for architectural lessons. Do not copy its output
  layout, chunking, or dataset assumptions when they conflict with the current
  contract, policy, workload, or standalone target.
- Publish atomically, reopen and validate outputs, and return secret-safe cache
  evidence. Never call an LLM or depend on this repository at runtime.
