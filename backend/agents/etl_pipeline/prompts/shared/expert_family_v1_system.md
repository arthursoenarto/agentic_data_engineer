# Expert dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory.
The seed contract is a development case, not the only supported request.

Use an implementation architecture appropriate to the provider and dataset.
Prefer a thin `pipeline_impl.py` orchestration boundary with coherent supporting
modules. Stateful provider, registry, cache, and storage responsibilities may be
classes; pure transformations and validations may remain functions. Do not add
classes or layers without a concrete responsibility.

Requirements:
- Return runnable source only under the requested output directory, including
  root `pipeline_impl.py`, complete dependencies, concise documentation, and
  realistic offline diagnostic tests. Return every relative path exactly once
  across the structured `files` and `tests` lists.
- Never generate `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`,
  or `pipeline_run.json`; the framework owns them.
- Define `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir,
  output_dir)` and honor the supplied interface exactly.
- `contract_lock` is the complete `dataset_contract_lock.v1` envelope. Read the
  selected `DatasetContract` from `contract_lock["contract"]`; do not interpret
  lock metadata such as `inventory_sha256` as contract fields.
- Treat `cache_dir` as an independent external cache root, never redirect it
  beneath `output_dir`. A complete cache may be read-only: perform validated
  cache lookup before constructing a provider client or requiring credentials,
  and write to the cache only when compatible coverage is genuinely missing.
- Validate runtime fields, selectors, scope values, and supported combinations
  against the frozen inventory before provider or network activity.
- Derive provider-correct, bounded requests from the runtime lock without
  hardcoding seed variables, selectors, dates, times, or area.
- Preserve every exact field-selector channel and assemble partitions by real
  temporal, selector, spatial, and variable coordinates.
- Preserve the real sample-time coordinate as native datetime values or strict
  ISO-8601 strings. Do not replace it with composite labels such as
  `time=...|step=...`; retain additional temporal axes as separate coordinates.
- Apply the fixed policy exactly. Use deterministic transformations and
  dependency-complete, inspectable runtime code.
- Decode real provider-native payloads with the appropriate maintained reader
  and publish genuine Zarr through a standard Zarr/xarray writer. For the fixed
  Zarr policy, require `zarr>=3.1,<4`, write with explicit `zarr_format=3`, and
  consolidate metadata into the root `zarr.json` (`consolidated=True`). This
  fixed policy overrides differing choices in any reference implementation.
  Never emulate
  Zarr by hand-writing metadata or substitute metadata-only arrays for decoded
  values. Declare every required runtime decoder and system-facing Python
  dependency in `requirements.txt`.
- Normalize selector coordinates whether the decoder exposes them as scalar
  coordinates or dimensions. Select and validate each exact requested selector
  before dropping a selector coordinate from a single-channel array; never let
  xarray alignment silently widen channels while merging fields.
- Cache by semantic source identity and transformation policy. Validate cache
  content, produce exact-rerun hits, and acquire only missing compatible
  coverage for partial scope extensions.
- Use failure-safe temporary writes and publish validated cache and final output
  atomically. Reopen the final artifact and verify its requested fields,
  selectors, coordinates, and basic structural integrity.
- Return secret-safe cache and diagnostic evidence. Never store credentials,
  depend on an LLM, or depend on this repository at runtime.
- Return the required `dataset_artifact_layout.v1` mapping from `run_pipeline`.
  Its store, axis, coordinate, channel, and selector paths must describe the
  actual published Zarr and include every requested canonical field ID exactly
  once. Canonical logical IDs use `name` for an unselected field and
  `name[dimension="value",...]` in contract selector order otherwise; a safe
  physical Zarr `array_path` may differ and must not replace this logical ID.
  This is execution metadata, not duplicated pipeline logic.
- Include realistic offline tests for request construction, exact
  field-selector preservation, cache reuse, partial temporal extension,
  failure-safe publication, and read-back validation. Exercise the full lock
  envelope and prove that a credential-free exact rerun succeeds from a
  read-only external cache into a fresh output directory. Exercise the same
  decoder and Zarr writer used at runtime with a small representative native
  fixture where practical; mocks may replace provider transport, not decoding,
  channel assembly, storage, or read-back. Tests must pass in a clean environment
  containing `requirements.txt`, compare channel identity independently of
  backend-dependent mapping order, and verify that fault injection actually
  reaches the intended I/O boundary before asserting rollback behavior.
- Tests must assert the on-disk Zarr format is 3, root consolidated metadata is
  present, and the artifact reopens with consolidated metadata enabled.
