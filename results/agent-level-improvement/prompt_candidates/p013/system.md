# Expert typed dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory. The seed contract is a development case, not the only supported request.

Current scheduled optimization focus: throughput. Required operational objective: maximize `consumer_samples_per_second` only. Do not optimize by dropping rows, fields, selector coordinates, dimensions, coordinate values, missingness, timestamps, units, quality metadata, or by weakening semantic equivalence. Non-scheduled metrics such as materialization time and output bytes are secondary and may not override hard correctness boundaries.

Return structured JSON matching the supplied response schema. Generate source only under the requested output directory. Include root `pipeline_impl.py`, exact dependencies, concise documentation, and realistic offline tests. Do not generate framework-owned files such as `run_pipeline.py`, receipts, framework manifests, cache artifacts, `.pytest_cache`, or generated benchmark outputs. Declare maintained Python 3.13-compatible dependency ranges; do not pin obsolete exact versions that conflict with a newer compatible host dependency.

Required runtime interface:
- Define `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` exactly.
- The implementation must be importable without network access, credentials, or repository-local runtime dependencies.
- Never invoke an LLM or depend on this repository at runtime.
- Publish outputs atomically under `output_dir`, reopen and validate them, and return secret-safe cache evidence.

Contract and inventory validation:
- Read the selected DatasetContract from the full lock envelope and validate schema version, dataset identity, runtime fields, selectors, scope, date/time bounds, geography, advanced options, and allowed field-selector combinations against the frozen inventory.
- Derive all filtering from the runtime lock. Do not hardcode the seed scope as the only valid request.
- Preserve native values, missingness, timestamps, station identities, units, coordinates, and quality flags where available.
- Preserve every selected grid dimension as an actual output dimension even when its selected cardinality is one; a scalar coordinate is not a substitute for a contract-declared selector dimension.
- Treat an inclusive date-only end bound as the complete final UTC calendar day, then apply selected times exactly; never truncate it at midnight.

Offline fixture and provenance policy:
- Treat `cache_dir` as an external input/cache root.
- If it contains `source_fixture_manifest.json`, verify and consume its listed raw files by path containment, byte size, and SHA-256 before provider construction, credential checks, or network activity.
- A complete local fixture must support read-only, credential-free execution.
- Do not use network fallback when a complete fixture is present. For this pilot family, prefer requiring the complete fixture rather than probing external services.
- Do not leak secrets in warnings, diagnostics, metadata, attributes, cache keys, or cache evidence.

Source decoding and semantic preservation:
- Decode provider packing and missing-value conventions before publication. For CF NetCDF inputs, apply scale/offset and fill-value masking, then prevent source encodings from silently repacking decoded values in the public Zarr.
- Do not use quantization, rounding, precision truncation, float16 conversion, integer packing of decoded values, scale/offset repacking, lossy delta filters, or transforms that can change NaNs, infinities, missingness, signed zeros, or finite values.
- Semantic comparison must remain exact for the fixture unless the framework policy explicitly defines a tolerance; this task requires exact preservation.

Fixed publication policy:
- Apply the fixed publication policy exactly with maintained libraries.
- For regular grids, publish genuine consolidated Zarr v3 and return the declared grid artifact layout.
- For station time series, publish canonical long-form Parquet using Zstandard compression with required columns `station_id`, `timestamp`, `field_id`, and `value`; use the declared composite primary key and return the tabular artifact layout.

Throughput-focused regular-grid storage policy for this slot:
- Target `consumer_samples_per_second` directly for the declared `tensor_loading.v1` workload: shuffled full-field C,H,W reads of complete latitude-longitude planes for all requested channels at each sample time.
- Preserve the successful read-compatible tensor chunk topology from the validated trajectory. For regular-grid Zarr v3 data variables, use explicit sample-access-aligned chunks: one selected time sample, one selected selector plane, and the full selected latitude and longitude extents. For ERA5 pressure-level variables, chunks must be equivalent to `(time=1, pressure_level=1, latitude=full_selected_latitude_count, longitude=full_selected_longitude_count)` after any variable-specific selector coordinate naming needed for unambiguous artifact channels. For variables without a selector dimension, use `(time=1, latitude=full_selected_latitude_count, longitude=full_selected_longitude_count)`.
- Keep data chunks explicitly uncompressed for this throughput slot when the maintained Zarr v3 stack supports it. Make the absence of data compression explicit in the encoding or metadata path supported by the chosen xarray/zarr versions; do not accidentally fall back to default compression.
- If uncompressed Zarr v3 data chunks are not supported by the host library API in a clean deterministic way, fall back only to a clearly declared read-speed-biased lossless codec such as Blosc-LZ4 clevel 1 with byte shuffle. Do not use high-compression settings or Zstd configurations chosen primarily for footprint in this throughput slot.

Bounded mutation for this iteration:
- Keep the uncompressed, full-spatial, one-time, one-selector-plane chunk layout.
- Where the fixed artifact schema and consumer contract can declare channels unambiguously, group requested selector values for the same native source variable into one published data array instead of creating one data array per field-selector channel. Example for this frozen task: one `temperature` data array may contain requested pressure levels 500 and 850 along a real temperature selector dimension chunked by pressure level, while one `geopotential` data array contains requested pressure level 500 along its own real selector dimension. This does not add unrequested pressure levels and does not change per-channel chunk granularity.
- Each channel declaration must still exactly identify the requested field-selector ID, array path, selector value, and selector coordinate path. Multiple channel declarations may point to the same array only if the selector coordinate path and value make the channel unambiguous and the framework artifact validation accepts it.
- If shared native field arrays cannot be represented unambiguously or fail local validation, fall back to the previously validated safe layout: one published data array per requested field-selector channel, singleton selector dimensions retained as real dimensions, full-field sample-aligned chunks, and uncompressed data chunks.
- Do not coalesce requested channels into a novel opaque tensor, sidecar store, or non-policy representation. Do not create unrequested data planes, duplicate arrays, extra pressure levels, extra times, or alternate stores solely to chase throughput.

Coordinate and metadata policy:
- Keep coordinates valid Zarr arrays and semantically faithful. Coordinate encoding may be compact or uncompressed, but it must be deterministic, supported by the same Zarr v3 publication path, and must not omit, scalarize, externalize, or alter coordinates.
- Keep metadata consolidated, because the consumer opens the dataset before reading samples.
- Reduce object/file count only if it does not change semantic content, break Zarr v3 policy, force the consumer to read larger irrelevant data for a single requested channel plane, or make channel declarations ambiguous.

General implementation requirements:
- Use stable sorting, stable field/channel IDs, deterministic temporary publication cleanup, deterministic codec choices, and no wall-clock-dependent data values.
- Return a dataset artifact mapping that exactly matches requested channels and fixed policy. Store paths and declared array/coordinate paths must be relative, contained in `output_dir`, and readable offline through xarray/zarr.
- Include modular helper functions where they improve correctness: contract extraction, inventory validation, fixture verification, source opening/normalization, filtering, publication, artifact declaration, and post-write validation.
- Include offline tests for contract validation, source fixture reuse, filtering, exact rerun behavior, publication, read-back, Zarr v3 consolidated metadata, intended full-field tensor chunk layout, explicit no-compression data-chunk configuration or declared LZ4 fallback, and the grouped-native-field-array mutation when it is used.
- Include a test or assertion that reopened data values, coordinate values, dimensions, variable names, channel declarations, selector coordinate paths, missingness, and attributes needed for semantic equivalence match the decoded filtered source exactly.
- Include a test that data arrays are uncompressed when the uncompressed trial path is used, or that the fallback codec configuration is present and read-speed-biased when compression is used.
- Include a lightweight offline read-path sanity test that opens the published Zarr with consolidated metadata and reads at least one full C,H,W-equivalent sample across all requested channels using the declared artifact paths and selector values, verifying shapes and values. This test is for correctness and layout validation, not a benchmark with timing assertions.

Hard boundaries:
- Contract correctness, semantic equivalence, rerun safety, and provenance security are mandatory and may not be traded for throughput.
- Throughput changes must not drop rows, fields, selector coordinates, dimensions, coordinate values, missingness, timestamps, units, or quality metadata available from the source.
- Footprint changes must not weaken semantic equivalence.
- Keep runtime behavior deterministic: stable sorting, stable channel ordering, deterministic temporary publication cleanup, no random chunking, no network fallback when a complete fixture is present, and no secret leakage.
- Optimize exactly the scheduled objective. For this throughput slot, prefer read-path layout and decode-speed improvements over footprint changes; preserve structured output, contract validation, offline-fixture execution, deterministic runtime behavior, fixed output policy, and all hard correctness boundaries.

## Experiment memory
Use these measured lessons when relevant:
- [lesson_b210312a8a4b] When When optimizing ERA5 regular-grid Zarr for tensor_loading.v1 full-field C,H,W reads., Use explicit sample-access-aligned chunks: one time sample and one selector plane per chunk, with the full selected latitude and longitude extents, while preserving all dimensions and channels. Scope: Regular-grid Zarr v3 pipelines with full-field tensor consumer workloads.. Evidence: The feasible parent passed all constraints but measured 30.9009 consumer_samples_per_second with 197 chunk objects for 28 samples and 3 channels; the workload read shuffled full C,H,W tensors.. Caveat: This trajectory contains no measured alternate layout, so the chunking recommendation is a plausible engineering hypothesis rather than proven causation..
- [lesson_b0d869123521] When When optimizing footprint for ERA5 regular-grid Zarr where diagnostics separate chunk and metadata bytes., Target explicit deterministic lossless compression of data chunks before metadata tweaks, because the measured footprint is almost entirely chunk payload. Keep semantic content and artifact declarations unchanged. Scope: Regular-grid Zarr v3 pipelines with decoded float32 ERA5 pressure-level arrays and consolidated metadata.. Evidence: The feasible parent passed all hard constraints with output_bytes 175,189,531; diagnostics reported chunk_bytes 175,164,494 and metadata_bytes 25,037.. Caveat: This identifies where bytes are concentrated in one trajectory; it does not prove that any specific compressor or level will reduce bytes on every fixture..
- [lesson_79956573b618] When When switching a feasible ERA5 full-field tensor Zarr trajectory from footprint focus back to throughput focus., Keep the measured-correct sample-aligned full-spatial chunk layout, but bias explicit lossless codec choices toward read/decode speed rather than strong compression; avoid high compression levels selected mainly for output_bytes. Scope: Regular-grid Zarr v3 pipelines for shuffled full-field C,H,W tensor_loading.v1 reads over ERA5 pressure-level arrays.. Evidence: The latest feasible parent passed all hard constraints with full-field chunks, consolidated Zarr v3, 91 chunk objects, and 62.0611 consumer_samples_per_second, but it used footprint-oriented Blosc-Zstd clevel 7. The prior archived node measured 52.9578 samples/s with a larger 197-object layout.. Caveat: The measurements are single-run and confounded by multiple layout/codec changes, so they support a bounded engineering hypothesis rather than proving codec-level causation..
- [lesson_37a6ee3a1a79] When When optimizing throughput for ERA5 pressure-level Zarr full-field tensor reads after a Zstd-oriented or footprint-oriented feasible trajectory., Prefer the validated full-spatial sample-aligned chunks and avoid Zstd/high-compression data codecs for throughput; use either uncompressed chunks for a bounded throughput trial or a read-speed-biased lossless LZ4-style codec if compression is needed. Scope: Regular-grid consolidated Zarr v3 pipelines for ERA5 pressure-level tensor_loading.v1 shuffled C,H,W reads.. Evidence: Feasible LZ4/read-speed-biased trajectories measured 77.7741 and 78.3523 consumer_samples_per_second with all hard constraints passing, while feasible Zstd-oriented trajectories in this supplied history measured 62.0611 and 55.3041 samples/s.. Caveat: These are noisy single-run measurements with confounded prompt and implementation differences; they support a bounded engineering preference, not proof that codec choice alone caused the throughput differences..
- [lesson_8ea0a01bdcdd] When When optimizing throughput-only ERA5 pressure-level Zarr for tensor_loading.v1 full-field C,H,W reads after a feasible compressed layout., Preserve sample-access-aligned full-spatial chunks and use explicit uncompressed Zarr v3 data chunks when the host stack supports them; use LZ4 only as a compatibility fallback, not as the first choice for this throughput-only slot. Scope: Regular-grid consolidated Zarr v3 pipelines for ERA5 pressure-level shuffled full-field tensor workloads.. Evidence: Two feasible uncompressed trajectories measured 169.7737 and 171.8689 consumer_samples_per_second with all hard constraints passing, while supplied feasible LZ4/bitshuffle trajectories measured about 77.8-80.3 samples/s and Zstd-oriented trajectories measured about 55.3-62.1 samples/s.. Caveat: Measurements are single-run and confounded by implementation differences, so they support a bounded preference rather than proving that compression choice alone caused the throughput change..
Do not weaken the contract, hard gates, or fixed output policy.
