# Dataset-family pipeline generation

Generate a standalone deterministic adapter for one frozen dataset inventory. The seed contract is a development example, not the pipeline's only supported request.

Requirements:
- Return structured JSON matching the supplied `PipelineGenerationResult` schema.
- Generate source only under the requested output directory, including root `pipeline_impl.py`, complete `requirements.txt`, concise documentation, and offline tests.
- Never generate `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, or `pipeline_run.json`; the framework owns them.
- `pipeline_impl.py` must define `run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
- Validate every runtime field, selector, scope value, and supported combination against the supplied inventory before provider or network activity.
- Build provider requests and outputs from the runtime lock. Do not hardcode the seed variables, selectors, dates, times, or area as the only supported request.
- Apply the fixed policy exactly. Do not expose unsupported transport choices as user capabilities.
- Cache by semantic source identity and transformation policy at a granularity that supports exact-rerun hits and partial date/scope extension.
- Assemble multiple cached partitions by their real coordinates: temporal extensions must concatenate along time, selector extensions along selector dimensions, and distinct variables must merge without discarding or overriding requested channels.
- Validate cache entries before reuse; write cache entries and outputs through temporary paths and publish only after validation.
- Return secret-safe cache evidence with hit, miss, acquired, reused-key, and acquired-key values.
- Decode provider-native payloads and write genuine Zarr with maintained
  libraries; do not hand-write Zarr metadata or publish metadata-only values.
  When the fixed policy specifies Zarr, require `zarr>=3.1,<4`, write explicit
  Zarr format 3, and consolidate metadata into the root `zarr.json`.
- Return the required `dataset_artifact_layout.v1` execution mapping described
  by the framework interface. It must name the actual Zarr store, regular-grid
  axes and coordinates, and every canonical requested field-selector channel.
- Do not depend on an LLM, this repository, or TerraIO at runtime.
- Include offline tests for at least two dates, two exact field-selector channels, exact cache reuse, and partial temporal extension.
- For Zarr output, test format 3, consolidated root metadata, and consolidated
  reopen using the same writer path as production.
