# Dataset-family pipeline generation with read-only reference context

Generate a standalone deterministic adapter for one frozen dataset inventory. Use the supplied TerraIO excerpts only as architectural reference. Never import TerraIO or assume it exists at runtime.

Requirements:
- Return structured JSON matching the supplied `PipelineGenerationResult` schema.
- Generate source only under the requested output directory, including root `pipeline_impl.py`, complete `requirements.txt`, concise documentation, and offline tests.
- Never generate `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, or `pipeline_run.json`; the framework owns them.
- `pipeline_impl.py` must define `run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.
- Treat the seed contract as one development example, not a fixed implementation scope.
- Validate runtime fields, selectors, scope, and combinations against the inventory before any provider call.
- Derive requests, decoding, output assembly, and validation expectations from the runtime lock.
- Apply the fixed policy exactly and cache semantic source partitions for exact and partial reuse.
- Assemble partitions along their actual time and selector coordinates; do not merge conflicting scalar coordinates or override requested channels.
- Validate cache content and publish cache/output artifacts atomically.
- Return secret-safe hit, miss, acquisition, and cache-key evidence.
- Do not use an LLM, this repository, or TerraIO at runtime.
- Test multi-date and multi-channel assembly as well as exact and partial cache reuse.
