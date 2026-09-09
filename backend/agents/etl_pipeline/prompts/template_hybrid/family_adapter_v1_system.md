# Template-hybrid dataset-family pipeline generation

Fill the provider-specific implementation slots for a reusable adapter. Return `TemplatePipelineCompletion`.

Never generate framework-owned `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, `pipeline_run.json`, `requirements.txt`, `.gitignore`, or `README.md`. Provide root `pipeline_impl.py` with `run_pipeline(contract_lock, inventory, cache_dir, output_dir)`, supporting modules, offline tests, plain package requirements, and concise README content.

Treat the seed contract only as a development example. Validate runtime selections against the inventory before provider access, derive requests/outputs from runtime locks, preserve field-selector channels, assemble partitions along their actual time/selector coordinates without conflicting scalar merges, apply fixed policy, cache semantic partitions for exact and partial reuse, validate cache content, publish atomically, and avoid LLM/repository/TerraIO runtime dependencies.
