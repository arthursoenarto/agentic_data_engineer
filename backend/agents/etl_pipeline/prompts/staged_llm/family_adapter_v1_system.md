# Staged dataset-family pipeline generation

Design, implement, review, and if needed revise a standalone deterministic adapter for one frozen dataset inventory. The seed contract is a development example only.

The framework owns `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, and `pipeline_run.json`. Generated code must expose `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)` and must not replace framework-owned files.

Reject invalid runtime selections before network activity. Derive requests and outputs from each runtime lock. Apply the fixed policy, preserve exact field-selector channels, assemble partitions along their actual time/selector coordinates without conflicting scalar merges, use semantic incremental caching, validate reused content, publish atomically, and remain independent of LLMs, this repository, and TerraIO at runtime.
