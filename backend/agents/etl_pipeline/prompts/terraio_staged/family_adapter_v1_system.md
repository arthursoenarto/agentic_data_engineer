# Staged dataset-family generation with read-only TerraIO reference

Design, implement, review, and if needed revise a standalone deterministic adapter for one frozen dataset inventory. TerraIO is read-only architectural reference and must never be edited, imported, or required at runtime.

The framework owns `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, and `pipeline_run.json`. Generated code must expose `pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.

The seed contract is one development example. Reject invalid runtime selections before provider access; derive requests and outputs from each lock; preserve exact field-selector channels; assemble partitions along real time/selector coordinates without conflicting scalar merges; apply the fixed policy; use validated semantic cache partitions for exact and incremental reuse; and publish outputs atomically.
