# Minimally instructed typed dataset pipeline generation

Generate a standalone deterministic Python adapter for the supplied dataset and
goal. Return runnable files under the requested output directory, including a
root `pipeline_impl.py` with `run_pipeline(contract_lock, inventory, cache_dir,
output_dir)`.

Honor the supplied output and execution policies. The framework owns
`run_pipeline.py`, `pipeline_contract.json`, `manifest.json`, and
`pipeline_run.json`; do not return them. Do not write outside the output or
cache roots, include secret values, depend on this repository at runtime, or
invoke an LLM from generated code. Prefer a supplied local source fixture over
network access.
