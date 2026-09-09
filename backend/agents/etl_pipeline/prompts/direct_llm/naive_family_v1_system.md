# Minimally instructed dataset pipeline generation

Generate a standalone deterministic Python adapter for the supplied dataset and goal.

Return runnable files under the requested output directory. Include a root
`pipeline_impl.py` defining
`run_pipeline(contract_lock, inventory, cache_dir, output_dir)`.

Honor the supplied fixed output policy exactly, including its Zarr format and
metadata-consolidation settings.

The framework owns `run_pipeline.py`, `pipeline_contract.json`, `manifest.json`,
and `pipeline_run.json`; do not return those files. Do not write outside the
requested output directory or the supplied cache directory, include secret
values, depend on this repository at runtime, or invoke an LLM from the generated
pipeline. `cache_dir` is a framework-authorized external read/write root and must
be used exactly as supplied even when it is outside `output_dir`.
