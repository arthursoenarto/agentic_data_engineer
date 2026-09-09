Fill the implementation slots for this pipeline scaffold.

Dataset directory:
{dataset_dir}

Pipeline output directory:
{output_dir}

Network probing during generation is allowed:
{allow_network_probe}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

The framework will create:
- `run_pipeline.py`, which imports and executes `pipeline_impl.main`;
- `requirements.txt` from your dependency list;
- `.gitignore`;
- `README.md` from your `readme` field;
- `manifest.json` with framework-owned provenance.

Provide:
- `pipeline_impl.py` and only the additional implementation modules genuinely needed;
- offline tests under `tests/`;
- a complete dependency list, including the test runner;
- concise README content with execution, credentials by variable name, outputs, rerun behavior, and validation.

The complete no-argument workflow must:
1. load permitted `.env` files without overriding explicit environment values;
2. construct the provider request for the exact selected fields and selectors;
3. retrieve or safely reuse validated raw inputs;
4. transform in bounded memory;
5. write to a temporary derived location;
6. reopen and validate scientific structure and scope;
7. atomically publish the final artifact;
8. return nonzero on any failure.

Do not return protected scaffold paths. Paths in `files` and `tests` are relative to the pipeline output directory, so use `pipeline_impl.py`, supporting module paths, and `tests/test_*.py` rather than prefixing them with `{output_dir}/`.
