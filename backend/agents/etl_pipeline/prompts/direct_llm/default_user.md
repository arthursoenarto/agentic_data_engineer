Create an ETL pipeline artifact set for this DatasetContract.

Dataset directory:
{dataset_dir}

Pipeline output directory under the dataset directory:
{output_dir}

Allow network probe during generation:
{allow_network_probe}

DatasetContract JSON:
{contract_json}

AccessContext JSON:
{access_context_json}

Choose the internal pipeline file layout yourself, as long as every generated file lives under the pipeline output directory shown above. Prefer the simplest structure that makes the pipeline understandable, runnable, and testable for this dataset.

The evaluator executes the pipeline from its output directory with `python run_pipeline.py`. Therefore:
- include `{output_dir}/run_pipeline.py` as a no-argument, end-to-end entrypoint;
- include `{output_dir}/requirements.txt` with all runtime dependencies;
- make the entrypoint retrieve, transform, store, reopen, and minimally validate the requested data;
- make reruns safe and resumable, reusing valid downloads where practical and replacing incomplete derived outputs safely;
- return a nonzero process exit code for any failed stage or validation.

You may create any additional modules that improve correctness. Do not force the implementation into one file.

The pipeline should be designed for the current selected fields and scope only. For ERA5 pressure-level fields, preserve each pressure_level selector exactly and avoid silently widening to unintended field-selector combinations.

Credential handling must be low-friction and secret-safe. Generated code should load `.env` files from the pipeline directory and its ancestors, then read credential values from environment variables. For Copernicus CDS datasets, prefer `CDSAPI_URL` and `CDSAPI_KEY`, while accepting `CDS_PERSONAL_ACCESS_TOKEN` and `CDS_API_TOKEN` as key aliases.

If AccessContext is not null, follow it exactly for:
- provider/client package and construction
- credential environment variable names and aliases
- whether access has already been verified
- `.env` loading behavior

Do not invent alternate credential variable names when AccessContext provides them.

PipelineGenerationResult JSON schema:
{response_schema_json}
