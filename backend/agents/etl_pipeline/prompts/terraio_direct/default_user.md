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

The following curated TerraIO source files are provided as read-only design references. Learn from their data acquisition, coverage, storage, caching, and local-read behavior, but generate a standalone pipeline that does not import or access TerraIO.

Selected reference files:
{reference_context_files_json}

Reference context size in characters:
{reference_context_size}

TerraIO reference context:
{reference_context}

Choose the pipeline file layout yourself, as long as every generated file lives under the pipeline output directory shown above. Prefer the simplest structure that makes the pipeline understandable, runnable, and testable for this dataset.

Include enough files for a human or later evaluator to understand how to run the pipeline. At minimum, include a clear executable entrypoint somewhere under the requested pipeline output directory.

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
