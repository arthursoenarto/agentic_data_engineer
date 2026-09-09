Create a correctness-first ETL pipeline artifact set for this DatasetContract.

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

The following curated TerraIO files are read-only architectural references. Extract useful invariants from them, but generate the smallest standalone pipeline that correctly satisfies this contract.

Selected reference files:
{reference_context_files_json}

Reference context size in characters:
{reference_context_size}

TerraIO reference context:
{reference_context}

Choose one or multiple Python modules based on what is easiest to keep correct. At minimum, include a documented executable entrypoint and offline tests. Do not add architectural layers merely because TerraIO contains them.

For this ERA5 pressure-level contract:
- Preserve exactly temperature at 500 hPa, temperature at 850 hPa, and geopotential at 500 hPa.
- A provider coordinate represented as 500.0 must satisfy a contract selector represented as "500".
- After selecting a pressure level, ensure scalar level coordinates cannot conflict when output variables are combined.
- Keep array operations lazy or bounded; a global month of multiple fields must not be fully loaded into memory at once.
- Write Zarr with xarray-supported Zarr encoding, and verify the same encoding path in a small synthetic write/read-back test.
- Group CDS fields only when grouping cannot widen the exact requested field-selector combinations.

Credential handling must remain secret-safe. Follow AccessContext exactly and do not invent alternate credential variable names.

The result will be evaluated without editing generated code by compiling it, running its generated tests, and transforming realistic ERA5-shaped NetCDF inputs into a readable Zarr store. Optimize for passing that evaluation, not for producing the most features.

PipelineGenerationResult JSON schema:
{response_schema_json}
