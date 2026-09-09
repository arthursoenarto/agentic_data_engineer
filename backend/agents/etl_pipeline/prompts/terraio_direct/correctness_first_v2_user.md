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

Use the curated TerraIO files below only to extract relevant engineering invariants. Generate the smallest standalone pipeline that correctly satisfies the contract; do not reproduce TerraIO's full architecture.

Selected reference files:
{reference_context_files_json}

Reference context size in characters:
{reference_context_size}

TerraIO reference context:
{reference_context}

The exact outputs are temperature at 500 hPa, temperature at 850 hPa, and geopotential at 500 hPa. Provider coordinates may represent these levels as integer, float, NumPy scalar, or string values. Scalar pressure coordinates must not remain to conflict during dataset combination.

The evaluation uses real ERA5 NetCDF files opened as Dask-backed xarray datasets. Their source chunk boundaries may differ from your desired Zarr layout. The generated transform must avoid xarray's unsafe-chunk error by doing one of the following correctly:
- omit explicit Zarr chunks and preserve a valid Dask-derived layout; or
- rechunk output arrays to an explicit grid and use exactly matching Zarr chunks.

Do not merely test small eager NumPy arrays. Include a Dask-backed synthetic test with intentionally different source chunks, then write and reopen the Zarr output.

Choose one or multiple Python modules based on correctness. The evaluator executes the pipeline from its output directory with `python run_pipeline.py`. Include that no-argument root entrypoint, a root `requirements.txt`, secret-safe credential behavior, offline tests, and read-back validation. The entrypoint must run retrieval through validated persisted output, be safe to rerun after partial failure, and exit nonzero on failure. Do not force the implementation into one file. Group CDS fields only when grouping cannot widen requested selector combinations.

The result will be compiled, tested, and run unchanged against existing real ERA5 files. Optimize for that evaluation rather than feature count.

PipelineGenerationResult JSON schema:
{response_schema_json}
