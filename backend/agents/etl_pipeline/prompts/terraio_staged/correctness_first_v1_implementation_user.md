Stage: IMPLEMENTATION

Create the complete standalone pipeline from the distilled reference-informed design.

Dataset directory:
{dataset_dir}

Pipeline output directory:
{output_dir}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Internal design:
{design_json}

Return all files under `{output_dir}/`, including root `run_pipeline.py`, complete requirements, concise documentation, and offline tests.

Implementation acceptance requirements:
- The no-argument runner performs provider retrieval, transformation, safe publication, reopening, and validation.
- Exact field-selector pairs are explicit and tested.
- Credentials use only AccessContext names/aliases and approved `.env` behavior.
- Raw files are reused only after request identity and scientific validation.
- Datetime conversion handles xarray `datetime64[ns]` arrays without `.tolist()` integer reinterpretation; tests must assert the full expected timestamp sequence through the actual validation function.
- Lazy source datasets remain open until all dependent writes complete and are then closed deterministically.
- Final validation proves exact scope, coordinates/selectors, nonempty shapes, and at least bounded evidence of finite/non-fill values for every selected field.
- Temporary and final artifact behavior is safe under failure and rerun.
- Tests use realistic writer/reader representations and require no network or credentials.
- No TerraIO or repository generation/evaluation import appears in runtime code.

Model-supplied manifest fields are provisional and will be replaced by the framework.
