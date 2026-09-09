Stage: IMPLEMENTATION

Create the complete standalone pipeline from the internal design.

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

Return all source, documentation, requirements, and offline tests under `{output_dir}/`.

Acceptance requirements:
- Root `run_pipeline.py` performs retrieval, transformation, safe publication, reopening, and validation without arguments.
- Provider requests preserve exactly temperature@500, temperature@850, and geopotential@500 for this contract.
- Credentials use only AccessContext references and never leak.
- Raw reuse requires validated request identity; sidecar-free files are not silently assigned provenance.
- Datetime handling works for xarray-decoded `datetime64[ns]` arrays without `.tolist()` unit corruption.
- Pressure selection removes conflicting scalar `pressure_level` coordinates before combining 500 and 850 hPa channels, while retaining selector provenance per output field.
- Lazy source files remain open until write completion.
- Global validation proves latitude endpoints and cyclic longitude coverage based on ordered spacing, including coarse but complete regular grids.
- Every output field has bounded finite/non-fill data evidence.
- Writes are bounded and failure-safe; the published artifact is reopened and validated.
- Offline tests execute the same two-source merge and writer/reader path as runtime, test global versus subregional coordinates, and run without network or credentials.
- Include every imported third-party package. Do not import TerraIO or repository generation/evaluation modules.

Model-provided manifest fields are provisional.
