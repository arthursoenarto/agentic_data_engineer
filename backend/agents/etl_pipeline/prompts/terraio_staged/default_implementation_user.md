Stage: IMPLEMENTATION

Create a standalone pipeline from the distilled design. The raw TerraIO source is intentionally not provided at this stage; the design contains the selected reference lessons.

Dataset directory:
{dataset_dir}

Pipeline output directory:
{output_dir}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Reference-informed internal design:
{design_json}

Return every required source, configuration, documentation, and test file under `{output_dir}/`.

Requirements:
- Include root `run_pipeline.py` and complete `requirements.txt`.
- `python run_pipeline.py` with no arguments must perform retrieval, transformation, storage, reopening, and validation.
- Load allowed `.env` files without overriding explicit shell values, and use only AccessContext credential names and aliases.
- Keep provider requests explicit and preserve exact field-selector pairs.
- Validate raw download completion before reuse.
- Bound memory use; align target chunks and writes with actual source behavior.
- Publish derived output atomically only after successful read-back validation.
- Include all third-party imports in requirements and useful offline tests.
- Do not import TerraIO or this repository's generation/evaluation modules at runtime.

The model-provided manifest fields are provisional and will be replaced by the framework.
