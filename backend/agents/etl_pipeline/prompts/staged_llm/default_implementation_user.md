Stage: IMPLEMENTATION

Create the complete pipeline from the approved internal design.

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

Return every required source, configuration, documentation, and test file. All paths must be under `{output_dir}/`.

Implementation requirements:
- Include root `run_pipeline.py` and `requirements.txt`.
- Running `python run_pipeline.py` with no arguments must retrieve, transform, store, reopen, and validate the requested artifact.
- Load `.env` without overriding explicit shell variables, searching the pipeline and relevant ancestor directories when AccessContext permits it.
- Use only credential names and aliases declared by AccessContext.
- Keep provider request construction explicit and testable.
- Preserve selected field-selector pairs exactly.
- Reuse only validated raw downloads; never mistake a partial file for a completed download.
- Write derived output through a temporary location and publish it only after read-back validation, or implement an equivalently safe policy.
- Bound memory use and make chunk choices compatible with the source representation and target writer.
- Include all imported third-party packages in `requirements.txt`.
- Include meaningful offline tests that do not require credentials or network access.
- Do not import this repository's generation or evaluation packages at runtime.

The manifest content returned by the model is provisional. Framework-owned provenance fields will be replaced after generation.
