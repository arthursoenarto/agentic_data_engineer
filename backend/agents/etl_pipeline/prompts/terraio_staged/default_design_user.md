Stage: DESIGN

Design the implementation before writing code. Use the reference source only where it provides concrete, relevant scientific data engineering lessons.

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

Curated reference files:
{reference_context_files_json}

Reference context size:
{reference_context_size}

Curated read-only TerraIO context:
{reference_context}

Produce a compact, implementation-ready design. Resolve:
- the exact provider request for each selected field-selector combination;
- credential discovery using only AccessContext references;
- raw acquisition, validation, and resumable reuse boundaries;
- transformations, dimensions, coordinate conventions, channel ordering, and final storage;
- bounded-memory chunking and incremental or partitioned writes;
- atomic publication and read-back validation;
- a minimal standalone file layout with root `run_pipeline.py` and `requirements.txt`;
- offline tests for request construction, selector preservation, transformations, and validation.

Record only reference invariants that materially improve this contract. Do not copy TerraIO wholesale, require it as a dependency, or add ML feature/target concepts that are absent from the acquisition contract.
