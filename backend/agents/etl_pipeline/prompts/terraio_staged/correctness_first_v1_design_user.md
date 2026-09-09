Stage: DESIGN

Design a standalone implementation before writing code. Use the reference only for concrete lessons relevant to this contract.

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

Resolve the exact provider requests, credential lookup, validated raw reuse, bounded transformation, final storage, atomic publication, read-back validation, and a minimal standalone file layout.

The design must explicitly address:
- each exact field-selector pair and how unwanted combinations are prevented;
- xarray/NumPy datetime decoding and dtype-safe conversion to the contract's timestamps;
- source file lifetime through lazy computation and final write;
- checks for finite/non-fill scientific values using bounded reads;
- source chunks, target chunks/partitions, and memory bounds;
- offline acceptance tests using the same datetime dtype and writer/reader path as runtime;
- root `run_pipeline.py`, complete `requirements.txt`, and no TerraIO runtime dependency.

Do not add catalogs, ML feature/target concepts, services, or broad plugin systems.
