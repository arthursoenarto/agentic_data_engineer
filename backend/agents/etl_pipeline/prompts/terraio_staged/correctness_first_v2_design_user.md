Stage: DESIGN

Design a standalone implementation for this exact contract.

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

Produce a compact implementation design covering exact provider requests, credentials, validated raw reuse, bounded transformation, final storage, failure-safe publication, read-back validation, and offline acceptance tests.

Explicitly resolve:
- exact field-selector pairs and request grouping;
- dtype-safe timestamp normalization matching xarray's decoded representation;
- selector extraction from multiple files with different scalar pressure coordinates, including dropping/renaming conflicting selector coordinates before final dataset construction;
- source resource lifetime through lazy computation;
- global latitude and cyclic-longitude coverage at arbitrary regular resolution;
- bounded finite/non-fill checks;
- request identity rules that never invent provenance for sidecar-free data;
- an end-to-end synthetic test that opens separate 500 and 850 hPa files, assembles all three channels, writes, reopens, validates, and reruns;
- root runner, complete requirements, and no TerraIO runtime dependency.

Avoid ML roles, catalogs, services, or plugin systems not required by the contract.
