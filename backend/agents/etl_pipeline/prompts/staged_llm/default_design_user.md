Stage: DESIGN

Design the implementation before writing code.

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

Produce a compact, implementation-ready design. Resolve:
- the exact provider request for every selected field-selector combination;
- credential discovery using only the references in AccessContext;
- raw acquisition boundaries and safe reuse of valid existing downloads;
- transformation and storage format based on the scientific data shape;
- chunking or partitioning that avoids loading the full requested scope into memory;
- atomic replacement or another explicit partial-write policy;
- read-back checks for fields, selectors, coordinates, scope, shape, and nonempty values;
- a minimal file layout with a root `run_pipeline.py` and complete `requirements.txt`;
- offline tests for request construction, selector preservation, and validation logic.

Do not add catalogs, schedulers, services, databases, or plugin systems unless the contract actually requires them. State uncertainties explicitly instead of inventing provider behavior.
