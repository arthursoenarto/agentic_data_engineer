Design a reusable adapter for pipeline `{pipeline_id}` under `{output_dir}`.

Seed contract:
{contract_json}

Frozen inventory:
{inventory_json}

Access context:
{access_context_json}

Fixed policy:
{policy_json}

Framework interface:
{execution_interface_json}

The design must cover runtime inventory validation before provider access, generic request construction, exact field-selector output assembly, read-back validation, semantic cache partitions, exact-rerun reuse, partial temporal extension, temporary writes, and atomic publication. Propose root `pipeline_impl.py`, dependencies, focused offline tests, and supporting modules. Do not propose framework-owned files.

