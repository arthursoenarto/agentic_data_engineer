Create a reusable expert-quality dataset-family adapter.

Pipeline ID: `{pipeline_id}`
Output directory: `{output_dir}`
Network probing during generation: `{allow_network_probe}`

Seed DatasetContract:
{contract_json}

Frozen DatasetInventory:
{inventory_json}

Secret-free AccessContext:
{access_context_json}

Fixed pipeline policy:
{policy_json}

Framework interface:
{execution_interface_json}

Reference-context treatment:
{reference_context_block}

The generated adapter must support other valid locks inside this exact frozen
inventory and policy. Keep the implementation compact, but separate genuinely
stateful provider/cache/storage concerns from pure transformation and validation
logic when that improves correctness.
