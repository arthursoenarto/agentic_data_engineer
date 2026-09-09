Create a reusable dataset-family adapter.

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

Selected read-only reference paths:
{reference_context_files_json}

Rendered reference context:
{reference_context}

The seed is only a development case. Preserve exact field-selector channels, support other valid inventory selections, and partition cache coverage so exact reruns acquire nothing and temporal extensions acquire only missing partitions.

Response schema:
{response_schema_json}

