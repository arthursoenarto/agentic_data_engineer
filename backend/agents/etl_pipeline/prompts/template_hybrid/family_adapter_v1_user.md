Fill the reusable adapter slots for pipeline `{pipeline_id}` under `{output_dir}`.

Seed contract:
{contract_json}

Frozen inventory:
{inventory_json}

Secret-free access context:
{access_context_json}

Fixed policy:
{policy_json}

Framework interface:
{execution_interface_json}

The framework passes validated path arguments and publishes the staging output atomically. The implementation remains responsible for provider-specific validation, request construction, decoding, cache identity/content validation, exact-rerun reuse, partial temporal acquisition, output assembly, and read-back checks. Return hit/miss/acquisition evidence without secrets.

