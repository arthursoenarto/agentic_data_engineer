Stage: REVISION

Return a complete corrected replacement for the generated pipeline, not a patch or commentary.

Pipeline output directory:
{output_dir}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Internal design:
{design_json}

Original draft:
{draft_json}

Review:
{review_json}

Fix every required review issue while preserving correct behavior. Recheck the complete entrypoint, dependency list, provider request, selector mapping, secret handling, rerun behavior, bounded-memory transformation, atomic output publication, read-back validation, and offline tests. All returned paths must remain under `{output_dir}/`.
