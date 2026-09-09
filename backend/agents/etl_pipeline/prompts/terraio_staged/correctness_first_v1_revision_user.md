Stage: REVISION

Return a complete corrected standalone pipeline, not a patch or explanation.

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

Fix every required issue and recheck the complete no-argument workflow. Pay particular attention to dtype-safe datetime normalization, realistic temporal tests, explicit lazy source lifetime, finite/non-fill checks for every selected field, exact selector pairs, provider correctness, secret safety, validated raw reuse, bounded writes, atomic publication, complete dependencies, and offline tests. Preserve relevant reference lessons without adding a TerraIO runtime dependency. Keep every path under `{output_dir}/`.
