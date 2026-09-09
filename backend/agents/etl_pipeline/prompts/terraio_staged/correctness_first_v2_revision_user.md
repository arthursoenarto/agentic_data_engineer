Stage: REVISION

Return a complete corrected replacement pipeline, not a patch or commentary.

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

Fix every required issue and recheck the no-argument workflow. Ensure dtype-safe times, removal of conflicting scalar selector coordinates before channel assembly, exact field-selector provenance, realistic two-source merge/write/reopen tests, resolution-aware global coverage, explicit lazy resource lifetime, bounded finite/non-fill checks, trustworthy request identity, provider correctness, safe credentials, failure-safe publication, complete dependencies, and no TerraIO runtime dependency. Keep all paths under `{output_dir}/`.
