Stage: REVISION

Return a complete corrected standalone pipeline, not a patch or commentary.

Pipeline output directory:
{output_dir}

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Reference-informed design:
{design_json}

Original draft:
{draft_json}

Review:
{review_json}

Fix every required issue and recheck the full no-argument workflow. Preserve relevant reference invariants without introducing a TerraIO runtime dependency. Ensure provider-correct requests, exact selectors, safe credentials, validated resumability, bounded-memory transformation, atomic output publication, read-back checks, complete requirements, and offline tests. All paths must remain under `{output_dir}/`.
