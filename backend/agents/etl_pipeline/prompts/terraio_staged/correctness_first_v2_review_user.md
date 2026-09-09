Stage: REVIEW

Review the actual draft; return findings, not rewritten code.

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Internal design:
{design_json}

Generated draft:
{draft_json}

Trace the complete no-argument workflow and inspect actual xarray coordinate behavior.

Request revision for any of these high-severity defects:
- datetime unit reinterpretation or unrealistic temporal tests;
- selected channel arrays retaining conflicting scalar pressure coordinates when 500/850 sources are assembled;
- tests that do not perform the real multi-source merge and final write/reopen;
- a “global” check that accepts an arbitrary small subregion, or rejects a valid coarse cyclic global grid solely because its last longitude is below 360;
- source datasets closing before lazy computation;
- all-fill/non-finite output passing validation;
- invented provenance for sidecar-free raw data;
- selector widening, wrong provider fields, secret exposure, unsafe partial reuse/publication, or missing dependencies.

Also assess maintainability and unnecessary architecture. Accept only if the full source and tests are internally coherent and likely to run unchanged.
