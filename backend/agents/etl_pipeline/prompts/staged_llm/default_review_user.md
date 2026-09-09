Stage: REVIEW

Review the generated draft as a senior scientific data engineer. Do not rewrite code in this response.

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Internal design:
{design_json}

Generated draft:
{draft_json}

Trace the actual code path from `run_pipeline.py` through provider request construction, acquisition, transformation, write, and read-back validation.

Check specifically for:
- exact field-selector and time/geography preservation;
- correct provider parameter names, client construction, and credential aliases;
- accidental secret exposure;
- missing imports or requirements and incompatible library APIs;
- full-memory materialization, unsafe source/target chunk interactions, and invalid format writes;
- partial-download reuse, non-atomic output replacement, and rerun corruption;
- validation that only checks existence rather than scientific content;
- tests that cannot run offline or fail to exercise request and selector logic;
- unnecessary architecture that increases failure surface.

Use `critical` or `high` for defects likely to cause runtime failure, wrong data, corruption, or secret exposure. Request revision for any critical/high issue or a combination of medium issues that makes the pipeline unreliable. Accept only if the draft is likely to run unchanged and satisfy the contract.
