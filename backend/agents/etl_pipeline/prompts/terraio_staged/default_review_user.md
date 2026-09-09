Stage: REVIEW

Review the generated draft as a senior scientific data engineer. Do not rewrite code in this response.

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Reference-informed design:
{design_json}

Generated draft:
{draft_json}

Trace the actual code from root entrypoint through provider request, acquisition, transformation, write, and read-back validation. Verify that relevant design invariants were implemented rather than merely described.

Check exact selector preservation, provider API correctness, credential aliases, secret safety, dependency completeness, source/target chunk compatibility, bounded memory, partial artifact handling, atomic publication, coordinate/channel semantics, rerun behavior, and offline test quality. Also reject any TerraIO runtime dependency or unnecessary imitation of its broader architecture.

Classify likely runtime failure, wrong data, corruption, or secret exposure as critical/high. Request revision for any critical/high issue or a combination of medium issues that makes the result unreliable. Accept only if the pipeline is likely to run unchanged and satisfy the contract.
