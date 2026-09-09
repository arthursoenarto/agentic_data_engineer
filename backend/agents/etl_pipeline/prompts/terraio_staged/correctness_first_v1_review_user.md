Stage: REVIEW

Review the actual draft as a senior scientific data engineer. Return findings only, not rewritten code.

DatasetContract:
{contract_json}

AccessContext:
{access_context_json}

Internal design:
{design_json}

Generated draft:
{draft_json}

Trace the no-argument path through request construction, acquisition, raw validation, transformation, publication, and final validation.

Treat these as high-severity when present:
- datetime code that converts `datetime64[ns]` through `.tolist()` or otherwise reinterprets integer epoch values with the wrong unit;
- validation tests that pass a different temporal representation than xarray runtime data;
- lazy arrays whose source datasets may close or be garbage-collected before compute/write;
- final validation that accepts all-fill, all-NaN, or all-infinite selected fields;
- wrong provider parameters, selector widening, unsafe credentials, missing requirements, partial artifact reuse, or destructive publication.

Also assess memory bounds, coordinate semantics, rerun behavior, test realism, and needless architecture. Request revision for any critical/high issue or combined medium issues that undermine reliability. Accept only if the code and tests are likely to run unchanged.
