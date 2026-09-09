Implement this frozen TerraIO repository-extension task:

{extension_contract_json}

Frozen source and test context from the contract's exact base commit:

{reference_context}

Return a complete `RepositoryExtensionResult`. Each change must contain the
entire resulting file content, the correct `add` or `modify` operation, and a
short rationale. Put source and documentation changes in `files` and all test
changes in `tests`. The README field should explain the proposed extension,
architecture fit, and deterministic verification commands.
