Create a DatasetInventory for this user-provided DatasetCandidate.

Dataset slug:
{dataset_slug}

DatasetCandidate JSON:
{candidate_json}

Output requirements:
- Return only JSON matching the provided schema.
- Store available provider options under options, such as variables, levels, bands, tables, formats, product types, times, or request dimensions.
- Preserve provider-native option names where possible.
- Do not choose a small workflow slice here.
- Include evidence URLs for the docs or metadata used.
- Set extractor_name to "generic_web_inventory_agent" and extraction_method to "llm".

DatasetInventory JSON schema:
{schema_json}
