Create a DatasetContract draft for this user-provided DatasetCandidate.

DatasetCandidate JSON:
{candidate_json}

DatasetInventory JSON:
{inventory_json}

Output requirements:
- Return only JSON matching the provided schema.
- Draft a concise intent from user_goal, description, name, and source docs.
- The scope must be small and directly editable by a scientist or ML practitioner.
- Include only minimal user-facing decisions. Examples: selected fields, selector values, date range, timestep, geography, station/location, aggregation, table/file selection.
- If the user goal gives concrete selections, copy those selections into the contract instead of replacing them with generic defaults.
- Include selected fields only. Do not enumerate the full dataset variable inventory in the contract.
- Do not include feature/target/input/output roles. If the user says "feature" or "target", translate that into selected fields only.
- For datasets with selectors, each selected field-selector combination should be a separate field.
- Use DatasetInventory options to validate selected fields, selectors, times, product types, and formats when it is not null.
- Include provider-specific request details only if they are necessary for the user to choose the scope.
- Include evidence URLs for the docs used.
- Set human_confirmed to false.

DatasetContract JSON schema:
{schema_json}
