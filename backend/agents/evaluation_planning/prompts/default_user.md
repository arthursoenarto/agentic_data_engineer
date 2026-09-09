Supported evaluator targets:

1. `regular_rectilinear_grid` -> `zarr` v3 -> `full_field_tensor`
2. `station_time_series` -> `parquet` v1 -> `filtered_station_scan`

Both use engineering profile `general_pipeline`.

Trusted inventory summary:

{inventory_json}

Trusted frozen contract:

{contract_json}

Canonical contract field IDs that the mapping must contain exactly:

{expected_field_ids_json}

Existing trusted catalog summary:

{catalog_json}

Evaluator-owned planning constraints and frozen source context:

{planning_context_json}

Return one concise planning draft. `domain_tags` should contain only broad
domain labels such as `earth-science`, not dataset names, providers, formats,
cadences, or variables. Do not repeat existing checks as proposals.
