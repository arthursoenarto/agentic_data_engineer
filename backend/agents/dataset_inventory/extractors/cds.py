"""Copernicus Climate and Atmosphere Data Store inventory extractor."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from backend.agents.contract_drafting.schemas import DatasetCandidate
from backend.agents.dataset_inventory.schemas import DatasetInventory, InventoryEvidence


COPERNICUS_DATA_STORE_HOSTS = frozenset(
    {
        "ads.atmosphere.copernicus.eu",
        "cds.climate.copernicus.eu",
    }
)
CONSTRAINT_KEYS = (
    "type",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "format",
    "pattern",
)


class CDSProcessMetadataExtractor:
    """Extract CDS/ADS options from official Retrieve API process metadata."""

    name = "cds_process_metadata"
    extraction_method = "deterministic"

    def can_handle(self, candidate: DatasetCandidate) -> bool:
        parsed = urlparse(str(candidate.url))
        return parsed.netloc in COPERNICUS_DATA_STORE_HOSTS and "/datasets/" in parsed.path

    def extract(self, candidate: DatasetCandidate, *, dataset_slug: str) -> DatasetInventory:
        dataset_id = dataset_id_from_cds_url(str(candidate.url))
        parsed = urlparse(str(candidate.url))
        api_root = f"{parsed.scheme}://{parsed.netloc}/api"
        metadata_url = f"{api_root}/retrieve/v1/processes/{dataset_id}"
        catalogue_url = f"{api_root}/catalogue/v1/collections/{dataset_id}"
        process = fetch_cds_process_metadata(metadata_url)
        catalogue = fetch_cds_catalogue_metadata(catalogue_url)
        warnings: list[str] = []

        form_url = catalogue_link_url(catalogue, "form")
        constraints_url = catalogue_link_url(catalogue, "constraints")
        layout_url = catalogue_link_url(catalogue, "layout")
        form = fetch_optional_json(form_url, warnings=warnings, label="CDS form metadata")
        availability_rules = fetch_optional_json(constraints_url, warnings=warnings, label="CDS availability constraints")
        layout = fetch_optional_json(layout_url, warnings=warnings, label="CDS layout metadata")

        raw_inputs = process.get("inputs", {})
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        layout_tables = extract_layout_tables(layout)
        input_fields: dict[str, Any] = {}
        options: dict[str, Any] = {}
        option_metadata: dict[str, Any] = {}
        defaults: dict[str, Any] = {}
        constraints: dict[str, Any] = {}

        for input_name, input_spec in inputs.items():
            schema = input_schema(input_spec)
            input_fields[input_name] = input_field_summary(input_spec)

            values = enum_values(input_spec)
            if values:
                options[input_name] = values

            if "default" in schema:
                defaults[input_name] = schema["default"]

            field_constraints = schema_constraints(schema)
            if field_constraints:
                constraints[input_name] = field_constraints

        merge_form_metadata(
            form,
            input_fields=input_fields,
            options=options,
            option_metadata=option_metadata,
            defaults=defaults,
        )
        merge_layout_metadata(layout_tables, option_metadata=option_metadata)

        option_units: dict[str, str] = {}
        if "pressure_level" in options:
            option_units["pressure_level"] = "hPa"
        if "time" in options:
            option_units["time"] = "UTC"
        if "area" in inputs:
            option_units["area"] = "north/west/south/east degrees"

        return DatasetInventory(
            dataset_slug=dataset_slug,
            title=text_or_none(catalogue.get("title")) or text_or_none(process.get("title")) or candidate.name,
            description=text_or_none(catalogue.get("description")) or text_or_none(process.get("description")),
            source_url=candidate.url,
            source_metadata_url=metadata_url,
            generated_at=datetime.now(UTC).isoformat(),
            provider=catalogue_provider(catalogue) or "Copernicus Climate Data Store",
            dataset_id=dataset_id,
            process_version=text_or_none(process.get("version")),
            extractor_name=self.name,
            extraction_method=self.extraction_method,
            note="Extracted from official CDS process, catalogue, form, and layout metadata without LLM calls.",
            catalogue_metadata=catalogue_summary(
                catalogue,
                catalogue_url=catalogue_url,
                form_url=form_url,
                constraints_url=constraints_url,
                layout_url=layout_url,
                layout_tables=layout_tables,
            ),
            request_fields=list(inputs.keys()),
            input_fields=input_fields,
            options=options,
            option_metadata=option_metadata,
            option_units=option_units,
            defaults=defaults,
            constraints=constraints,
            availability=availability_summary(availability_rules, constraints_url),
            outputs=dict_or_empty(process.get("outputs")),
            links=merge_links(process.get("links"), catalogue.get("links")),
            job_control_options=str_list(process.get("jobControlOptions")),
            output_transmission=str_list(process.get("outputTransmission")),
            evidence=[
                InventoryEvidence(
                    title="CDS Retrieve API process metadata",
                    url=metadata_url,
                    note="Machine-readable official request schema used to extract inventory options.",
                ),
                InventoryEvidence(
                    title="CDS catalogue metadata",
                    url=catalogue_url,
                    note="Official STAC collection metadata used for licence, extent, DOI, links, and provider details.",
                ),
                *optional_evidence("CDS form metadata", form_url, "Official web-form metadata used for labels, defaults, and field help."),
                *optional_evidence(
                    "CDS availability constraints",
                    constraints_url,
                    "Official availability constraints summarized without storing every repeated rule inline.",
                ),
                *optional_evidence("CDS layout metadata", layout_url, "Official dataset page layout used for data-description and variable metadata tables."),
            ],
            warnings=warnings,
        )


def dataset_id_from_cds_url(url: str) -> str:
    """Extract the CDS dataset id from a Climate Data Store dataset URL."""

    match = re.search(r"/datasets/([^/?#]+)", urlparse(url).path)
    if not match:
        raise ValueError(f"Could not find CDS dataset id in URL: {url}")
    return match.group(1)


def fetch_cds_process_metadata(metadata_url: str) -> dict[str, Any]:
    """Fetch official CDS process metadata without requiring CDS credentials."""

    request = urllib.request.Request(metadata_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CDS metadata request failed with HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"CDS metadata request failed: {error}") from error


def fetch_cds_catalogue_metadata(catalogue_url: str) -> dict[str, Any]:
    """Fetch official CDS catalogue metadata without requiring CDS credentials."""

    request = urllib.request.Request(catalogue_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CDS catalogue request failed with HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"CDS catalogue request failed: {error}") from error


def fetch_optional_json(url: str | None, *, warnings: list[str], label: str) -> Any:
    """Fetch optional CDS JSON metadata and preserve a warning on failure."""

    if not url:
        return None

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError) as error:
        warnings.append(f"{label} could not be fetched from {url}: {error}")
        return None


def catalogue_link_url(catalogue: dict[str, Any], rel: str) -> str | None:
    """Return a link href from a CDS catalogue collection by relation."""

    links = catalogue.get("links", [])
    if not isinstance(links, list):
        return None
    for link in links:
        if isinstance(link, dict) and link.get("rel") == rel and isinstance(link.get("href"), str):
            return link["href"]
    return None


def enum_values(input_spec: dict[str, Any]) -> list[str]:
    """Extract enum values from a CDS input schema."""

    schema = input_schema(input_spec)
    if isinstance(schema.get("items"), dict) and isinstance(schema["items"].get("enum"), list):
        return [str(value) for value in schema["items"]["enum"]]
    if isinstance(schema.get("enum"), list):
        return [str(value) for value in schema["enum"]]
    return []


def merge_form_metadata(
    form: Any,
    *,
    input_fields: dict[str, Any],
    options: dict[str, Any],
    option_metadata: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    """Merge official CDS web-form metadata into normalized inventory fields."""

    if not isinstance(form, list):
        return

    for field in form:
        if not isinstance(field, dict):
            continue

        name = field.get("name") or field.get("id")
        if not isinstance(name, str) or not name:
            continue
        if name not in input_fields:
            continue

        field_summary = input_fields.setdefault(name, {})
        for source_key, target_key in (("label", "label"), ("help", "help"), ("type", "widget_type")):
            value = field.get(source_key)
            if isinstance(value, str) and value:
                field_summary[target_key] = value

        if isinstance(field.get("required"), bool):
            field_summary["required"] = field["required"]

        details = field.get("details", {})
        if not isinstance(details, dict):
            continue

        values = details.get("values")
        if isinstance(values, list) and values and name not in options:
            options[name] = [str(value) for value in values]
        if isinstance(values, list):
            field_summary["form_option_count"] = len(values)

        if "default" in details and name not in defaults:
            defaults[name] = details["default"]
            field_summary.setdefault("default", details["default"])

        labels = details.get("labels")
        if isinstance(labels, dict):
            for value, label in labels.items():
                if isinstance(label, str):
                    metadata = option_metadata.setdefault(name, {}).setdefault(str(value), {})
                    metadata["label"] = label


def merge_layout_metadata(layout_tables: dict[str, Any], *, option_metadata: dict[str, Any]) -> None:
    """Merge useful official CDS layout tables into option metadata."""

    variable_rows = layout_tables.get("main_variables")
    if not isinstance(variable_rows, list):
        return

    variable_metadata = option_metadata.setdefault("variable", {})
    label_to_option: dict[str, str] = {}
    for option, metadata in variable_metadata.items():
        if isinstance(metadata, dict) and isinstance(metadata.get("label"), str):
            label_to_option[normalize_label(metadata["label"])] = option

    for row in variable_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            continue
        option = label_to_option.get(normalize_label(row["name"]))
        if not option:
            continue
        metadata = variable_metadata.setdefault(option, {})
        metadata.setdefault("label", row["name"])
        for key in ("units", "description"):
            if isinstance(row.get(key), str) and row[key]:
                metadata[key] = row[key]


def input_schema(input_spec: dict[str, Any]) -> dict[str, Any]:
    """Return a CDS input schema if present."""

    schema = input_spec.get("schema", {})
    return schema if isinstance(schema, dict) else {}


def input_field_summary(input_spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize human-readable metadata for one CDS request field."""

    schema = input_schema(input_spec)
    summary: dict[str, Any] = {}
    for key in ("title", "description"):
        if isinstance(input_spec.get(key), str):
            summary[key] = input_spec[key]

    compact_schema = schema_constraints(schema)
    if compact_schema:
        summary["schema"] = compact_schema

    values = enum_values(input_spec)
    if values:
        summary["option_count"] = len(values)

    if "default" in schema:
        summary["default"] = schema["default"]

    return summary


def schema_constraints(schema: dict[str, Any]) -> dict[str, Any]:
    """Preserve useful non-enum request constraints from a CDS JSON schema."""

    constraints: dict[str, Any] = {}
    for key in CONSTRAINT_KEYS:
        if key not in schema:
            continue

        value = schema[key]
        if key == "items" and isinstance(value, dict):
            item_constraints = {item_key: item_value for item_key, item_value in value.items() if item_key != "enum"}
            if item_constraints:
                constraints[key] = item_constraints
        else:
            constraints[key] = value

    return constraints


def extract_layout_tables(layout: Any) -> dict[str, Any]:
    """Extract table content from the official CDS page-layout JSON."""

    tables: dict[str, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "table" and isinstance(value.get("id"), str):
                tables[value["id"]] = value.get("content", [])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(layout)
    return tables


def catalogue_summary(
    catalogue: dict[str, Any],
    *,
    catalogue_url: str,
    form_url: str | None,
    constraints_url: str | None,
    layout_url: str | None,
    layout_tables: dict[str, Any],
) -> dict[str, Any]:
    """Keep useful official STAC collection metadata without storing a raw dump."""

    summary: dict[str, Any] = {}
    simple_keys = {
        "stac_version": "stac_version",
        "license": "license",
        "published": "published",
        "updated": "updated",
        "sci:doi": "doi",
        "cads:update_frequency": "update_frequency",
        "cads:fair": "fair_score",
    }
    for source_key, target_key in simple_keys.items():
        if catalogue.get(source_key) is not None:
            summary[target_key] = catalogue[source_key]

    for key in ("keywords", "extent", "providers", "assets", "summaries"):
        value = catalogue.get(key)
        if value:
            summary[key] = value

    data_description = layout_tables.get("data_description")
    if isinstance(data_description, list) and data_description:
        summary["data_description"] = data_description[0]

    metadata_links = {
        "catalogue": catalogue_url,
        "form": form_url,
        "constraints": constraints_url,
        "layout": layout_url,
    }
    summary["metadata_links"] = {key: value for key, value in metadata_links.items() if value}
    return summary


def availability_summary(availability_rules: Any, constraints_url: str | None) -> dict[str, Any]:
    """Summarize CDS availability rules without duplicating every repeated rule."""

    if not isinstance(availability_rules, list):
        return {"constraints_url": constraints_url} if constraints_url else {}

    fields: set[str] = set()
    for rule in availability_rules:
        if isinstance(rule, dict):
            fields.update(str(key) for key in rule)

    return {
        "constraint_rule_count": len(availability_rules),
        "constraint_fields": sorted(fields),
        "constraints_url": constraints_url,
    }


def catalogue_provider(catalogue: dict[str, Any]) -> str | None:
    """Return a concise provider name from STAC provider metadata."""

    providers = catalogue.get("providers", [])
    if not isinstance(providers, list):
        return None
    names = [provider.get("name") for provider in providers if isinstance(provider, dict) and isinstance(provider.get("name"), str)]
    return ", ".join(names) if names else None


def merge_links(*groups: Any) -> list[dict[str, Any]]:
    """Merge official link arrays while avoiding duplicate rel/href pairs."""

    links: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for link in group:
            if not isinstance(link, dict):
                continue
            key = (text_or_none(link.get("rel")), text_or_none(link.get("href")))
            if key in seen:
                continue
            seen.add(key)
            links.append(link)
    return links


def optional_evidence(title: str, url: str | None, note: str) -> list[InventoryEvidence]:
    """Build evidence only when an optional metadata URL is known."""

    if not url:
        return []
    return [InventoryEvidence(title=title, url=url, note=note)]


def dict_or_empty(value: Any) -> dict[str, Any]:
    """Return a dict when the value is a dict, else an empty mapping."""

    return value if isinstance(value, dict) else {}


def str_list(value: Any) -> list[str]:
    """Return a list of strings from provider metadata."""

    return [str(item) for item in value] if isinstance(value, list) else []


def text_or_none(value: Any) -> str | None:
    """Return a non-empty string or None."""

    return value if isinstance(value, str) and value else None


def normalize_label(value: str) -> str:
    """Normalize labels for matching form options to layout table rows."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())
