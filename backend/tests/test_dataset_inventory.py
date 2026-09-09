from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agents.contract_drafting import DatasetCandidate
from backend.agents.dataset_inventory.extractors.cds import CDSProcessMetadataExtractor, dataset_id_from_cds_url
from backend.agents.dataset_inventory.workflow import build_dataset_inventory


CDS_PROCESS_PAYLOAD = {
    "title": "ERA5 hourly data on pressure levels from 1940 to present",
    "description": "ERA5 pressure-level process metadata.",
    "id": "reanalysis-era5-pressure-levels",
    "version": "1.0.0",
    "jobControlOptions": ["async-execute"],
    "outputTransmission": ["reference"],
    "links": [
        {
            "href": "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-pressure-levels",
            "rel": "self",
        }
    ],
    "outputs": {
        "asset": {
            "title": "Asset",
            "description": "Downloadable asset description",
        }
    },
    "inputs": {
        "product_type": {
            "title": "Product type",
            "schema": {"type": "array", "items": {"type": "string", "enum": ["reanalysis", "ensemble_mean"]}},
        },
        "variable": {
            "title": "Variable",
            "schema": {"type": "array", "items": {"type": "string", "enum": ["temperature", "geopotential"]}},
        },
        "pressure_level": {
            "title": "Pressure level",
            "schema": {"type": "array", "items": {"type": "string", "enum": ["500", "850"]}},
        },
        "data_format": {"title": "Data format", "schema": {"type": "string", "enum": ["grib", "netcdf"], "default": "grib"}},
        "download_format": {
            "title": "Download format",
            "schema": {"type": "string", "enum": ["zip", "unarchived"], "default": "unarchived"},
        },
        "area": {
            "title": "Sub-region extraction",
            "schema": {
                "type": "array",
                "default": [90, -180, -90, 180],
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
        },
    },
}

CDS_CATALOGUE_PAYLOAD = {
    "title": "ERA5 hourly data on pressure levels from 1940 to present",
    "description": "ERA5 hourly pressure-level reanalysis data.",
    "stac_version": "1.1.0",
    "license": "CC-BY-4.0",
    "published": "2018-06-14T00:00:00Z",
    "updated": "2026-07-06T00:00:00Z",
    "sci:doi": "10.24381/cds.bd0915c6",
    "cads:update_frequency": "Daily",
    "cads:fair": 92,
    "keywords": ["Spatial coverage: Global"],
    "extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2026-06-30T00:00:00+00:00"]]}},
    "providers": [{"name": "ECMWF"}],
    "assets": {"thumbnail": {"href": "https://example.com/thumb.png"}},
    "links": [
        {"href": "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-pressure-levels", "rel": "self"},
        {"href": "https://example.com/form.json", "rel": "form"},
        {"href": "https://example.com/constraints.json", "rel": "constraints"},
        {"href": "https://example.com/layout.json", "rel": "layout"},
    ],
}

CDS_FORM_PAYLOAD = [
    {
        "name": "product_type",
        "label": "Product type",
        "required": True,
        "type": "StringListWidget",
        "details": {
            "values": ["reanalysis", "ensemble_mean"],
            "labels": {"reanalysis": "Reanalysis", "ensemble_mean": "Ensemble mean"},
            "default": ["reanalysis"],
        },
    },
    {
        "name": "variable",
        "label": "Variable",
        "help": "Choose pressure-level variables.",
        "required": True,
        "type": "StringListWidget",
        "details": {
            "values": ["temperature", "geopotential"],
            "labels": {"temperature": "Temperature", "geopotential": "Geopotential"},
        },
    },
    {
        "name": "licences",
        "label": "Licence acknowledgement",
        "required": True,
        "type": "LicenceWidget",
        "details": {"default": ["accepted"]},
    },
    {
        "name": "area",
        "label": "Sub-region extraction",
        "type": "GeographicExtentWidget",
        "details": {"values": []},
    },
]

CDS_LAYOUT_PAYLOAD = {
    "body": {
        "main": {
            "sections": [
                {
                    "blocks": [
                        {
                            "type": "table",
                            "id": "data_description",
                            "content": [{"temporal_resolution": "Hourly", "vertical_resolution": "37 pressure levels"}],
                        },
                        {
                            "type": "table",
                            "id": "main_variables",
                            "content": [
                                {"name": "Temperature", "units": "K", "description": "Air temperature."},
                                {"name": "Geopotential", "units": "m2 s-2", "description": "Gravitational potential energy."},
                            ],
                        },
                    ]
                }
            ]
        }
    }
}

CDS_CONSTRAINTS_PAYLOAD = [
    {"year": ["2024"], "month": ["01"], "day": ["01"], "time": ["00:00"], "variable": ["temperature"]},
    {"year": ["2024"], "month": ["01"], "day": ["02"], "time": ["00:00"], "variable": ["geopotential"]},
]


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def fake_urlopen(request: object, timeout: int = 30) -> FakeResponse:
    url = getattr(request, "full_url", str(request))
    if "api/retrieve/v1/processes" in url:
        return FakeResponse(CDS_PROCESS_PAYLOAD)
    if "api/catalogue/v1/collections" in url:
        return FakeResponse(CDS_CATALOGUE_PAYLOAD)
    if url == "https://example.com/form.json":
        return FakeResponse(CDS_FORM_PAYLOAD)
    if url == "https://example.com/layout.json":
        return FakeResponse(CDS_LAYOUT_PAYLOAD)
    if url == "https://example.com/constraints.json":
        return FakeResponse(CDS_CONSTRAINTS_PAYLOAD)
    raise AssertionError(f"Unexpected URL: {url}")


class DatasetInventoryTests(unittest.TestCase):
    def test_dataset_id_from_cds_url(self) -> None:
        self.assertEqual(
            dataset_id_from_cds_url("https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download"),
            "reanalysis-era5-pressure-levels",
        )

    def test_cds_extractor_reads_official_enum_options(self) -> None:
        candidate = DatasetCandidate(
            name="ERA5 pressure levels",
            slug="era5_pressure",
            url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
        )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            inventory = CDSProcessMetadataExtractor().extract(candidate, dataset_slug="era5_pressure")

        self.assertEqual(inventory.extraction_method, "deterministic")
        self.assertEqual(inventory.dataset_id, "reanalysis-era5-pressure-levels")
        self.assertEqual(inventory.description, "ERA5 hourly pressure-level reanalysis data.")
        self.assertEqual(inventory.provider, "ECMWF")
        self.assertEqual(inventory.process_version, "1.0.0")
        self.assertEqual(inventory.job_control_options, ["async-execute"])
        self.assertEqual(inventory.output_transmission, ["reference"])
        self.assertEqual(inventory.links[0]["rel"], "self")
        self.assertEqual(inventory.outputs["asset"]["title"], "Asset")
        self.assertEqual(inventory.catalogue_metadata["license"], "CC-BY-4.0")
        self.assertEqual(inventory.catalogue_metadata["doi"], "10.24381/cds.bd0915c6")
        self.assertEqual(inventory.catalogue_metadata["data_description"]["temporal_resolution"], "Hourly")
        self.assertEqual(inventory.availability["constraint_rule_count"], 2)
        self.assertEqual(inventory.availability["constraint_fields"], ["day", "month", "time", "variable", "year"])
        self.assertEqual(inventory.options["variable"], ["temperature", "geopotential"])
        self.assertEqual(inventory.options["pressure_level"], ["500", "850"])
        self.assertEqual(inventory.options["data_format"], ["grib", "netcdf"])
        self.assertNotIn("area", inventory.options)
        self.assertEqual(inventory.option_units["pressure_level"], "hPa")
        self.assertEqual(inventory.option_units["area"], "north/west/south/east degrees")
        self.assertEqual(inventory.defaults["product_type"], ["reanalysis"])
        self.assertEqual(inventory.defaults["area"], [90, -180, -90, 180])
        self.assertEqual(inventory.defaults["data_format"], "grib")
        self.assertEqual(inventory.constraints["area"]["minItems"], 4)
        self.assertEqual(inventory.constraints["area"]["items"], {"type": "number"})
        self.assertEqual(inventory.input_fields["area"]["title"], "Sub-region extraction")
        self.assertEqual(inventory.input_fields["variable"]["help"], "Choose pressure-level variables.")
        self.assertNotIn("licences", inventory.input_fields)
        self.assertEqual(inventory.input_fields["variable"]["option_count"], 2)
        self.assertEqual(inventory.option_metadata["variable"]["temperature"]["label"], "Temperature")
        self.assertEqual(inventory.option_metadata["variable"]["temperature"]["units"], "K")
        self.assertEqual(inventory.option_metadata["variable"]["temperature"]["description"], "Air temperature.")

    def test_ads_extractor_uses_matching_ads_api(self) -> None:
        candidate = DatasetCandidate(
            name="CAMS global reanalysis",
            slug="cams_eac4",
            url="https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4?tab=download",
        )
        extractor = CDSProcessMetadataExtractor()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen) as urlopen:
            inventory = extractor.extract(candidate, dataset_slug="cams_eac4")

        requested_urls = [call.args[0].full_url for call in urlopen.call_args_list]
        self.assertTrue(extractor.can_handle(candidate))
        self.assertEqual(inventory.dataset_id, "cams-global-reanalysis-eac4")
        self.assertIn(
            "https://ads.atmosphere.copernicus.eu/api/retrieve/v1/processes/cams-global-reanalysis-eac4",
            requested_urls,
        )
        self.assertIn(
            "https://ads.atmosphere.copernicus.eu/api/catalogue/v1/collections/cams-global-reanalysis-eac4",
            requested_urls,
        )

    def test_workflow_writes_cds_inventory_artifact(self) -> None:
        candidate = DatasetCandidate(
            name="ERA5 pressure levels",
            slug="era5_pressure",
            url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
        )

        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen", side_effect=fake_urlopen):
            inventory, path = build_dataset_inventory(candidate, datasets_dir=Path(tmp), root=Path(tmp))

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(inventory.options["pressure_level"], ["500", "850"])
        self.assertEqual(path.name, "dataset_inventory.json")
        self.assertEqual(payload["extractor_name"], "cds_process_metadata")

    def test_workflow_uses_minimal_inventory_when_no_extractor_matches(self) -> None:
        candidate = DatasetCandidate(name="OpenAQ", slug="openaq", url="https://docs.openaq.org/")

        with tempfile.TemporaryDirectory() as tmp:
            inventory, path = build_dataset_inventory(candidate, datasets_dir=Path(tmp), root=Path(tmp))
            self.assertTrue(path.exists())

        self.assertEqual(inventory.extraction_method, "manual")
        self.assertEqual(inventory.extractor_name, "minimal_manual_inventory")


if __name__ == "__main__":
    unittest.main()
