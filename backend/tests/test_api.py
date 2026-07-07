from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.agents.contract_drafting import DatasetContract
from backend.credentials import AccessProbeResult
from backend.api.main import app


class ApiTests(unittest.TestCase):
    def test_health(self) -> None:
        client = TestClient(app)

        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_create_dataset_contract_rejects_bad_url(self) -> None:
        client = TestClient(app)

        response = client.post("/api/dataset-contracts", json={"name": "Bad", "url": "not-a-url"})

        self.assertEqual(response.status_code, 422)

    def test_create_dataset_contract_writes_project_artifacts_with_mocked_agent(self) -> None:
        class FakeAgent:
            def __init__(self, llm, *, prompt_name="default"):  # type: ignore[no-untyped-def]
                self.llm = llm

            def draft_contract(self, candidate, *, inventory=None):  # type: ignore[no-untyped-def]
                return DatasetContract(
                    dataset_slug=candidate.slug or candidate.name.lower(),
                    title=candidate.name,
                    source_url=candidate.url,
                    intent="Explore whether this dataset supports the project goal.",
                    summary="Drafted from official documentation.",
                    access_methods=["REST API"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_dir = root / "project/datasets"

            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
                patch("backend.agents.contract_drafting.workflow.LLMClient", lambda **kwargs: object()),
                patch("backend.agents.contract_drafting.workflow.ContractDraftingAgent", FakeAgent),
            ):
                client = TestClient(app)
                response = client.post(
                    "/api/dataset-contracts",
                    json={"name": "OpenAQ", "url": "https://docs.openaq.org/", "slug": "air_quality"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["schema_version"], "dataset_contract_run.v1")
            self.assertEqual(payload["candidate_file"], "project/datasets/air_quality/candidate.json")
            self.assertEqual(payload["contract_file"], "project/datasets/air_quality/contract.json")
            self.assertEqual(payload["inventory_file"], "project/datasets/air_quality/dataset_inventory.json")
            self.assertEqual(payload["contract"]["title"], "OpenAQ")
            self.assertEqual(payload["candidate"]["slug"], "air_quality")

            candidate_payload = json.loads((datasets_dir / "air_quality/candidate.json").read_text(encoding="utf-8"))
            contract_payload = json.loads((datasets_dir / "air_quality/contract.json").read_text(encoding="utf-8"))

            self.assertEqual(candidate_payload["name"], "OpenAQ")
            self.assertEqual(candidate_payload["slug"], "air_quality")
            self.assertEqual(contract_payload["contract"]["summary"], "Drafted from official documentation.")
            self.assertTrue((datasets_dir / "air_quality/dataset_inventory.json").exists())

    def test_create_dataset_contract_accepts_url_only_input(self) -> None:
        class FakeAgent:
            def __init__(self, llm, *, prompt_name="default"):  # type: ignore[no-untyped-def]
                self.llm = llm

            def draft_contract(self, candidate, *, inventory=None):  # type: ignore[no-untyped-def]
                return DatasetContract(
                    dataset_slug=candidate.slug or candidate.name.lower(),
                    title=candidate.name,
                    source_url=candidate.url,
                    intent="Explore whether this dataset supports the project goal.",
                    summary="Drafted from a URL-only candidate.",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_dir = root / "project/datasets"

            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
                patch("backend.agents.contract_drafting.workflow.LLMClient", lambda **kwargs: object()),
                patch("backend.agents.contract_drafting.workflow.ContractDraftingAgent", FakeAgent),
            ):
                client = TestClient(app)
                response = client.post(
                    "/api/dataset-contracts",
                    json={"url": "https://example.com/datasets/reanalysis-era5-single-levels"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(
                payload["candidate_file"],
                "project/datasets/reanalysis_era5_single_levels/candidate.json",
            )
            self.assertEqual(payload["contract"]["title"], "Reanalysis Era5 Single Levels")
            self.assertEqual(payload["candidate"]["name"], "Reanalysis Era5 Single Levels")

    def test_list_dataset_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_dir = root / "project/datasets"
            openaq_dir = datasets_dir / "openaq"
            openaq_dir.mkdir(parents=True)
            (openaq_dir / "candidate.json").write_text(
                json.dumps({"name": "OpenAQ", "slug": "openaq", "url": "https://docs.openaq.org/"}),
                encoding="utf-8",
            )
            (openaq_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_contract_run.v1",
                        "candidate_file": "project/datasets/openaq/candidate.json",
                        "contract_file": "project/datasets/openaq/contract.json",
                        "generated_at": "2026-06-24T00:00:00+00:00",
                        "contract": {
                            "schema_version": "dataset_contract.v1",
                            "dataset_slug": "openaq",
                            "title": "OpenAQ",
                            "source_url": "https://docs.openaq.org/",
                            "intent": "Explore recent air quality.",
                            "summary": "Existing contract.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
            ):
                client = TestClient(app)
                response = client.get("/api/dataset-contracts")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["contracts"][0]["contract"]["title"], "OpenAQ")
            self.assertEqual(payload["contracts"][0]["candidate"]["slug"], "openaq")

    def test_dataset_contract_and_inventory_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_dir = root / "project/datasets"
            dataset_dir = datasets_dir / "era5"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "candidate.json").write_text(
                json.dumps({"name": "ERA5", "slug": "era5", "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels"}),
                encoding="utf-8",
            )
            (dataset_dir / "dataset_inventory.json").write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_inventory.v1",
                        "dataset_slug": "era5",
                        "title": "ERA5",
                        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels",
                        "generated_at": "2026-07-06T00:00:00+00:00",
                        "extractor_name": "fixture",
                        "extraction_method": "deterministic",
                        "options": {"variable": ["temperature", "geopotential"], "time": ["00:00", "03:00"]},
                    }
                ),
                encoding="utf-8",
            )
            (dataset_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_contract_run.v1",
                        "candidate_file": "project/datasets/era5/candidate.json",
                        "contract_file": "project/datasets/era5/contract.json",
                        "inventory_file": "project/datasets/era5/dataset_inventory.json",
                        "generated_at": "2026-07-06T00:00:00+00:00",
                        "contract": {
                            "schema_version": "dataset_contract.v1",
                            "dataset_slug": "era5",
                            "title": "ERA5",
                            "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels",
                            "intent": "Explore pressure levels.",
                            "summary": "Existing ERA5 contract.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
            ):
                client = TestClient(app)
                contract_response = client.get("/api/datasets/era5/contract")
                inventory_response = client.get("/api/datasets/era5/inventory")
                save_response = client.put(
                    "/api/datasets/era5/contract",
                    json={
                        "schema_version": "dataset_contract.v1",
                        "dataset_slug": "wrong_slug",
                        "title": "ERA5 edited",
                        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels",
                        "intent": "Explore pressure levels.",
                        "summary": "Edited from the contract UI.",
                    },
                )

            self.assertEqual(contract_response.status_code, 200)
            self.assertEqual(contract_response.json()["contract"]["title"], "ERA5")
            self.assertEqual(inventory_response.status_code, 200)
            self.assertEqual(inventory_response.json()["options"]["variable"], ["temperature", "geopotential"])
            self.assertEqual(save_response.status_code, 200)
            self.assertEqual(save_response.json()["contract"]["dataset_slug"], "era5")
            self.assertEqual(save_response.json()["contract"]["title"], "ERA5 edited")

            saved_payload = json.loads((dataset_dir / "contract.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["contract"]["summary"], "Edited from the contract UI.")

    def test_credential_endpoints_do_not_return_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text("MY_OPENAQ_TOKEN=secret-value\n", encoding="utf-8")
            datasets_dir = root / "project/datasets"
            openaq_dir = datasets_dir / "openaq"
            openaq_dir.mkdir(parents=True)
            (openaq_dir / "candidate.json").write_text(
                json.dumps({"name": "OpenAQ", "slug": "openaq", "url": "https://docs.openaq.org/"}),
                encoding="utf-8",
            )
            (openaq_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_contract_run.v1",
                        "candidate_file": "project/datasets/openaq/candidate.json",
                        "contract_file": "project/datasets/openaq/contract.json",
                        "generated_at": "2026-06-25T00:00:00+00:00",
                        "contract": {
                            "schema_version": "dataset_contract.v1",
                            "dataset_slug": "openaq",
                            "title": "OpenAQ",
                            "source_url": "https://docs.openaq.org/",
                            "intent": "Explore recent air quality.",
                            "summary": "OpenAQ uses an API key with X-API-Key.",
                            "access_methods": ["REST API with X-API-Key"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
                patch("backend.api.main.ENV_PATH", env_path),
            ):
                client = TestClient(app)
                response = client.put(
                    "/api/datasets/openaq/credentials",
                    json={
                        "references": [
                            {
                                "requirement": "OPENAQ_API_KEY",
                                "source": "env",
                                "env_var": "MY_OPENAQ_TOKEN",
                            }
                        ]
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("MY_OPENAQ_TOKEN", response.json()["set_env_vars"])
            self.assertNotIn("secret-value", response.text)

    def test_access_probe_endpoint_returns_probe_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_dir = root / "project/datasets"
            openaq_dir = datasets_dir / "openaq"
            openaq_dir.mkdir(parents=True)
            (openaq_dir / "candidate.json").write_text(
                json.dumps({"name": "OpenAQ", "slug": "openaq", "url": "https://docs.openaq.org/"}),
                encoding="utf-8",
            )
            (openaq_dir / "contract.json").write_text(
                json.dumps(
                    {
                        "schema_version": "dataset_contract_run.v1",
                        "candidate_file": "project/datasets/openaq/candidate.json",
                        "contract_file": "project/datasets/openaq/contract.json",
                        "generated_at": "2026-06-25T00:00:00+00:00",
                        "contract": {
                            "schema_version": "dataset_contract.v1",
                            "dataset_slug": "openaq",
                            "title": "OpenAQ",
                            "source_url": "https://docs.openaq.org/",
                            "intent": "Explore recent air quality.",
                            "summary": "OpenAQ uses an API key with X-API-Key.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            fake_probe = AccessProbeResult(
                dataset_slug="openaq",
                provider="openaq",
                ok=True,
                status="verified",
                checked_at="2026-06-25T00:00:00+00:00",
                credential_names=["OPENAQ_API_KEY"],
                summary="Probe passed.",
            )
            with (
                patch("backend.api.main.ROOT", root),
                patch("backend.api.main.DATASETS_DIR", datasets_dir),
                patch("backend.api.main.run_access_probe", return_value=fake_probe),
            ):
                client = TestClient(app)
                response = client.post("/api/datasets/openaq/access-probe")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])


if __name__ == "__main__":
    unittest.main()
