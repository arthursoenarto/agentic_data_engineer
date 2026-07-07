from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.agents.access_probe.schemas import AccessProbeScript
from backend.agents.contract_drafting import DatasetCandidate, DatasetContract
from backend.agents.contract_drafting.workflow import DatasetContractRun
from backend.credentials import (
    CredentialReference,
    CredentialUpdate,
    credential_status,
    infer_credential_requirements,
    run_access_probe,
    save_credentials,
)

def openaq_run() -> DatasetContractRun:
    return DatasetContractRun(
        candidate_file="project/datasets/openaq/candidate.json",
        contract_file="project/datasets/openaq/contract.json",
        generated_at="2026-06-25T00:00:00+00:00",
        candidate=DatasetCandidate(name="OpenAQ", slug="openaq", url="https://docs.openaq.org/"),
        contract=DatasetContract(
            dataset_slug="openaq",
            title="OpenAQ",
            source_url="https://docs.openaq.org/",
            intent="Explore recent air quality.",
            summary="OpenAQ requires an API key sent with the X-API-Key header.",
            access_methods=["REST API with X-API-Key"],
        ),
    )


FAKE_OPENAQ_PROBE = """
import json
import os
from datetime import UTC, datetime

api_key = os.environ.get("OPENAQ_API_KEY")
print(json.dumps({
    "schema_version": "access_probe.v1",
    "dataset_slug": os.environ.get("DATASET_SLUG", "openaq"),
    "provider": "openaq",
    "ok": bool(api_key),
    "status": "verified" if api_key else "missing_credentials",
    "checked_at": datetime.now(UTC).isoformat(),
    "credential_names": ["OPENAQ_API_KEY"],
    "summary": "Fake project-local probe completed.",
    "details": {"secret_echo": api_key or ""}
}))
"""


def write_fake_probe(dataset_dir: Path, code: str = FAKE_OPENAQ_PROBE) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "access_probe.py").write_text(code, encoding="utf-8")


class CredentialTests(unittest.TestCase):
    def test_infers_openaq_credential_requirement(self) -> None:
        requirements = infer_credential_requirements(openaq_run())

        self.assertEqual(requirements[0].env_var, "OPENAQ_API_KEY")
        self.assertIn("OPENAQ_KEY", requirements[0].aliases)

    def test_save_credentials_and_status_do_not_return_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("MY_OPENAQ_TOKEN=secret-value\n", encoding="utf-8")
            dataset_dir = Path(tmp) / "project/datasets/openaq"
            dataset_dir.mkdir(parents=True)

            save_credentials(
                CredentialUpdate(
                    references=[
                        CredentialReference(requirement="OPENAQ_API_KEY", source="env", env_var="MY_OPENAQ_TOKEN")
                    ]
                ),
                dataset_dir=dataset_dir,
            )
            status = credential_status(dataset_slug="openaq", run=openaq_run(), dataset_dir=dataset_dir, env_path=env_path)

            self.assertIn("MY_OPENAQ_TOKEN", status.set_env_vars)
            self.assertNotIn("secret-value", status.model_dump_json())
            self.assertIn("MY_OPENAQ_TOKEN", (dataset_dir / "credentials.json").read_text(encoding="utf-8"))
            self.assertNotIn("secret-value", (dataset_dir / "credentials.json").read_text(encoding="utf-8"))

    def test_access_probe_reports_missing_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            dataset_dir = Path(tmp) / "project/datasets/openaq"
            write_fake_probe(dataset_dir)

            with patch.dict("os.environ", {}, clear=True):
                result = run_access_probe(dataset_slug="openaq", run=openaq_run(), dataset_dir=dataset_dir, env_path=env_path)

            self.assertFalse(result.ok)
            self.assertEqual(result.status, "missing_credentials")
            self.assertTrue((dataset_dir / "access_probe.json").exists())

    def test_access_probe_runs_project_local_script_and_redacts_secret_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAQ_API_KEY=test-key\n", encoding="utf-8")
            dataset_dir = Path(tmp) / "project/datasets/openaq"
            write_fake_probe(dataset_dir)

            result = run_access_probe(dataset_slug="openaq", run=openaq_run(), dataset_dir=dataset_dir, env_path=env_path)

            self.assertTrue(result.ok)
            self.assertEqual(result.status, "verified")
            self.assertEqual(result.details["secret_echo"], "[REDACTED]")
            self.assertNotIn("test-key", (dataset_dir / "access_probe.json").read_text(encoding="utf-8"))

    def test_access_probe_maps_custom_env_var_reference_to_canonical_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("MY_OPENAQ_TOKEN=test-key\n", encoding="utf-8")
            dataset_dir = Path(tmp) / "project/datasets/openaq"
            write_fake_probe(dataset_dir)
            save_credentials(
                CredentialUpdate(
                    references=[
                        CredentialReference(requirement="OPENAQ_API_KEY", source="env", env_var="MY_OPENAQ_TOKEN")
                    ]
                ),
                dataset_dir=dataset_dir,
            )

            result = run_access_probe(dataset_slug="openaq", run=openaq_run(), dataset_dir=dataset_dir, env_path=env_path)

            self.assertTrue(result.ok)
            self.assertEqual(result.details["secret_echo"], "[REDACTED]")

    def test_access_probe_accepts_relative_dataset_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_path = root / ".env"
            env_path.write_text("OPENAQ_API_KEY=test-key\n", encoding="utf-8")
            dataset_dir = root / "project/datasets/openaq"
            write_fake_probe(dataset_dir)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                relative_dir = Path("project/datasets/openaq")
                result = run_access_probe(
                    dataset_slug="openaq",
                    run=openaq_run(),
                    dataset_dir=relative_dir,
                    env_path=env_path,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertTrue(result.ok)

    def test_access_probe_can_generate_missing_project_local_script(self) -> None:
        class FakeAgent:
            def __init__(self, llm):  # type: ignore[no-untyped-def]
                self.llm = llm

            def create_access_probe_script(self, **kwargs):  # type: ignore[no-untyped-def]
                return AccessProbeScript(python_code=FAKE_OPENAQ_PROBE, notes=["generated in test"])

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAQ_API_KEY=test-key\n", encoding="utf-8")
            dataset_dir = Path(tmp) / "project/datasets/openaq"

            with (
                patch("backend.access_probes.workflow.LLMClient", lambda **kwargs: object()),
                patch("backend.access_probes.workflow.AccessProbeGenerationAgent", FakeAgent),
            ):
                result = run_access_probe(dataset_slug="openaq", run=openaq_run(), dataset_dir=dataset_dir, env_path=env_path)

            self.assertTrue(result.ok)
            self.assertTrue((dataset_dir / "access_probe.py").exists())
            self.assertTrue((dataset_dir / "access_probe.metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
