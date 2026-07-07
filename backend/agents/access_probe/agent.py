"""Agent that writes dataset-specific access probe scripts."""

from __future__ import annotations

import json
from typing import Any, Sequence

from backend.access_probes.schemas import AccessProbeResult
from backend.agents.access_probe.schemas import AccessProbeScript
from backend.agents.contract_drafting.workflow import DatasetContractRun
from backend.llm import LLMClient


class AccessProbeGenerationAgent:
    """Generates deterministic project-local code for probing dataset access."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def create_access_probe_script(
        self,
        *,
        dataset_slug: str,
        run: DatasetContractRun,
        credential_requirements: Sequence[dict[str, Any]],
    ) -> AccessProbeScript:
        """Generate project/datasets/{slug}/access_probe.py."""

        return self._llm.complete_json(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(dataset_slug, run, credential_requirements),
            response_model=AccessProbeScript,
            web_search=True,
            web_search_required=True,
            max_output_tokens=8192,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an Access Probe Generation Agent for an agentic data engineering system. "
            "Generate one small deterministic Python 3 script for exactly the supplied dataset. "
            "The script will be saved as project/datasets/{dataset_slug}/access_probe.py and run locally. "
            "Use the Python standard library by default. A small official provider SDK is acceptable only when "
            "provider documentation presents it as the normal API path. Do not call an LLM from the script. Do not download "
            "large data, mutate remote systems, or perform a full ingestion. The probe should "
            "only verify that credentials and the lightest relevant dataset/API endpoint are reachable. "
            "The script must read secrets only from environment variables and must never print secret values. "
            "It must print only one JSON object to stdout that matches access_probe.v1."
        )

    def _user_prompt(
        self,
        dataset_slug: str,
        run: DatasetContractRun,
        credential_requirements: Sequence[dict[str, Any]],
    ) -> str:
        return "\n".join(
            [
                f"Dataset slug: {dataset_slug}",
                "",
                "Dataset contract run JSON:",
                json.dumps(run.model_dump(mode="json"), indent=2),
                "",
                "Credential requirements JSON:",
                json.dumps(list(credential_requirements), indent=2),
                "",
                "AccessProbeResult JSON schema:",
                json.dumps(AccessProbeResult.model_json_schema(), indent=2),
                "",
                "Script requirements:",
                "- Read DATASET_SLUG from the environment and use it as dataset_slug.",
                "- Read ACCESS_PROBE_TIMEOUT_SECONDS from the environment and use it for HTTP timeouts.",
                "- If a required env var is missing, return ok=false and status='missing_credentials'.",
                "- For API-key datasets, send the credential exactly as provider docs require.",
                "- Use a metadata/account/list/head endpoint when available; avoid heavy data requests.",
                "- Catch HTTP and URL errors and return status='failed' with non-secret details.",
                "- Include endpoint URL, HTTP status, and tiny response facts in details when safe.",
                "- Return notes and assumptions outside python_code only; python_code should be executable.",
            ]
        )
