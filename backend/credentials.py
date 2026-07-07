"""Credential metadata and project-local deterministic access probes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.access_probes import AccessProbeResult, read_access_probe, run_project_access_probe, write_access_probe
from backend.access_probes.workflow import ensure_access_probe_script
from backend.agents.contract_drafting.workflow import DatasetContractRun
from backend.env import ENV_FILE, load_env


class CredentialRequirement(BaseModel):
    """A secret expected by a dataset access path."""

    env_var: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    note: str | None = None
    file_path_hint: str | None = None
    file_key_hint: str | None = None


class CredentialReference(BaseModel):
    """A non-secret pointer to where a credential can be resolved locally."""

    requirement: str
    source: Literal["env", "file"] = "env"
    env_var: str | None = None
    file_path: str | None = None
    file_key: str | None = None


class CredentialReferenceStatus(CredentialReference):
    """Credential reference with availability metadata and no secret values."""

    label: str
    is_set: bool
    note: str | None = None


class CredentialConfig(BaseModel):
    """Dataset-local credential reference metadata."""

    schema_version: str = "dataset_credentials.v1"
    references: list[CredentialReference] = Field(default_factory=list)


class CredentialStatus(BaseModel):
    """Credential metadata returned to the frontend without secret values."""

    schema_version: str = "credential_status.v1"
    dataset_slug: str
    env_file: str
    credential_file: str
    requirements: list[CredentialRequirement]
    references: list[CredentialReferenceStatus]
    set_env_vars: list[str]
    missing_env_vars: list[str]
    access_probe: AccessProbeResult | None = None


class CredentialUpdate(BaseModel):
    """Credential reference updates submitted by the frontend."""

    references: list[CredentialReference]


@dataclass(frozen=True)
class ResolvedCredential:
    requirement: CredentialRequirement
    reference: CredentialReference
    is_set: bool
    value: str | None


def _contract_text(run: DatasetContractRun) -> str:
    contract = run.contract
    return " ".join(
        [
            contract.title,
            contract.provider or "",
            contract.dataset_family or "",
            str(contract.source_url),
            contract.intent,
            contract.summary,
            *contract.access_methods,
            *contract.credential_requirements,
            *contract.assumptions,
            *contract.risks_or_unknowns,
            *[
                f"{item.name} {item.display_name or ''} "
                f"{' '.join(f'{selector.dimension} {selector.value}' for selector in item.selectors)} "
                f"{item.description or ''}"
                for item in contract.fields
            ],
            *[f"{item.title or ''} {item.url} {item.note or ''}" for item in contract.evidence],
        ]
    ).lower()


def infer_credential_requirements(run: DatasetContractRun) -> list[CredentialRequirement]:
    """Infer credential names from contract evidence and access notes."""

    text = _contract_text(run)
    needs_credential = re.search(r"api key|token|credential|auth|x-api-key|personal-access-token|cdsapirc", text)
    if not needs_credential:
        return []

    if re.search(r"era5|copernicus|cds\.climate|cdsapirc", text):
        return [
            CredentialRequirement(
                env_var="CDSAPI_URL",
                label="CDS API URL",
                note="Copernicus CDS API endpoint. This may also be stored in ~/.cdsapirc.",
                file_path_hint="~/.cdsapirc",
                file_key_hint="url",
            ),
            CredentialRequirement(
                env_var="CDSAPI_KEY",
                label="CDS API key",
                note="Copernicus CDS personal access token. This may also be stored in ~/.cdsapirc.",
                aliases=["CDS_PERSONAL_ACCESS_TOKEN"],
                file_path_hint="~/.cdsapirc",
                file_key_hint="key",
            ),
        ]

    if re.search(r"openaq|x-api-key", text):
        return [
            CredentialRequirement(
                env_var="OPENAQ_API_KEY",
                label="OpenAQ API key",
                aliases=["OPENAQ_KEY"],
                note="Used as the OpenAQ X-API-Key header. OPENAQ_KEY is accepted as a local alias.",
            )
        ]

    return [
        CredentialRequirement(
            env_var="DATASET_API_KEY",
            label="Dataset API key",
            note="Rename this to a provider-specific credential before building a production probe.",
        )
    ]


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Read KEY=VALUE pairs from a local env file."""

    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _credential_config_path(dataset_dir: Path) -> Path:
    return dataset_dir / "credentials.json"


def _read_credential_config(dataset_dir: Path) -> CredentialConfig | None:
    path = _credential_config_path(dataset_dir)
    if not path.exists():
        return None
    return CredentialConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _write_credential_config(config: CredentialConfig, dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _credential_config_path(dataset_dir).write_text(
        config.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_env_var_name(value: str) -> str:
    env_var = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env_var):
        raise ValueError(f"Invalid environment variable name: {value}")
    return env_var


def _default_reference(requirement: CredentialRequirement, env_values: dict[str, str]) -> CredentialReference:
    env_var = next((name for name in [requirement.env_var, *requirement.aliases] if env_values.get(name)), requirement.env_var)
    return CredentialReference(requirement=requirement.env_var, source="env", env_var=env_var)


def _credential_references(
    *,
    requirements: list[CredentialRequirement],
    dataset_dir: Path,
    env_values: dict[str, str],
) -> list[CredentialReference]:
    config = _read_credential_config(dataset_dir)
    configured = {item.requirement: item for item in config.references} if config else {}
    references: list[CredentialReference] = []
    for requirement in requirements:
        references.append(configured.get(requirement.env_var) or _default_reference(requirement, env_values))
    return references


def _parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":" if ":" in line else None
        if not separator:
            continue
        key, value = line.split(separator, 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _file_key_for_requirement(requirement: CredentialRequirement, reference: CredentialReference) -> str:
    return reference.file_key or requirement.file_key_hint or requirement.env_var


def _resolve_reference(
    *,
    requirement: CredentialRequirement,
    reference: CredentialReference,
    env_values: dict[str, str],
) -> ResolvedCredential:
    if reference.source == "file":
        if not reference.file_path:
            return ResolvedCredential(requirement=requirement, reference=reference, is_set=False, value=None)
        path = Path(reference.file_path).expanduser()
        file_values = _parse_key_value_file(path)
        value = file_values.get(_file_key_for_requirement(requirement, reference))
        return ResolvedCredential(requirement=requirement, reference=reference, is_set=bool(value), value=value)

    env_var = _validate_env_var_name(reference.env_var or requirement.env_var)
    normalized = reference.model_copy(update={"env_var": env_var})
    value = env_values.get(env_var)
    return ResolvedCredential(requirement=requirement, reference=normalized, is_set=bool(value), value=value)


def _resolved_credentials(
    *,
    requirements: list[CredentialRequirement],
    dataset_dir: Path,
    env_path: Path,
) -> list[ResolvedCredential]:
    load_env(env_path)
    env_values = {**{key: value for key, value in os.environ.items() if value}, **read_env_file(env_path)}
    references = _credential_references(requirements=requirements, dataset_dir=dataset_dir, env_values=env_values)
    requirements_by_name = {item.env_var: item for item in requirements}
    resolved: list[ResolvedCredential] = []
    for reference in references:
        requirement = requirements_by_name.get(reference.requirement)
        if requirement:
            resolved.append(_resolve_reference(requirement=requirement, reference=reference, env_values=env_values))
    return resolved


def credential_status(
    *,
    dataset_slug: str,
    run: DatasetContractRun,
    dataset_dir: Path,
    env_path: Path = ENV_FILE,
) -> CredentialStatus:
    """Return credential setup metadata without exposing secret values."""

    load_env(env_path)
    env_values = {**{key: value for key, value in os.environ.items() if value}, **read_env_file(env_path)}
    requirements = infer_credential_requirements(run)
    resolved = _resolved_credentials(requirements=requirements, dataset_dir=dataset_dir, env_path=env_path)
    references = [
        CredentialReferenceStatus(
            **item.reference.model_dump(),
            label=item.requirement.label,
            is_set=item.is_set,
            note=item.requirement.note,
        )
        for item in resolved
    ]
    set_env_vars = [
        item.reference.env_var or item.requirement.env_var
        for item in resolved
        if item.is_set and item.reference.source == "env"
    ]
    missing_env_vars = [item.requirement.env_var for item in resolved if not item.is_set]
    return CredentialStatus(
        dataset_slug=dataset_slug,
        env_file=str(env_path),
        credential_file=str(_credential_config_path(dataset_dir)),
        requirements=requirements,
        references=references,
        set_env_vars=set_env_vars,
        missing_env_vars=missing_env_vars,
        access_probe=read_access_probe(dataset_dir),
    )


def save_credentials(update: CredentialUpdate, *, dataset_dir: Path) -> None:
    normalized: list[CredentialReference] = []
    for reference in update.references:
        if reference.source == "env":
            normalized.append(
                reference.model_copy(
                    update={
                        "env_var": _validate_env_var_name(reference.env_var or reference.requirement),
                        "file_path": None,
                        "file_key": None,
                    }
                )
            )
        else:
            if not reference.file_path or not reference.file_path.strip():
                raise ValueError("Credential file path is required for file-based references.")
            normalized.append(
                reference.model_copy(
                    update={
                        "env_var": None,
                        "file_path": reference.file_path.strip(),
                        "file_key": (reference.file_key or "").strip() or None,
                    }
                )
            )

    _write_credential_config(CredentialConfig(references=normalized), dataset_dir)


def credential_environment(
    *,
    run: DatasetContractRun,
    dataset_dir: Path,
    env_path: Path = ENV_FILE,
) -> tuple[list[CredentialRequirement], dict[str, str]]:
    """Resolve configured references into canonical env vars for probe scripts."""

    requirements = infer_credential_requirements(run)
    resolved = _resolved_credentials(requirements=requirements, dataset_dir=dataset_dir, env_path=env_path)
    env_overrides = {
        item.requirement.env_var: item.value
        for item in resolved
        if item.is_set and item.value is not None
    }
    return requirements, env_overrides


def run_access_probe(
    *,
    dataset_slug: str,
    run: DatasetContractRun,
    dataset_dir: Path,
    env_path: Path = ENV_FILE,
    timeout_seconds: int = 20,
    generate_if_missing: bool = True,
) -> AccessProbeResult:
    """Generate if needed, then run the dataset-local access probe script."""

    requirements, env_overrides = credential_environment(run=run, dataset_dir=dataset_dir, env_path=env_path)
    credential_names = [item.env_var for item in requirements]

    if generate_if_missing and not (dataset_dir / "access_probe.py").exists():
        try:
            ensure_access_probe_script(
                dataset_slug=dataset_slug,
                run=run,
                dataset_dir=dataset_dir,
                credential_requirements=requirements,
            )
        except Exception as error:
            result = AccessProbeResult(
                dataset_slug=dataset_slug,
                provider="unknown",
                ok=False,
                status="generation_failed",
                checked_at=datetime.now(UTC).isoformat(),
                credential_names=credential_names,
                summary="Could not generate a dataset-local access probe script.",
                details={"error": str(error)},
            )
            write_access_probe(dataset_dir, result)
            return result

    return run_project_access_probe(
        dataset_slug=dataset_slug,
        dataset_dir=dataset_dir,
        env_path=env_path,
        timeout_seconds=timeout_seconds,
        credential_names=credential_names,
        env_overrides=env_overrides,
    )
