"""Tagged evaluation-check catalog and deterministic suite-plan compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CheckLayer = Literal[
    "core",
    "data_class",
    "output_format",
    "workload",
    "objective",
    "engineering",
    "robustness",
]
CheckRole = Literal["hard_gate", "objective", "assessment"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationProfile(_StrictModel):
    """Canonical facts used to select checks; provider-native names stay elsewhere."""

    schema_version: Literal["evaluation_profile.v1"] = "evaluation_profile.v1"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
    data_class: str = Field(min_length=1)
    output_format: str = Field(min_length=1)
    output_format_version: int | None = Field(default=None, ge=1)
    workload_kind: str = Field(min_length=1)
    engineering_profile: str = Field(min_length=1)
    dataset_tags: list[str] = Field(default_factory=list)

    @field_validator("data_class", "output_format", "workload_kind")
    @classmethod
    def canonical_names_are_lowercase(cls, value: str) -> str:
        if value != value.lower() or " " in value:
            raise ValueError("Canonical profile names must be lowercase without spaces")
        return value

    @field_validator("dataset_tags")
    @classmethod
    def dataset_tags_are_descriptive(cls, value: list[str]) -> list[str]:
        allowed_prefixes = ("dataset:", "domain:", "family:", "provider:")
        if len(value) != len(set(value)):
            raise ValueError("evaluation_profile dataset_tags must be unique")
        invalid = [tag for tag in value if not tag.startswith(allowed_prefixes)]
        if invalid:
            raise ValueError(
                "Dataset tags are descriptive only and require a dataset/domain/"
                f"family/provider namespace: {invalid}"
            )
        return sorted(value)

    def selection_tags(
        self,
        *,
        engineering_enabled: bool,
        extensibility_probe_enabled: bool,
    ) -> list[str]:
        tags = {
            "core",
            _data_class_tag(self.data_class),
            self.output_format,
            f"workload:{self.workload_kind.replace('_', '-')}",
            *self.dataset_tags,
        }
        if self.output_format_version is not None:
            tags.add(f"{self.output_format}-v{self.output_format_version}")
        if engineering_enabled:
            tags.add("engineering:meroda")
        if extensibility_probe_enabled:
            tags.add("robustness:alternate-contract")
        return sorted(tags)


class EvaluationCheckDefinition(_StrictModel):
    """One reusable check or measurement exposed by an evaluator adapter."""

    check_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    layer: CheckLayer
    role: CheckRole
    tags: list[str] = Field(min_length=1)
    required_profile_tags: list[str] = Field(min_length=1)
    parameter_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def tags_are_unique_and_applicability_is_described(
        self,
    ) -> "EvaluationCheckDefinition":
        if len(self.tags) != len(set(self.tags)):
            raise ValueError(f"Duplicate tags for check {self.check_id}")
        if len(self.required_profile_tags) != len(set(self.required_profile_tags)):
            raise ValueError(f"Duplicate required tags for check {self.check_id}")
        if not set(self.required_profile_tags).issubset(set(self.tags)):
            raise ValueError(
                f"Check {self.check_id} must expose its required profile tags"
            )
        return self


class ResolvedEvaluationCheck(EvaluationCheckDefinition):
    """Frozen catalog entry selected for one suite."""


class ResolvedEvaluationCheckPlan(_StrictModel):
    """Deterministically compiled check plan embedded in a v3 suite."""

    schema_version: Literal["evaluation_check_plan.v1"] = "evaluation_check_plan.v1"
    catalog_version: Literal["evaluation_check_catalog.v1"] = (
        "evaluation_check_catalog.v1"
    )
    profile: EvaluationProfile
    profile_tags: list[str]
    checks: list[ResolvedEvaluationCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def selected_checks_are_unique(self) -> "ResolvedEvaluationCheckPlan":
        ids = [check.check_id for check in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("Resolved evaluation check IDs must be unique")
        return self


class DeclarativeEvaluationCheckInstance(_StrictModel):
    """Data-only parameter binding for one trusted check primitive."""

    schema_version: Literal["evaluation_check_instance.v1"] = (
        "evaluation_check_instance.v1"
    )
    check_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    parameters: dict[str, Any]


MANDATORY_CORE_GATE_IDS = frozenset(
    {
        "contract.grounding",
        "contract.initial_materialization",
        "contract.initial_execution",
        "rerun.execution",
        "rerun.logical_equivalence",
        "provenance.identities",
        "provenance.trusted_inputs_unchanged",
        "provenance.candidate_claims",
        "provenance.receipt.initial",
        "provenance.receipt.rerun",
        "provenance.receipts",
        "security.execution_isolation",
        "security.secret_scan",
    }
)

MANDATORY_SEMANTIC_GATE_BY_DATA_CLASS = {
    "regular_rectilinear_grid": "semantic.native_values_and_missingness",
    "station_time_series": "semantic.station_values_and_missingness",
}


def load_evaluation_profile(path: Path) -> EvaluationProfile:
    """Load one human-editable profile; suite builders freeze its typed result."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Evaluation profile must contain a YAML mapping: {path}")
    return EvaluationProfile.model_validate(payload)


def evaluation_check_catalog() -> tuple[EvaluationCheckDefinition, ...]:
    """Return the immutable built-in catalog in stable check-ID order."""

    return _CATALOG


def compile_evaluation_check_plan(
    profile: EvaluationProfile,
    *,
    engineering_enabled: bool,
    extensibility_probe_enabled: bool,
) -> ResolvedEvaluationCheckPlan:
    """Resolve all applicable checks; callers cannot deselect mandatory entries."""

    profile_tags = profile.selection_tags(
        engineering_enabled=engineering_enabled,
        extensibility_probe_enabled=extensibility_probe_enabled,
    )
    available = set(profile_tags)
    checks = [
        ResolvedEvaluationCheck.model_validate(check.model_dump(mode="json"))
        for check in _CATALOG
        if set(check.required_profile_tags).issubset(available)
    ]
    plan = ResolvedEvaluationCheckPlan(
        profile=profile,
        profile_tags=profile_tags,
        checks=checks,
    )
    validate_mandatory_gate_coverage(plan)
    return plan


def validate_mandatory_gate_coverage(plan: ResolvedEvaluationCheckPlan) -> None:
    """Require fixed core families and the applicable independent semantic gate."""

    hard_gate_ids = {
        check.check_id for check in plan.checks if check.role == "hard_gate"
    }
    required = set(MANDATORY_CORE_GATE_IDS)
    semantic = MANDATORY_SEMANTIC_GATE_BY_DATA_CLASS.get(plan.profile.data_class)
    if semantic is None:
        raise ValueError(
            f"No mandatory semantic gate is registered for {plan.profile.data_class!r}"
        )
    required.add(semantic)
    missing = sorted(required - hard_gate_ids)
    if missing:
        raise ValueError(f"Resolved plan is missing mandatory hard gates: {missing}")


def bind_declarative_check_instance(
    *,
    check_id: str,
    parameters: Mapping[str, Any],
) -> DeclarativeEvaluationCheckInstance:
    """Bind data to trusted logic; arbitrary generated executable code is rejected."""

    definitions = {check.check_id: check for check in _CATALOG}
    definition = definitions.get(check_id)
    if definition is None:
        raise ValueError(
            "Declarative instances may reference trusted catalog checks only; "
            f"unknown check_id={check_id!r}"
        )
    supplied_roots = set(parameters)
    required_roots = {path.split(".", 1)[0] for path in definition.parameter_paths}
    missing = sorted(required_roots - supplied_roots)
    if missing:
        raise ValueError(
            f"Declarative check instance is missing required parameters: {missing}"
        )
    return DeclarativeEvaluationCheckInstance(
        check_id=check_id,
        parameters=dict(parameters),
    )


def validate_emitted_check_ids(
    plan: ResolvedEvaluationCheckPlan,
    emitted_check_ids: set[str],
) -> None:
    """Reject evaluator results that emit checks outside the frozen plan."""

    planned = {
        check.check_id for check in plan.checks if check.role == "hard_gate"
    }
    unknown = sorted(emitted_check_ids - planned)
    if unknown:
        raise ValueError(f"Evaluator emitted checks outside the frozen plan: {unknown}")


def _data_class_tag(data_class: str) -> str:
    aliases = {"regular_rectilinear_grid": "regular-grid"}
    return aliases.get(data_class, f"data-class:{data_class.replace('_', '-')}")


def _definition(
    check_id: str,
    title: str,
    description: str,
    *,
    layer: CheckLayer,
    role: CheckRole,
    tags: tuple[str, ...],
    required: tuple[str, ...],
    parameters: tuple[str, ...] = (),
) -> EvaluationCheckDefinition:
    return EvaluationCheckDefinition(
        check_id=check_id,
        title=title,
        description=description,
        layer=layer,
        role=role,
        tags=list(tags),
        required_profile_tags=list(required),
        parameter_paths=list(parameters),
    )


_CORE = ("core",)
_GRID = ("regular-grid",)
_ZARR = ("zarr",)
_ZARR_V3 = ("zarr", "zarr-v3")
_TENSOR = ("workload:full-field-tensor",)
_STATION = ("data-class:station-time-series",)
_PARQUET = ("parquet",)
_STATION_SCAN = ("workload:filtered-station-scan",)
_MERODA = ("engineering:meroda",)
_ALTERNATE = ("robustness:alternate-contract",)


_CATALOG = tuple(
    sorted(
        (
            _definition(
                "contract.grounding",
                "Suite grounding",
                "Validates the frozen contract, inventory, source, command, and mappings.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:contract-correctness",),
                required=_CORE,
                parameters=(
                    "contract_lock",
                    "inventory",
                    "candidate_source",
                    "execution",
                ),
            ),
            _definition(
                "contract.initial_materialization",
                "Initial materialization availability",
                "Records whether trusted inputs allow the initial controlled run.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:contract-correctness",),
                required=_CORE,
            ),
            _definition(
                "contract.initial_execution",
                "Initial controlled execution",
                "Requires the candidate command to finish successfully in evaluator control.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:contract-correctness",),
                required=_CORE,
                parameters=("execution.command", "execution.timeout_seconds"),
            ),
            _definition(
                "rerun.execution",
                "Exact rerun execution",
                "Executes the same frozen request again in a fresh output location.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:rerun-safety",),
                required=_CORE,
                parameters=("execution.command",),
            ),
            _definition(
                "rerun.logical_equivalence",
                "Exact rerun equivalence",
                "Compares evaluator-owned logical fingerprints from both executions.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:rerun-safety",),
                required=_CORE,
                parameters=("comparison.value_policy",),
            ),
            _definition(
                "provenance.identities",
                "Provenance identities",
                "Requires hashes for trusted inputs, source, oracle, cache, and outputs.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:provenance-security",),
                required=_CORE,
            ),
            _definition(
                "provenance.trusted_inputs_unchanged",
                "Trusted input immutability",
                "Rehashes evaluator-owned inputs after candidate execution.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:provenance-security",),
                required=_CORE,
            ),
            _definition(
                "provenance.candidate_claims",
                "Candidate claim consistency",
                "Checks manifest and pipeline-contract claims against trusted observations.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:provenance-security",),
                required=_CORE,
                parameters=("execution.pipeline_contract", "execution.manifest"),
            ),
            *(
                _definition(
                    check_id,
                    title,
                    description,
                    layer="core",
                    role="hard_gate",
                    tags=_CORE + ("gate:provenance-security",),
                    required=_CORE,
                )
                for check_id, title, description in (
                    (
                        "provenance.receipt.initial",
                        "Initial receipt validity",
                        "Validates linked identities and cache evidence in the initial receipt.",
                    ),
                    (
                        "provenance.receipt.rerun",
                        "Rerun receipt validity",
                        "Validates linked identities and cache evidence in the rerun receipt.",
                    ),
                    (
                        "provenance.receipts",
                        "Receipt availability",
                        "Records that receipt validation was blocked before execution.",
                    ),
                )
            ),
            _definition(
                "security.execution_isolation",
                "Execution isolation",
                "Checks network denial, write confinement, and secret-free execution.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:provenance-security",),
                required=_CORE,
                parameters=("execution_policy",),
            ),
            _definition(
                "security.secret_scan",
                "Secret scan",
                "Scans candidate evidence and outputs for configured credential material.",
                layer="core",
                role="hard_gate",
                tags=_CORE + ("gate:provenance-security",),
                required=_CORE,
                parameters=("security.secret_environment_variables",),
            ),
            _definition(
                "contract.reference_integrity",
                "Reference integrity",
                "Requires the independent regular-grid oracle to be readable and complete.",
                layer="data_class",
                role="hard_gate",
                tags=_GRID + ("gate:contract-correctness",),
                required=_GRID,
                parameters=("reference", "grid.reference_grid"),
            ),
            _definition(
                "contract.scope_coordinates",
                "Grid scope and coordinates",
                "Compares sample and spatial coordinates with declared normalization.",
                layer="data_class",
                role="hard_gate",
                tags=_GRID + ("gate:contract-correctness",),
                required=_GRID,
                parameters=(
                    "grid.expected_shape",
                    "grid.candidate_grid",
                    "comparison.coordinates_atol",
                ),
            ),
            _definition(
                "contract.declared_metadata",
                "Declared grid metadata",
                "Checks only dtype, fill-value, and metadata assertions declared by the suite.",
                layer="data_class",
                role="hard_gate",
                tags=_GRID + ("gate:contract-correctness",),
                required=_GRID,
                parameters=("grid.channels",),
            ),
            _definition(
                "rerun.declared_metadata",
                "Rerun metadata stability",
                "Checks declared metadata again on the independently published rerun.",
                layer="data_class",
                role="hard_gate",
                tags=_GRID + ("gate:rerun-safety",),
                required=_GRID,
                parameters=("grid.channels",),
            ),
            _definition(
                "semantic.native_values_and_missingness",
                "Native value equivalence",
                "Compares all mapped logical values and missingness against the oracle.",
                layer="data_class",
                role="hard_gate",
                tags=_GRID + ("gate:semantic-equivalence",),
                required=_GRID,
                parameters=("grid.channels", "comparison.value_policy"),
            ),
            _definition(
                "contract.zarr_integrity",
                "Zarr integrity",
                "Requires every mapped Zarr array and chunk to be readable.",
                layer="output_format",
                role="hard_gate",
                tags=_ZARR + ("gate:contract-correctness",),
                required=_ZARR,
                parameters=("grid", "comparison.max_block_bytes"),
            ),
            *(
                _definition(
                    check_id,
                    title,
                    description,
                    layer="output_format",
                    role="hard_gate",
                    tags=_ZARR_V3 + (gate,),
                    required=_ZARR_V3,
                    parameters=("output_policy",),
                )
                for check_id, title, description, gate in (
                    (
                        "contract.zarr_output_policy",
                        "Zarr v3 publication policy",
                        "Checks Zarr v3 and complete consolidated metadata.",
                        "gate:contract-correctness",
                    ),
                    (
                        "rerun.zarr_output_policy",
                        "Rerun Zarr v3 policy",
                        "Checks that the rerun preserves the publication policy.",
                        "gate:rerun-safety",
                    ),
                    (
                        "rerun.public_output_policy",
                        "Blocked rerun output policy",
                        "Records a generation-visible output-policy failure before comparison.",
                        "gate:rerun-safety",
                    ),
                )
            ),
            _definition(
                "contract.pytorch_consumer",
                "Full-field tensor consumption",
                "Loads shuffled warm float32 C,H,W samples through the shared consumer.",
                layer="workload",
                role="hard_gate",
                tags=_TENSOR + ("gate:contract-correctness",),
                required=_TENSOR,
                parameters=("workload", "grid.channels"),
            ),
            _definition(
                "contract.station_reference_integrity",
                "Station reference integrity",
                "Requires the independent canonical station-table oracle to be readable.",
                layer="data_class",
                role="hard_gate",
                tags=_STATION + ("gate:contract-correctness",),
                required=_STATION,
                parameters=("reference", "table"),
            ),
            _definition(
                "contract.station_scope_and_keys",
                "Station scope and primary keys",
                "Checks field/station identities, UTC time bounds, required columns, and unique keys.",
                layer="data_class",
                role="hard_gate",
                tags=_STATION + ("gate:contract-correctness",),
                required=_STATION,
                parameters=("table.expected_field_ids", "table.expected_station_ids"),
            ),
            _definition(
                "semantic.station_values_and_missingness",
                "Station value equivalence",
                "Compares all canonical keys, native values, and missingness with the oracle.",
                layer="data_class",
                role="hard_gate",
                tags=_STATION + ("gate:semantic-equivalence",),
                required=_STATION,
                parameters=("table.value_atol", "table.value_rtol"),
            ),
            _definition(
                "rerun.station_logical_equivalence",
                "Station exact-rerun equivalence",
                "Requires independently materialized canonical station tables to match logically.",
                layer="data_class",
                role="hard_gate",
                tags=_STATION + ("gate:rerun-safety",),
                required=_STATION,
                parameters=("output_policy.primary_key",),
            ),
            _definition(
                "contract.parquet_integrity",
                "Parquet integrity",
                "Requires readable Parquet metadata, row groups, schema, and Zstandard encoding.",
                layer="output_format",
                role="hard_gate",
                tags=_PARQUET + ("gate:contract-correctness",),
                required=_PARQUET,
                parameters=("output_policy",),
            ),
            _definition(
                "contract.filtered_station_scan",
                "Filtered station scan",
                "Loads and filters one declared station-field workload from the published table.",
                layer="workload",
                role="hard_gate",
                tags=_STATION_SCAN + ("gate:contract-correctness",),
                required=_STATION_SCAN,
                parameters=("workload",),
            ),
            *(
                _definition(
                    check_id,
                    title,
                    description,
                    layer="objective",
                    role="objective",
                    tags=_CORE + ("objective",),
                    required=_CORE,
                    parameters=parameters,
                )
                for check_id, title, description, parameters in (
                    (
                        "objective.materialization_seconds",
                        "Materialization time",
                        "Measures frozen local input to published output wall time.",
                        ("execution",),
                    ),
                    (
                        "objective.consumer_samples_per_second",
                        "Consumer throughput",
                        "Measures median samples per second for the declared workload.",
                        ("workload",),
                    ),
                    (
                        "objective.output_bytes",
                        "Native output footprint",
                        "Counts bytes across unique mapped native output stores.",
                        ("grid",),
                    ),
                )
            ),
            _definition(
                "engineering.meroda",
                "MERODA engineering quality",
                "Assesses applicable engineering dimensions from blinded bounded evidence.",
                layer="engineering",
                role="assessment",
                tags=_MERODA,
                required=_MERODA,
                parameters=("engineering_quality", "candidate_source.paths"),
            ),
            _definition(
                "robustness.alternate_contract",
                "Unchanged-source alternate contract",
                "Runs the same candidate against another valid frozen contract.",
                layer="robustness",
                role="assessment",
                tags=_ALTERNATE,
                required=_ALTERNATE,
                parameters=("extensibility_probe.suite",),
            ),
        ),
        key=lambda check: check.check_id,
    )
)
