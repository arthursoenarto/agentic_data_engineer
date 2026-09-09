"""Deterministic editable-contract locking and inventory validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from backend.agents.contract_drafting.schemas import DatasetContract
from backend.agents.dataset_inventory.schemas import DatasetInventory
from backend.contracts.schemas import DatasetContractLock


LOCK_NAME_PATTERN = re.compile(r"contract_v(?P<version>[1-9]\d*)\.lock\.json")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(^|_)(api_?key|key|token|secret|password|authorization)($|_)",
    re.IGNORECASE,
)


class DuplicateKeyError(ValueError):
    """Raised when an editable YAML contract contains a duplicate mapping key."""


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate keys."""


def _construct_unique_mapping(
    loader: _DuplicateKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError(
                f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ContractLockResult:
    """A newly-created or reused immutable contract lock."""

    lock: DatasetContractLock
    path: Path
    created: bool

    @property
    def sha256(self) -> str:
        return content_hash(self.lock.model_dump(mode="json"))


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible content deterministically."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return a SHA-256 hash over canonical JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    """Return a SHA-256 hash over exact file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_editable_contract(path: Path) -> DatasetContract:
    """Parse a human-editable YAML contract safely and validate its schema."""

    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_DuplicateKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid contract YAML: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Editable contract YAML must contain one mapping at its root.")
    try:
        return DatasetContract.model_validate(dict(payload))
    except ValidationError as error:
        raise ValueError(f"Editable contract does not match DatasetContract: {error}") from error


def read_contract_lock(path: Path) -> DatasetContractLock:
    """Read and validate one immutable contract lock."""

    return DatasetContractLock.model_validate_json(path.read_text(encoding="utf-8"))


def load_dataset_contract(path: Path) -> DatasetContract:
    """Read a lock, editable YAML, or historical contract JSON."""

    if path.suffix.lower() in {".yaml", ".yml"}:
        return read_editable_contract(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "dataset_contract_lock.v1":
        return DatasetContractLock.model_validate(payload).contract
    if "contract" in payload:
        return DatasetContract.model_validate(payload["contract"])
    return DatasetContract.model_validate(payload)


def write_editable_contract(contract: DatasetContract, path: Path) -> Path:
    """Write a DatasetContract as human-editable YAML without secret values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.model_dump(mode="json", exclude_none=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    _write_atomic(path, text)
    return path


def lock_dataset_contract(
    *,
    editable_contract_path: Path,
    inventory_path: Path,
    locks_dir: Path | None = None,
    created_at: datetime | None = None,
    intention_path: Path | None = None,
    interaction_path: Path | None = None,
) -> ContractLockResult:
    """Validate an editable contract and create or reuse its immutable lock.

    Lock allocation is append-only. Re-locking unchanged YAML against the same
    inventory reuses the existing version; changed source content allocates the
    next version.
    """

    editable_contract_path = editable_contract_path.resolve()
    inventory_path = inventory_path.resolve()
    lock_directory = (locks_dir or editable_contract_path.parent).resolve()

    contract = read_editable_contract(editable_contract_path)
    _reject_secret_fields(contract.model_dump(mode="json"))
    inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory = DatasetInventory.model_validate(inventory_payload)
    normalized = validate_contract_against_inventory(contract, inventory)

    source_hash = file_hash(editable_contract_path)
    inventory_hash = content_hash(inventory_payload)
    intention_hash = file_hash(intention_path.resolve()) if intention_path else None
    interaction_hash = file_hash(interaction_path.resolve()) if interaction_path else None
    existing = _existing_locks(lock_directory)
    for _, path in existing:
        lock = read_contract_lock(path)
        if (
            lock.source_yaml_sha256 == source_hash
            and lock.inventory_sha256 == inventory_hash
            and lock.contract == normalized
            and lock.intention_sha256 == intention_hash
            and lock.interaction_sha256 == interaction_hash
        ):
            return ContractLockResult(lock=lock, path=path, created=False)

    next_version = existing[-1][0] + 1 if existing else 1
    lock = DatasetContractLock(
        lock_version=next_version,
        created_at=(created_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        source_yaml_sha256=source_hash,
        inventory_sha256=inventory_hash,
        contract_schema_version=normalized.schema_version,
        inventory_schema_version=inventory.schema_version,
        contract=normalized,
        intention_sha256=intention_hash,
        interaction_sha256=interaction_hash,
    )
    path = lock_directory / f"contract_v{next_version}.lock.json"
    lock_directory.mkdir(parents=True, exist_ok=True)
    _write_new(path, canonical_json(lock) + "\n")
    return ContractLockResult(lock=lock, path=path, created=True)


def validate_contract_against_inventory(
    contract: DatasetContract,
    inventory: DatasetInventory,
) -> DatasetContract:
    """Resolve provider defaults and reject selections outside an inventory."""

    if contract.dataset_slug != inventory.dataset_slug:
        raise ValueError(
            "Contract dataset_slug does not match the inventory: "
            f"{contract.dataset_slug!r} != {inventory.dataset_slug!r}."
        )

    scope = _normalized_scope(contract.scope, inventory)
    advanced = _normalized_advanced_options(contract.advanced_options, inventory)
    normalized = contract.model_copy(update={"scope": scope, "advanced_options": advanced})

    variable_options = inventory.options.get("variable")
    for index, field in enumerate(normalized.fields):
        if isinstance(variable_options, list):
            _require_option("field name", field.name, variable_options, location=f"fields[{index}]")
        seen_dimensions: set[str] = set()
        for selector in field.selectors:
            if selector.dimension in seen_dimensions:
                raise ValueError(
                    f"fields[{index}] repeats selector dimension {selector.dimension!r}."
                )
            seen_dimensions.add(selector.dimension)
            options = inventory.options.get(selector.dimension)
            if not isinstance(options, list):
                raise ValueError(
                    f"fields[{index}] uses unsupported selector dimension "
                    f"{selector.dimension!r}."
                )
            _require_option(
                f"selector {selector.dimension}",
                selector.value,
                options,
                location=f"fields[{index}]",
            )
        _validate_explicit_combinations(field.name, field.selectors, inventory)

    _validate_scope(scope, inventory)
    _validate_advanced_options(advanced, inventory)
    return normalized


def _normalized_scope(scope: Mapping[str, Any], inventory: DatasetInventory) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(scope)))
    defaults = inventory.defaults

    if "product_type" not in normalized and "product_type" in defaults:
        product_default = defaults["product_type"]
        normalized["product_type"] = (
            product_default[0]
            if isinstance(product_default, list) and len(product_default) == 1
            else product_default
        )

    geography = normalized.get("geography")
    if geography is None and "area" in defaults:
        normalized["geography"] = {
            "area": "global" if defaults["area"] == [90, -180, -90, 180] else "custom",
            "cds_area": defaults["area"],
            "cds_area_order": ["north", "west", "south", "east"],
        }
    elif isinstance(geography, dict) and "cds_area" not in geography and "area" in defaults:
        geography["cds_area"] = defaults["area"]

    return normalized


def _normalized_advanced_options(
    advanced: Mapping[str, Any],
    inventory: DatasetInventory,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(advanced)))
    for name, value in inventory.defaults.items():
        if name in inventory.options and name not in normalized:
            normalized[name] = value
    if inventory.dataset_id and "dataset_id" not in normalized:
        normalized["dataset_id"] = inventory.dataset_id
    return normalized


def _validate_scope(scope: Mapping[str, Any], inventory: DatasetInventory) -> None:
    product_type = scope.get("product_type")
    if product_type is not None and isinstance(inventory.options.get("product_type"), list):
        values = product_type if isinstance(product_type, list) else [product_type]
        for value in values:
            _require_option("product_type", value, inventory.options["product_type"])

    date_range = scope.get("date_range")
    if date_range is not None:
        if not isinstance(date_range, Mapping):
            raise ValueError("scope.date_range must be a mapping.")
        try:
            start = date.fromisoformat(str(date_range["start_date"]))
            end = date.fromisoformat(str(date_range["end_date"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "scope.date_range requires valid ISO start_date and end_date values."
            ) from error
        if end < start:
            raise ValueError("scope.date_range end_date must not precede start_date.")
        current = start
        while current <= end:
            _require_date_option("year", f"{current.year:04d}", inventory)
            _require_date_option("month", f"{current.month:02d}", inventory)
            _require_date_option("day", f"{current.day:02d}", inventory)
            current = date.fromordinal(current.toordinal() + 1)

    time_scope = scope.get("time")
    if time_scope is not None:
        if not isinstance(time_scope, Mapping):
            raise ValueError("scope.time must be a mapping.")
        selected_times = time_scope.get("selected_times", [])
        if not isinstance(selected_times, list):
            raise ValueError("scope.time.selected_times must be a list.")
        options = inventory.options.get("time")
        if isinstance(options, list):
            for selected_time in selected_times:
                _require_option("time", selected_time, options)

    geography = scope.get("geography")
    if geography is not None:
        if not isinstance(geography, Mapping):
            raise ValueError("scope.geography must be a mapping.")
        area = geography.get("cds_area")
        if area is not None:
            _validate_area(area)


def _validate_advanced_options(
    advanced: Mapping[str, Any],
    inventory: DatasetInventory,
) -> None:
    for name, value in advanced.items():
        options = inventory.options.get(name)
        if not isinstance(options, list):
            continue
        values = value if isinstance(value, list) else [value]
        for selected in values:
            _require_option(name, selected, options, location="advanced_options")


def _validate_area(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            "scope.geography.cds_area must contain [north, west, south, east]."
        )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("scope.geography.cds_area values must be numeric.")
    north, west, south, east = [float(item) for item in value]
    if not (-90 <= south <= north <= 90):
        raise ValueError("Area latitude must satisfy -90 <= south <= north <= 90.")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Area longitude values must be within [-180, 180].")
    if west > east:
        raise ValueError("Dateline-crossing areas are not supported by this contract schema.")


def _validate_explicit_combinations(
    field_name: str,
    selectors: list[Any],
    inventory: DatasetInventory,
) -> None:
    """Apply combination rules when an inventory embeds them.

    Inventories may expose either a compact field-selector mapping under
    ``constraints.field_selector_combinations`` or provider rules under
    ``availability.constraint_rules``. Absence means the inventory only
    supports independent option membership checks.
    """

    compact = inventory.constraints.get("field_selector_combinations")
    if isinstance(compact, Mapping):
        field_rules = compact.get(field_name)
        if field_rules is None:
            raise ValueError(f"No supported selector combinations exist for field {field_name!r}.")
        selector_map = {selector.dimension: str(selector.value) for selector in selectors}
        if isinstance(field_rules, list) and not any(
            isinstance(rule, Mapping)
            and all(str(rule.get(name)) == value for name, value in selector_map.items())
            for rule in field_rules
        ):
            raise ValueError(
                f"Field {field_name!r} and selectors {selector_map!r} are not a supported combination."
            )

    provider_rules = inventory.availability.get("constraint_rules")
    if not isinstance(provider_rules, list) or not provider_rules:
        return
    selection: dict[str, str] = {"variable": str(field_name)}
    selection.update({selector.dimension: str(selector.value) for selector in selectors})
    if not any(_rule_allows(rule, selection) for rule in provider_rules):
        raise ValueError(
            f"Field {field_name!r} and selectors are excluded by provider combination rules."
        )


def _rule_allows(rule: Any, selection: Mapping[str, str]) -> bool:
    if not isinstance(rule, Mapping):
        return False
    for name, selected in selection.items():
        allowed = rule.get(name)
        if allowed is None:
            continue
        allowed_values = allowed if isinstance(allowed, list) else [allowed]
        if selected not in {str(value) for value in allowed_values}:
            return False
    return True


def _require_date_option(name: str, value: str, inventory: DatasetInventory) -> None:
    options = inventory.options.get(name)
    if isinstance(options, list):
        _require_option(name, value, options, location="scope.date_range")


def _require_option(
    name: str,
    value: Any,
    options: list[Any],
    *,
    location: str = "scope",
) -> None:
    allowed = {str(option) for option in options}
    if str(value) not in allowed:
        raise ValueError(
            f"{location} selected {name}={value!r}, which is outside the frozen inventory."
        )


def _existing_locks(directory: Path) -> list[tuple[int, Path]]:
    locks: list[tuple[int, Path]] = []
    if not directory.exists():
        return locks
    for path in directory.iterdir():
        match = LOCK_NAME_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            locks.append((int(match.group("version")), path))
    return sorted(locks)


def _reject_secret_fields(value: Any, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if SENSITIVE_KEY_PATTERN.search(str(key)) and child not in (None, "", [], {}):
                raise ValueError(
                    f"Secret value found at {child_path}; store only an environment-variable name."
                )
            _reject_secret_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")


def _write_new(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
