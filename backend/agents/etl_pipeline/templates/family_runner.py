"""Framework-owned standalone entrypoint for a generated dataset-family pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_runtime_inputs(
    lock: Mapping[str, Any],
    inventory: Mapping[str, Any],
    pipeline_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    accepted_locks = pipeline_contract["accepted_lock_schema_versions"]
    if lock.get("schema_version") not in accepted_locks:
        raise ValueError("Unsupported contract-lock schema version.")
    accepted_contracts = pipeline_contract["accepted_contract_schema_versions"]
    contract = lock.get("contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") not in accepted_contracts:
        raise ValueError("Unsupported DatasetContract schema version.")
    accepted_inventories = pipeline_contract["accepted_inventory_schema_versions"]
    if inventory.get("schema_version") not in accepted_inventories:
        raise ValueError("Unsupported dataset inventory schema version.")
    if lock.get("inventory_sha256") != _canonical_hash(inventory):
        raise ValueError("Contract lock was not validated against this inventory snapshot.")
    dataset_slug = pipeline_contract["dataset_slug"]
    if contract.get("dataset_slug") != dataset_slug or inventory.get("dataset_slug") != dataset_slug:
        raise ValueError("Dataset identity differs across lock, inventory, and pipeline contract.")

    manifest_core = dict(manifest)
    manifest_core.pop("pipeline_contract_hash", None)
    if pipeline_contract["generation_manifest_hash"] != _canonical_hash(manifest_core):
        raise ValueError("Generation manifest hash does not match pipeline_contract.json.")
    if manifest.get("pipeline_contract_hash") != _canonical_hash(pipeline_contract):
        raise ValueError("Pipeline contract hash does not match manifest.json.")

    _validate_contract_against_inventory(contract, inventory)


def _validate_contract_against_inventory(
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    options = inventory.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("Inventory does not expose an option mapping.")

    variable_options = options.get("variable")
    for index, field in enumerate(contract.get("fields", [])):
        if not isinstance(field, Mapping):
            raise ValueError(f"fields[{index}] must be an object.")
        if isinstance(variable_options, list):
            _require_option(field.get("name"), variable_options, f"fields[{index}].name")
        seen: set[str] = set()
        for selector in field.get("selectors", []):
            if not isinstance(selector, Mapping):
                raise ValueError(f"fields[{index}].selectors must contain objects.")
            dimension = selector.get("dimension")
            if not isinstance(dimension, str) or dimension in seen:
                raise ValueError(f"fields[{index}] has an invalid or duplicate selector.")
            seen.add(dimension)
            selector_options = options.get(dimension)
            if not isinstance(selector_options, list):
                raise ValueError(f"Unsupported selector dimension: {dimension}")
            _require_option(
                selector.get("value"),
                selector_options,
                f"fields[{index}].selectors.{dimension}",
            )
        _validate_combination(field, inventory)

    scope = contract.get("scope", {})
    if not isinstance(scope, Mapping):
        raise ValueError("Contract scope must be an object.")
    product_type = scope.get("product_type")
    if product_type is not None and isinstance(options.get("product_type"), list):
        for value in product_type if isinstance(product_type, list) else [product_type]:
            _require_option(value, options["product_type"], "scope.product_type")

    date_range = scope.get("date_range")
    if date_range is not None:
        if not isinstance(date_range, Mapping):
            raise ValueError("scope.date_range must be an object.")
        start = date.fromisoformat(str(date_range["start_date"]))
        end = date.fromisoformat(str(date_range["end_date"]))
        if end < start:
            raise ValueError("Contract end date precedes start date.")
        current = start
        while current <= end:
            for name, value in (
                ("year", f"{current.year:04d}"),
                ("month", f"{current.month:02d}"),
                ("day", f"{current.day:02d}"),
            ):
                if isinstance(options.get(name), list):
                    _require_option(value, options[name], f"scope.date_range.{name}")
            current = date.fromordinal(current.toordinal() + 1)

    time_scope = scope.get("time")
    if isinstance(time_scope, Mapping) and isinstance(options.get("time"), list):
        selected_times = time_scope.get("selected_times", [])
        if not isinstance(selected_times, list):
            raise ValueError("scope.time.selected_times must be a list.")
        for value in selected_times:
            _require_option(value, options["time"], "scope.time.selected_times")

    geography = scope.get("geography")
    if isinstance(geography, Mapping) and "cds_area" in geography:
        _validate_area(geography["cds_area"])

    advanced = contract.get("advanced_options", {})
    if isinstance(advanced, Mapping):
        for name, value in advanced.items():
            allowed = options.get(name)
            if not isinstance(allowed, list):
                continue
            for selected in value if isinstance(value, list) else [value]:
                _require_option(selected, allowed, f"advanced_options.{name}")


def _validate_combination(field: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    constraints = inventory.get("constraints", {})
    compact = constraints.get("field_selector_combinations") if isinstance(constraints, Mapping) else None
    if isinstance(compact, Mapping):
        rules = compact.get(field.get("name"))
        selection = {
            selector["dimension"]: str(selector["value"])
            for selector in field.get("selectors", [])
            if isinstance(selector, Mapping)
        }
        if not isinstance(rules, list) or not any(
            isinstance(rule, Mapping)
            and all(str(rule.get(name)) == value for name, value in selection.items())
            for rule in rules
        ):
            raise ValueError("Field-selector combination is excluded by the inventory.")

    availability = inventory.get("availability", {})
    provider_rules = (
        availability.get("constraint_rules")
        if isinstance(availability, Mapping)
        else None
    )
    if not isinstance(provider_rules, list) or not provider_rules:
        return
    selection = {"variable": str(field.get("name"))}
    selection.update(
        {
            str(selector.get("dimension")): str(selector.get("value"))
            for selector in field.get("selectors", [])
            if isinstance(selector, Mapping)
        }
    )
    if not any(_provider_rule_allows(rule, selection) for rule in provider_rules):
        raise ValueError("Field-selector combination is excluded by provider rules.")


def _provider_rule_allows(rule: Any, selection: Mapping[str, str]) -> bool:
    if not isinstance(rule, Mapping):
        return False
    for name, selected in selection.items():
        allowed = rule.get(name)
        if allowed is None:
            continue
        values = allowed if isinstance(allowed, list) else [allowed]
        if selected not in {str(value) for value in values}:
            return False
    return True


def _require_option(value: Any, options: list[Any], location: str) -> None:
    if str(value) not in {str(option) for option in options}:
        raise ValueError(f"{location}={value!r} is outside the frozen inventory.")


def _validate_area(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Geographic area must be [north, west, south, east].")
    north, west, south, east = [float(item) for item in value]
    if not (-90 <= south <= north <= 90):
        raise ValueError("Invalid area latitude bounds.")
    if not (-180 <= west <= east <= 180):
        raise ValueError("Invalid area longitude bounds.")


def _fingerprint(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    size = 0
    count = 0
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        count += 1
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    if count == 0:
        raise ValueError("Pipeline produced no output files.")
    return size, count, digest.hexdigest()


def _redact(value: str, credential_names: list[str]) -> str:
    redacted = value
    for name in credential_names:
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _merge_cache_evidence(
    cache_evidence: dict[str, Any],
    result: Mapping[str, Any],
) -> None:
    reported_cache = result.get("cache")
    if isinstance(reported_cache, Mapping):
        cache_evidence.update(reported_cache)
        return

    partitions = result.get("cache_evidence")
    if not isinstance(partitions, list):
        return
    reused_keys: list[str] = []
    acquired_keys: list[str] = []
    hits = 0
    misses = 0
    acquired = 0
    for partition in partitions:
        if not isinstance(partition, Mapping):
            continue
        hits += int(bool(partition.get("hit")))
        misses += int(bool(partition.get("miss")))
        acquired += int(bool(partition.get("acquired")))
        reused_key = partition.get("reused_key") or partition.get("reused-key")
        acquired_key = partition.get("acquired_key") or partition.get("acquired-key")
        if reused_key:
            reused_keys.append(str(reused_key))
        if acquired_key:
            acquired_keys.append(str(acquired_key))
    cache_evidence.update(
        {
            "hits": hits,
            "misses": misses,
            "acquired": acquired,
            "reused_keys": reused_keys,
            "acquired_keys": acquired_keys,
        }
    )


def _field_id(field: Mapping[str, Any]) -> str:
    name = str(field.get("name", ""))
    selectors = field.get("selectors", [])
    if not selectors:
        return name
    rendered = ",".join(
        f'{selector["dimension"]}={json.dumps(str(selector["value"]), ensure_ascii=True)}'
        for selector in selectors
        if isinstance(selector, Mapping)
    )
    return f"{name}[{rendered}]"


def _relative_artifact_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be relative and must not contain '..'.")
    return value


def _validate_dataset_artifact(
    value: Any,
    *,
    contract: Mapping[str, Any],
    output_dir: Path,
    zarr_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "The regular-grid family interface requires a dataset_artifact mapping."
        )
    if value.get("schema_version") != "dataset_artifact_layout.v1":
        raise ValueError("Unsupported dataset_artifact schema version.")
    if value.get("storage_format") != "zarr":
        raise ValueError("dataset_artifact.storage_format must be 'zarr'.")

    store_path = _relative_artifact_path(value.get("store_path"), label="store_path")
    dimensions = value.get("dimensions")
    coordinates = value.get("coordinates")
    required_axes = {"sample", "y", "x"}
    if not isinstance(dimensions, Mapping) or set(dimensions) != required_axes:
        raise ValueError("dataset_artifact.dimensions must contain sample, y, and x.")
    if not isinstance(coordinates, Mapping) or set(coordinates) != required_axes:
        raise ValueError("dataset_artifact.coordinates must contain sample, y, and x.")
    normalized_dimensions = {
        axis: _relative_artifact_path(path, label=f"dimensions.{axis}")
        for axis, path in dimensions.items()
    }
    normalized_coordinates = {
        axis: _relative_artifact_path(path, label=f"coordinates.{axis}")
        for axis, path in coordinates.items()
    }

    channels = value.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError("dataset_artifact.channels must be a non-empty list.")
    normalized_channels: list[dict[str, Any]] = []
    for index, channel in enumerate(channels):
        if not isinstance(channel, Mapping):
            raise ValueError(f"dataset_artifact.channels[{index}] must be a mapping.")
        field_id = channel.get("field_id")
        if not isinstance(field_id, str) or not field_id:
            raise ValueError(f"dataset_artifact.channels[{index}].field_id is required.")
        selectors = channel.get("selectors", {})
        selector_paths = channel.get("selector_coordinate_paths", {})
        if not isinstance(selectors, Mapping) or not isinstance(selector_paths, Mapping):
            raise ValueError("Channel selectors and selector_coordinate_paths must be mappings.")
        normalized_channels.append(
            {
                "field_id": field_id,
                "array_path": _relative_artifact_path(
                    channel.get("array_path"),
                    label=f"channels[{index}].array_path",
                ),
                "selectors": {str(name): str(selected) for name, selected in selectors.items()},
                "selector_coordinate_paths": {
                    str(name): _relative_artifact_path(
                        path,
                        label=f"channels[{index}].selector_coordinate_paths.{name}",
                    )
                    for name, path in selector_paths.items()
                },
            }
        )

    expected_fields = sorted(
        _field_id(field)
        for field in contract.get("fields", [])
        if isinstance(field, Mapping)
    )
    observed_fields = sorted(channel["field_id"] for channel in normalized_channels)
    if observed_fields != expected_fields:
        raise ValueError(
            "dataset_artifact channels do not exactly match requested field-selector IDs: "
            f"expected {expected_fields}, got {observed_fields}."
        )

    store = (output_dir / store_path).resolve()
    try:
        store.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("dataset_artifact store escapes the output directory.") from error
    if not store.is_dir() or not ((store / ".zgroup").is_file() or (store / "zarr.json").is_file()):
        raise ValueError("dataset_artifact store is not a published Zarr group.")
    _validate_zarr_output_policy(store, zarr_policy)
    for array_path in [
        *normalized_coordinates.values(),
        *(channel["array_path"] for channel in normalized_channels),
        *(
            path
            for channel in normalized_channels
            for path in channel["selector_coordinate_paths"].values()
        ),
    ]:
        array = store / array_path
        if not ((array / ".zarray").is_file() or (array / "zarr.json").is_file()):
            raise ValueError(f"Declared Zarr array is missing: {array_path}")

    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": store_path,
        "dimensions": normalized_dimensions,
        "coordinates": normalized_coordinates,
        "channels": normalized_channels,
    }


def _validate_tabular_dataset_artifact(
    value: Any,
    *,
    output_dir: Path,
    parquet_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("The station-time-series interface requires dataset_artifact.")
    if value.get("schema_version") != "dataset_artifact_layout.tabular.v1":
        raise ValueError("Unsupported tabular dataset_artifact schema version.")
    if value.get("storage_format") != "parquet":
        raise ValueError("Tabular dataset_artifact.storage_format must be 'parquet'.")
    file_path = _relative_artifact_path(value.get("file_path"), label="file_path")
    if not file_path.endswith(".parquet"):
        raise ValueError("Tabular dataset_artifact.file_path must end in .parquet.")
    columns = value.get("columns")
    primary_key = value.get("primary_key")
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ValueError("Tabular dataset_artifact.columns must be a unique list.")
    if (
        not isinstance(primary_key, list)
        or not primary_key
        or len(primary_key) != len(set(primary_key))
    ):
        raise ValueError("Tabular dataset_artifact.primary_key must be a unique list.")
    required = list(parquet_policy.get("required_columns", []))
    expected_key = list(parquet_policy.get("primary_key", []))
    if not set(required).issubset(columns):
        raise ValueError("Parquet artifact omits required canonical columns.")
    if primary_key != expected_key:
        raise ValueError("Parquet artifact primary_key differs from the output policy.")

    published = (output_dir / file_path).resolve()
    try:
        published.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("Parquet artifact escapes the output directory.") from error
    if not published.is_file():
        raise ValueError("Declared Parquet artifact does not exist.")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("The tabular family interface requires pyarrow.") from error
    metadata = pq.read_metadata(published)
    schema_columns = metadata.schema.to_arrow_schema().names
    if metadata.num_rows <= 0 or schema_columns != columns:
        raise ValueError("Parquet physical schema or row count differs from its declaration.")
    keys = pq.read_table(published, columns=primary_key).to_pandas()
    if keys.isnull().any(axis=None):
        raise ValueError("Parquet primary-key columns must not contain nulls.")
    if int(keys.drop_duplicates().shape[0]) != metadata.num_rows:
        raise ValueError("Parquet primary key contains duplicate rows.")
    return {
        "schema_version": "dataset_artifact_layout.tabular.v1",
        "storage_format": "parquet",
        "file_path": file_path,
        "columns": columns,
        "primary_key": primary_key,
    }


def _validate_zarr_output_policy(
    store: Path,
    policy: Mapping[str, Any] | None,
) -> None:
    if policy is None:
        return
    if policy.get("format_version") != 3:
        raise ValueError("Unsupported framework Zarr format policy.")
    if policy.get("consolidated_metadata") is not True:
        raise ValueError("Unsupported framework Zarr metadata policy.")
    root_metadata_path = store / "zarr.json"
    if not root_metadata_path.is_file() or (store / ".zgroup").exists():
        raise ValueError("Published output must be a Zarr format 3 group.")
    root_metadata = _read_json(root_metadata_path)
    if root_metadata.get("zarr_format") != 3 or root_metadata.get("node_type") != "group":
        raise ValueError("Published output root is not a Zarr format 3 group.")
    consolidated = root_metadata.get("consolidated_metadata")
    if not isinstance(consolidated, Mapping) or not isinstance(
        consolidated.get("metadata"), Mapping
    ):
        raise ValueError("Published Zarr v3 output lacks consolidated root metadata.")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generated dataset-family pipeline.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    started_clock = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    root = Path(__file__).resolve().parent
    pipeline_contract = _read_json(root / "pipeline_contract.json")
    manifest = _read_json(root / "manifest.json")
    credential_names = list(pipeline_contract.get("credential_environment_variables", []))
    run_id = args.run_receipt.parent.name
    staging = args.output_dir.parent / f".{args.output_dir.name}.tmp"
    cache_evidence: dict[str, Any] = {
        "cache_dir": str(args.cache_dir),
        "hits": 0,
        "misses": 0,
        "acquired": 0,
        "reused_keys": [],
        "acquired_keys": [],
        "notes": [],
    }
    warnings: list[str] = []
    diagnostics: list[str] = []
    outputs: list[dict[str, Any]] = []
    dataset_artifact: dict[str, Any] | None = None
    status = "failed"
    exit_code = 1

    if args.run_receipt.exists():
        raise FileExistsError(f"Run receipt already exists: {args.run_receipt}")

    lock: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    try:
        lock = _read_json(args.contract)
        inventory = _read_json(args.inventory)
        _validate_runtime_inputs(lock, inventory, pipeline_contract, manifest)

        if args.output_dir.exists():
            raise FileExistsError(f"Immutable output directory already exists: {args.output_dir}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)

        implementation = importlib.import_module("pipeline_impl")
        run = getattr(implementation, "run_pipeline", None)
        if not callable(run):
            raise TypeError("pipeline_impl.py must define callable run_pipeline(...).")
        result = run(
            contract_lock=lock,
            inventory=inventory,
            cache_dir=args.cache_dir,
            output_dir=staging,
        )
        if result is not None and not isinstance(result, Mapping):
            raise TypeError("pipeline_impl.run_pipeline() must return a mapping or None.")
        if isinstance(result, Mapping):
            _merge_cache_evidence(cache_evidence, result)
            reported_warnings = result.get("warnings")
            if isinstance(reported_warnings, list):
                warnings.extend(str(item) for item in reported_warnings)
            if pipeline_contract.get("implementation_interface_version") in {
                "family_pipeline_interface.v2",
                "family_pipeline_interface.v3",
                "family_pipeline_interface.v4",
            }:
                zarr_policy = pipeline_contract.get("policy", {}).get("zarr")
                parquet_policy = pipeline_contract.get("policy", {}).get("parquet")
                if (
                    pipeline_contract.get("implementation_interface_version")
                    == "family_pipeline_interface.v3"
                    and not isinstance(zarr_policy, Mapping)
                ):
                    raise ValueError(
                        "family_pipeline_interface.v3 requires a Zarr output policy."
                    )
                if (
                    pipeline_contract.get("implementation_interface_version")
                    == "family_pipeline_interface.v4"
                ):
                    if not isinstance(parquet_policy, Mapping):
                        raise ValueError(
                            "family_pipeline_interface.v4 requires a Parquet output policy."
                        )
                    dataset_artifact = _validate_tabular_dataset_artifact(
                        result.get("dataset_artifact"),
                        output_dir=staging,
                        parquet_policy=parquet_policy,
                    )
                else:
                    dataset_artifact = _validate_dataset_artifact(
                        result.get("dataset_artifact"),
                        contract=lock["contract"],
                        output_dir=staging,
                        zarr_policy=(
                            zarr_policy if isinstance(zarr_policy, Mapping) else None
                        ),
                    )

        size, count, digest = _fingerprint(staging)
        staging.replace(args.output_dir)
        artifact = pipeline_contract["output_artifact"]
        outputs.append(
            {
                "artifact_id": artifact["artifact_id"],
                "path": str(args.output_dir),
                "storage_format": artifact["storage_format"],
                "size_bytes": size,
                "file_count": count,
                "sha256": digest,
            }
        )
        status = "succeeded"
        exit_code = int(pipeline_contract["success_exit_code"])
    except Exception as error:
        if staging.exists():
            shutil.rmtree(staging)
        diagnostics.append(_redact(f"{type(error).__name__}: {error}", credential_names))
        print(diagnostics[-1], file=sys.stderr)

    completed_at = datetime.now(UTC).isoformat()
    manifest_core = dict(manifest)
    manifest_core.pop("pipeline_contract_hash", None)
    receipt = {
        "schema_version": "etl_pipeline_run.v1",
        "run_id": run_id,
        "pipeline_id": pipeline_contract["pipeline_id"],
        "manifest_hash": _canonical_hash(manifest_core),
        "pipeline_contract_hash": _canonical_hash(pipeline_contract),
        "contract_lock_hash": _canonical_hash(lock),
        "inventory_hash": _canonical_hash(inventory),
        "command": [sys.executable, *sys.argv],
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": time.monotonic() - started_clock,
        "exit_code": exit_code,
        "final_status": status,
        "cache": cache_evidence,
        "outputs": outputs,
        "dataset_artifact": dataset_artifact,
        "repair_run_reference": os.environ.get("PIPELINE_REPAIR_REFERENCE"),
        "warnings": [_redact(value, credential_names) for value in warnings],
        "diagnostics": diagnostics,
    }
    _write_new_json(args.run_receipt, receipt)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
