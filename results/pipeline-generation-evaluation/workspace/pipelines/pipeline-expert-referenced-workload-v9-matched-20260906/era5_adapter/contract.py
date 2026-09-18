from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


class ContractError(ValueError):
    """Raised when a runtime lock is incompatible with the frozen inventory."""


def canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return str(field["name"])
    parts = []
    for sel in selectors:
        parts.append(f"{sel['dimension']}={json.dumps(str(sel['value']), ensure_ascii=False)}")
    return f"{field['name']}[" + ",".join(parts) + "]"


def extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    """Extract the selected dataset_contract.v1 from common lock envelopes."""
    if not isinstance(lock, dict):
        raise ContractError("contract_lock must be an object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("selected_contract", "contract", "dataset_contract"):
        val = lock.get(key)
        if isinstance(val, dict) and val.get("schema_version") == "dataset_contract.v1":
            return val
    for key in ("contracts", "contract_candidates", "locks"):
        vals = lock.get(key)
        if isinstance(vals, list):
            matches = [v for v in vals if isinstance(v, dict) and v.get("schema_version") == "dataset_contract.v1"]
            confirmed = [v for v in matches if v.get("human_confirmed") is True]
            if confirmed:
                return confirmed[0]
            if matches:
                return matches[0]
    raise ContractError("could not find dataset_contract.v1 in contract_lock")


def extract_and_validate_contract(lock: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    c = extract_contract(lock)
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ContractError("inventory must be dataset_inventory.v1")
    if c.get("dataset_slug") != inventory.get("dataset_slug"):
        raise ContractError("contract dataset_slug does not match inventory")
    if c.get("dataset_slug") != "reanalysis_era5_pressure_levels":
        raise ContractError("this adapter supports only reanalysis_era5_pressure_levels")

    opts = inventory.get("options", {})
    adv = c.get("advanced_options", {}) or {}
    if adv.get("dataset_id") and adv.get("dataset_id") != inventory.get("dataset_id"):
        raise ContractError("advanced_options.dataset_id does not match inventory")
    if adv.get("data_format", inventory.get("defaults", {}).get("data_format")) not in opts.get("data_format", []):
        raise ContractError("unsupported data_format")
    if adv.get("download_format", inventory.get("defaults", {}).get("download_format")) not in opts.get("download_format", []):
        raise ContractError("unsupported download_format")

    scope = c.get("scope") or {}
    product_type = scope.get("product_type") or (adv.get("product_type") or [None])[0]
    if product_type not in opts.get("product_type", []):
        raise ContractError("unsupported product_type")
    if product_type != "reanalysis":
        raise ContractError("fixed policy supports product_type=reanalysis only")

    fields = c.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("contract must request at least one field")
    seen: set[str] = set()
    for field in fields:
        name = field.get("name")
        if name not in opts.get("variable", []):
            raise ContractError(f"unsupported variable: {name}")
        selectors = field.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ContractError("ERA5 pressure-level channels require exactly one pressure_level selector")
        level = str(selectors[0].get("value"))
        if level not in opts.get("pressure_level", []):
            raise ContractError(f"unsupported pressure_level: {level}")
        fid = canonical_field_id(field)
        if fid in seen:
            raise ContractError(f"duplicate field-selector channel: {fid}")
        seen.add(fid)

    selected_times = ((scope.get("time") or {}).get("selected_times") or [])
    if not selected_times:
        raise ContractError("scope.time.selected_times is required")
    for t in selected_times:
        if t not in opts.get("time", []):
            raise ContractError(f"unsupported selected time: {t}")
    if (scope.get("time") or {}).get("timezone", "UTC") != "UTC":
        raise ContractError("only UTC selected_times are supported")

    _requested_timestamps(c, inventory)  # validates dates and inventory temporal extent
    _validate_area(scope.get("geography") or {}, inventory)
    return c


def _parse_date_or_datetime(value: str, is_end: bool) -> datetime:
    if not isinstance(value, str):
        raise ContractError("date bounds must be strings")
    if "T" in value or len(value) > 10:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    d = date.fromisoformat(value)
    return datetime.combine(d, time.max if is_end else time.min).replace(microsecond=0)


def _requested_timestamps(contract: dict[str, Any], inventory: dict[str, Any]) -> list[datetime]:
    dr = (contract.get("scope") or {}).get("date_range") or {}
    start = _parse_date_or_datetime(dr.get("start_date"), False)
    end = _parse_date_or_datetime(dr.get("end_date"), True)
    if start > end:
        raise ContractError("start_date must be <= end_date")

    extent = (((inventory.get("catalogue_metadata") or {}).get("extent") or {}).get("temporal") or {}).get("interval") or []
    if extent and extent[0]:
        inv_start = datetime.fromisoformat(extent[0][0].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
        inv_end = datetime.fromisoformat(extent[0][1].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
        if start < inv_start or end > inv_end + timedelta(days=1):
            raise ContractError("requested date_range falls outside inventory temporal extent")

    selected_times = (contract.get("scope") or {}).get("time", {}).get("selected_times", [])
    parsed_times = [time.fromisoformat(t) for t in selected_times]
    out: list[datetime] = []
    day = start.date()
    while day <= end.date():
        for tod in parsed_times:
            dt = datetime.combine(day, tod)
            if start <= dt <= end:
                out.append(dt)
        day += timedelta(days=1)
    return out


def requested_timestamps(contract: dict[str, Any], inventory: dict[str, Any]) -> list[datetime]:
    return _requested_timestamps(contract, inventory)


def _validate_area(geo: dict[str, Any], inventory: dict[str, Any]) -> None:
    area = geo.get("cds_area") or geo.get("area")
    if area == "global" or area is None:
        return
    if not (isinstance(area, list) and len(area) == 4):
        raise ContractError("geography.cds_area must be [north, west, south, east]")
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90):
        raise ContractError("invalid latitude bounds in cds_area")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ContractError("invalid longitude bounds in cds_area")
