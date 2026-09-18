"""Standalone ERA5 pressure-level fixture-to-Zarr adapter.

Public entry point:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The implementation is intentionally offline-first.  If cache_dir contains a
source_fixture_manifest.json, every listed raw file is verified and consumed
before any provider/client construction.  This adapter does not perform network
retrieval; absence of a fixture is a deterministic validation error.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


CANONICAL_DIMS = ("time", "pressure_level", "latitude", "longitude")
STORE_NAME = "dataset.zarr"
FIXTURE_MANIFEST = "source_fixture_manifest.json"

COORD_ALIASES = {
    "time": ("time", "valid_time", "forecast_reference_time"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "level", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}

# Common cfgrib shortName aliases for ERA5 pressure-level variables.  Full CDS
# request names are preferred when present.
VARIABLE_ALIASES = {
    "divergence": ("divergence", "d"),
    "fraction_of_cloud_cover": ("fraction_of_cloud_cover", "cc"),
    "geopotential": ("geopotential", "z"),
    "ozone_mass_mixing_ratio": ("ozone_mass_mixing_ratio", "o3"),
    "potential_vorticity": ("potential_vorticity", "pv"),
    "relative_humidity": ("relative_humidity", "r"),
    "specific_cloud_ice_water_content": ("specific_cloud_ice_water_content", "ciwc"),
    "specific_cloud_liquid_water_content": ("specific_cloud_liquid_water_content", "clwc"),
    "specific_humidity": ("specific_humidity", "q"),
    "specific_rain_water_content": ("specific_rain_water_content", "crwc"),
    "specific_snow_water_content": ("specific_snow_water_content", "cswc"),
    "temperature": ("temperature", "t"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "vertical_velocity": ("vertical_velocity", "w"),
    "vorticity": ("vorticity", "vo"),
}


class PipelineValidationError(ValueError):
    """Raised for deterministic contract, inventory, or fixture errors."""


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Materialize a filtered ERA5 pressure-level regular grid as Zarr v3.

    Parameters are supplied by the framework.  `contract_lock` may be either a
    raw DatasetContract or an envelope containing one.  `cache_dir` is treated as
    external framework-owned storage; the adapter only reads verified fixture
    inputs from it and never writes to it.
    """
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Mandatory first external action: verify fixture files before provider or
    # credential logic.  This adapter has no provider fallback.
    fixture_entries, fixture_paths = _load_and_verify_fixture(cache_root)

    contract = _extract_contract(contract_lock)
    request = _validate_contract_against_inventory(contract, inventory)

    raw = _open_fixture_dataset(fixture_paths)
    try:
        canonical = _canonicalize_dataset(raw)
        selected = _filter_dataset(canonical, request)
        publication = _prepare_publication_dataset(selected, contract, inventory, request)
        _publish_zarr_atomic(publication, out_root)
        reopened = xr.open_zarr(out_root / STORE_NAME, consolidated=True)
        try:
            _validate_publication(reopened, publication, request)
        finally:
            reopened.close()
    finally:
        raw.close()

    channels = _artifact_channels(contract, publication)
    return {
        "cache": {
            "hits": len(fixture_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [_secret_safe_fixture_key(e) for e in fixture_entries],
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": STORE_NAME,
            "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
            "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
            "channels": channels,
        },
        "warnings": [],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise PipelineValidationError("contract_lock must be a JSON object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    candidate_paths = (
        ("contract",),
        ("dataset_contract",),
        ("resolved_contract",),
        ("selected_contract",),
        ("lock", "contract"),
        ("lock", "dataset_contract"),
        ("runtime", "contract"),
    )
    for path in candidate_paths:
        node: Any = lock
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict) and node.get("schema_version") == "dataset_contract.v1":
            return node
    raise PipelineValidationError("could not locate a dataset_contract.v1 object in contract_lock")


def _validate_contract_against_inventory(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineValidationError("inventory must have schema_version dataset_inventory.v1")
    if contract.get("schema_version") != "dataset_contract.v1":
        raise PipelineValidationError("contract must have schema_version dataset_contract.v1")
    if contract.get("dataset_slug") != inventory.get("dataset_slug"):
        raise PipelineValidationError("contract dataset_slug does not match inventory")

    options = inventory.get("options") or {}
    required_option_fields = ("product_type", "variable", "year", "month", "day", "time", "pressure_level", "data_format", "download_format")
    for name in required_option_fields:
        if name not in options:
            raise PipelineValidationError(f"inventory options missing {name!r}")

    advanced = contract.get("advanced_options") or {}
    dataset_id = advanced.get("dataset_id", inventory.get("dataset_id"))
    if dataset_id != inventory.get("dataset_id"):
        raise PipelineValidationError("advanced_options.dataset_id does not match inventory dataset_id")

    data_format = advanced.get("data_format", inventory.get("defaults", {}).get("data_format", "grib"))
    download_format = advanced.get("download_format", inventory.get("defaults", {}).get("download_format", "unarchived"))
    _validate_option("data_format", data_format, options)
    _validate_option("download_format", download_format, options)

    product_type = advanced.get("product_type", (contract.get("scope") or {}).get("product_type", inventory.get("defaults", {}).get("product_type", ["reanalysis"])))
    if isinstance(product_type, str):
        product_type_values = [product_type]
    else:
        product_type_values = list(product_type or [])
    if not product_type_values:
        raise PipelineValidationError("product_type selection is empty")
    for value in product_type_values:
        _validate_option("product_type", value, options)

    field_specs = contract.get("fields") or []
    if not field_specs:
        raise PipelineValidationError("contract.fields must not be empty")

    requested_variables: list[str] = []
    pressure_by_variable: dict[str, list[str]] = {}
    pressure_order: list[str] = []
    for field in field_specs:
        name = field.get("name")
        _validate_option("variable", name, options)
        if name not in requested_variables:
            requested_variables.append(name)
        selectors = field.get("selectors") or []
        for selector in selectors:
            dim = selector.get("dimension")
            if dim != "pressure_level":
                raise PipelineValidationError(f"unsupported selector dimension {dim!r}; ERA5 pressure-level inventory supports pressure_level")
            value = str(selector.get("value"))
            _validate_option("pressure_level", value, options)
            pressure_by_variable.setdefault(name, [])
            if value not in pressure_by_variable[name]:
                pressure_by_variable[name].append(value)
            if value not in pressure_order:
                pressure_order.append(value)
        if not selectors:
            # ERA5 pressure-level variables require at least one pressure_level selector.
            raise PipelineValidationError(f"field {name!r} is missing required pressure_level selector")

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start_date = _parse_date(dr.get("start_date"), "scope.date_range.start_date")
    end_date = _parse_date(dr.get("end_date"), "scope.date_range.end_date")
    inclusive = bool(dr.get("inclusive", True))
    if end_date < start_date:
        raise PipelineValidationError("date_range.end_date is before start_date")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone", "UTC") != "UTC":
        raise PipelineValidationError("only UTC time selections are supported for this inventory")
    selected_times = list(time_scope.get("selected_times") or [])
    if not selected_times:
        raise PipelineValidationError("scope.time.selected_times must not be empty")
    for t in selected_times:
        _validate_time_string(t)
        _validate_option("time", t, options)

    timestamps = _requested_timestamps(start_date, end_date, inclusive, selected_times)
    for ts in timestamps:
        day = np.datetime_as_string(ts, unit="D")
        _validate_option("year", day[0:4], options)
        _validate_option("month", day[5:7], options)
        _validate_option("day", day[8:10], options)

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area", [90, -180, -90, 180]))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise PipelineValidationError("scope.geography.cds_area must be a four-number [north, west, south, east] list")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90):
        raise PipelineValidationError("cds_area latitude bounds must satisfy -90 <= south <= north <= 90")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineValidationError("cds_area longitude bounds must be within [-360, 360]")

    return {
        "variables": requested_variables,
        "pressure_by_variable": pressure_by_variable,
        "pressure_order": pressure_order,
        "timestamps": timestamps,
        "area": [north, west, south, east],
        "data_format": data_format,
        "download_format": download_format,
        "product_type": product_type_values,
    }


def _validate_option(name: str, value: Any, options: dict[str, Any]) -> None:
    if str(value) not in {str(v) for v in (options.get(name) or [])}:
        raise PipelineValidationError(f"invalid {name} selection: {value!r}")


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PipelineValidationError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineValidationError(f"{label} is not a valid ISO date") from exc


def _validate_time_string(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", value):
        raise PipelineValidationError(f"invalid UTC time selection {value!r}")
    hh, mm = map(int, value.split(":"))
    if hh > 23 or mm > 59:
        raise PipelineValidationError(f"invalid UTC time selection {value!r}")


def _requested_timestamps(start: date, end: date, inclusive: bool, selected_times: list[str]) -> list[pd_timestamp_type]:
    # Date-only inclusive end means the complete final UTC calendar day.  Exact
    # selected times are then applied within each selected date.
    days: list[date] = []
    cur = start
    final = end if inclusive else end - timedelta(days=1)
    while cur <= final:
        days.append(cur)
        cur += timedelta(days=1)
    result = []
    for day in days:
        for t in selected_times:
            hh, mm = map(int, t.split(":"))
            result.append(np.datetime64(datetime.combine(day, time(hh, mm), tzinfo=timezone.utc).replace(tzinfo=None), "ns"))
    # Stable de-duplication preserves contract ordering for duplicate selected_times.
    seen: set[np.datetime64] = set()
    unique = []
    for ts in result:
        if ts not in seen:
            seen.add(ts)
            unique.append(ts)
    return unique


# Runtime-only type alias without importing pandas into annotations.
pd_timestamp_type = np.datetime64


def _load_and_verify_fixture(cache_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = cache_root / FIXTURE_MANIFEST
    if not manifest_path.exists():
        raise PipelineValidationError("source fixture manifest is required at cache_dir/source_fixture_manifest.json; network acquisition is disabled")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineValidationError("source fixture manifest has unsupported schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineValidationError("source fixture manifest entries must be a non-empty list")

    verified_paths: list[Path] = []
    for entry in entries:
        rel = entry.get("relative_path")
        expected_size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
        if not isinstance(rel, str) or rel.startswith("/") or "\x00" in rel:
            raise PipelineValidationError("fixture entry relative_path must be a safe relative path")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineValidationError("fixture entry escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineValidationError(f"fixture file missing for entry_id={entry.get('entry_id')!r}")
        size = path.stat().st_size
        if not isinstance(expected_size, int) or expected_size <= 0 or size != expected_size:
            raise PipelineValidationError(f"fixture size mismatch for entry_id={entry.get('entry_id')!r}")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise PipelineValidationError("fixture sha256 must be lowercase hex")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise PipelineValidationError(f"fixture sha256 mismatch for entry_id={entry.get('entry_id')!r}")
        verified_paths.append(path)
    return entries, verified_paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _secret_safe_fixture_key(entry: dict[str, Any]) -> str:
    entry_id = str(entry.get("entry_id", "fixture"))
    safe_id = re.sub(r"[^A-Za-z0-9_.=-]+", "_", entry_id)[:80]
    sha = str(entry.get("sha256", ""))[:16]
    return f"fixture:{safe_id}:sha256:{sha}"


def _open_fixture_dataset(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    for path in paths:
        try:
            datasets.append(_open_one_dataset(path))
        except Exception as exc:  # noqa: BLE001 - summarized without leaking paths beyond basename.
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not datasets:
        raise PipelineValidationError("no fixture files could be decoded: " + "; ".join(errors))
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.merge(datasets, compat="override", combine_attrs="override", join="outer")
    except Exception as exc:  # noqa: BLE001
        for ds in datasets:
            ds.close()
        raise PipelineValidationError(f"fixture datasets could not be merged: {exc}") from exc


def _open_one_dataset(path: Path) -> xr.Dataset:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
        try:
            return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_cf=True, mask_and_scale=True)
        except Exception:
            # Some evaluator fixtures use NetCDF content with a provider-like name.
            return xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
    return xr.open_dataset(path, decode_cf=True, mask_and_scale=True)


def _canonicalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for canonical, aliases in COORD_ALIASES.items():
        if canonical in ds.dims or canonical in ds.coords:
            continue
        for alias in aliases:
            if alias in ds.dims or alias in ds.coords or alias in ds.variables:
                rename[alias] = canonical
                break
    out = ds.rename(rename) if rename else ds
    for dim in ("time", "latitude", "longitude"):
        if dim not in out.dims and dim not in out.coords:
            raise PipelineValidationError(f"source fixture is missing required coordinate/dimension {dim!r}")
    if "pressure_level" not in out.dims and "pressure_level" not in out.coords:
        raise PipelineValidationError("source fixture is missing required pressure_level coordinate/dimension")
    return out


def _filter_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    out = ds
    requested_vars = [_find_source_variable(out, v) for v in request["variables"]]
    out = out[requested_vars]
    # Rename source aliases back to CDS canonical variable names.
    var_rename = {src: canonical for src, canonical in zip(requested_vars, request["variables"], strict=True) if src != canonical}
    if var_rename:
        out = out.rename(var_rename)

    if "time" not in out.indexes:
        out = out.assign_coords(time=np.asarray(out["time"].values, dtype="datetime64[ns]"))
    else:
        out = out.assign_coords(time=np.asarray(out.indexes["time"].values, dtype="datetime64[ns]"))
    requested_times = np.asarray(request["timestamps"], dtype="datetime64[ns]")
    available_times = set(np.asarray(out["time"].values, dtype="datetime64[ns]").tolist())
    missing_times = [str(t) for t in requested_times.tolist() if t not in available_times]
    if missing_times:
        raise PipelineValidationError("source fixture does not contain all requested timestamps; first missing=" + missing_times[0])
    out = out.sel(time=requested_times)

    selected_levels = request["pressure_order"]
    pressure_values = list(out["pressure_level"].values)
    level_lookup: dict[str, Any] = {}
    for v in pressure_values:
        native = _native_scalar(v)
        keys = {str(native)}
        if isinstance(native, (int, float, np.integer, np.floating)) and np.isfinite(float(native)):
            numeric = float(native)
            keys.add(str(int(numeric)) if numeric.is_integer() else format(numeric, "g"))
        for key in keys:
            level_lookup.setdefault(key, v)
    missing_levels = [level for level in selected_levels if level not in level_lookup]
    if missing_levels:
        raise PipelineValidationError("source fixture does not contain requested pressure_level " + missing_levels[0])
    native_levels = [level_lookup[level] for level in selected_levels]
    out = out.sel(pressure_level=native_levels)

    north, west, south, east = request["area"]
    lat_values = out["latitude"].values
    lat_mask = (lat_values >= south) & (lat_values <= north)
    if not bool(np.any(lat_mask)):
        raise PipelineValidationError("area selection produced no latitude cells")
    out = out.isel(latitude=np.nonzero(lat_mask)[0])

    lon_values = out["longitude"].values
    if not _is_global_area(west, east):
        lon_mask = _longitude_mask(lon_values, west, east)
        if not bool(np.any(lon_mask)):
            raise PipelineValidationError("area selection produced no longitude cells")
        out = out.isel(longitude=np.nonzero(lon_mask)[0])

    # Stable canonical dimension order for every data variable; preserve all
    # selected dimensions even when cardinality is one.
    transposed_vars = {}
    canonical_pressure_values = list(out["pressure_level"].values)
    for name, da in out.data_vars.items():
        if "pressure_level" not in da.dims:
            levels_for_var = request["pressure_by_variable"].get(name, [])
            if len(levels_for_var) != 1:
                raise PipelineValidationError(f"source fixture variable {name!r} does not expose pressure_level as a dimension")
            if "pressure_level" in da.coords:
                da = da.drop_vars("pressure_level")
            da = da.expand_dims({"pressure_level": [level_lookup[levels_for_var[0]]]}).reindex(pressure_level=canonical_pressure_values)
        dims = [d for d in CANONICAL_DIMS if d in da.dims]
        other = [d for d in da.dims if d not in dims]
        transposed_vars[name] = da.transpose(*(dims + other))
    out = xr.Dataset(transposed_vars, coords={c: out.coords[c] for c in out.coords}, attrs=dict(out.attrs))
    return out


def _find_source_variable(ds: xr.Dataset, canonical_name: str) -> str:
    for candidate in VARIABLE_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in ds.data_vars:
            return candidate
    raise PipelineValidationError(f"source fixture is missing requested variable {canonical_name!r}")


def _native_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_global_area(west: float, east: float) -> bool:
    return abs((east - west) % 360) < 1e-9 or (west <= -180 and east >= 180) or (west <= 0 and east >= 360)


def _longitude_mask(lon_values: np.ndarray, west: float, east: float) -> np.ndarray:
    lon = np.asarray(lon_values, dtype=float)
    if np.nanmin(lon) >= 0 and west < 0:
        west = west % 360
        east = east % 360
    elif np.nanmax(lon) <= 180 and east > 180:
        west = ((west + 180) % 360) - 180
        east = ((east + 180) % 360) - 180
    if west <= east:
        return (lon >= west) & (lon <= east)
    return (lon >= west) | (lon <= east)


def _prepare_publication_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> xr.Dataset:
    keep_vars = request["variables"]
    out = ds[keep_vars].copy(deep=False)

    for var in keep_vars:
        meta = ((inventory.get("option_metadata") or {}).get("variable") or {}).get(var, {})
        attrs = dict(out[var].attrs)
        if meta.get("units") is not None:
            attrs.setdefault("units", _strip_html_units(str(meta["units"])))
        if meta.get("description") is not None:
            attrs.setdefault("description", str(meta["description"]))
        attrs.setdefault("cds_variable", var)
        selected_for_var = request["pressure_by_variable"].get(var, [])
        attrs["requested_pressure_levels_hPa"] = json.dumps(selected_for_var, separators=(",", ":"))
        out[var].attrs = attrs

    out.attrs = {
        "dataset_slug": contract.get("dataset_slug"),
        "dataset_id": (contract.get("advanced_options") or {}).get("dataset_id"),
        "source_url": contract.get("source_url"),
        "publication_format": "zarr",
        "zarr_format": 3,
        "consolidated_metadata": True,
    }

    # Prevent NetCDF/GRIB encodings (scale_factor, add_offset, _FillValue,
    # compressor choices) from silently repacking already-decoded public values.
    for name in list(out.variables):
        out[name].encoding = {}
    return out


def _strip_html_units(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("sup", "")


def _publish_zarr_atomic(ds: xr.Dataset, out_root: Path) -> None:
    final = out_root / STORE_NAME
    tmp = out_root / (STORE_NAME + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    encoding: dict[str, dict[str, Any]] = {}
    lat_count = int(ds.sizes.get("latitude", 1))
    lon_count = int(ds.sizes.get("longitude", 1))
    for name, da in ds.data_vars.items():
        if {"time", "pressure_level", "latitude", "longitude"}.issubset(da.dims):
            encoding[name] = {"chunks": tuple(1 if d == "time" else 1 if d == "pressure_level" else lat_count if d == "latitude" else lon_count if d == "longitude" else da.sizes[d] for d in da.dims)}
        elif {"time", "latitude", "longitude"}.issubset(da.dims):
            encoding[name] = {"chunks": tuple(1 if d == "time" else lat_count if d == "latitude" else lon_count if d == "longitude" else da.sizes[d] for d in da.dims)}

    try:
        ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=3, encoding=encoding)
        # Reopen before publication so a corrupt stage is never exposed as final.
        check = xr.open_zarr(tmp, consolidated=True)
        check.close()
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def _validate_publication(reopened: xr.Dataset, expected: xr.Dataset, request: dict[str, Any]) -> None:
    for dim in ("time", "pressure_level", "latitude", "longitude"):
        if dim not in reopened.dims:
            raise PipelineValidationError(f"published Zarr is missing dimension {dim!r}")
        if int(reopened.sizes[dim]) != int(expected.sizes[dim]):
            raise PipelineValidationError(f"published Zarr dimension {dim!r} has wrong length")
    for var in request["variables"]:
        if var not in reopened.data_vars:
            raise PipelineValidationError(f"published Zarr missing data variable {var!r}")
        dims = tuple(reopened[var].dims)
        if dims[:4] != ("time", "pressure_level", "latitude", "longitude"):
            raise PipelineValidationError(f"published variable {var!r} does not use canonical leading dimensions")


def _artifact_channels(contract: dict[str, Any], ds: xr.Dataset) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for field in contract.get("fields") or []:
        name = field["name"]
        selectors = field.get("selectors") or []
        selector_map = {str(s["dimension"]): str(s["value"]) for s in selectors}
        channels.append({
            "field_id": _canonical_field_id(name, selectors),
            "array_path": name,
            "selectors": selector_map,
            "selector_coordinate_paths": {"pressure_level": "pressure_level"} if "pressure_level" in ds.dims else {},
        })
    return channels


def _canonical_field_id(name: str, selectors: Iterable[dict[str, Any]]) -> str:
    selectors = list(selectors)
    if not selectors:
        return name
    parts = []
    for selector in selectors:
        parts.append(f"{selector['dimension']}={json.dumps(str(selector['value']), ensure_ascii=False, separators=(',', ':'))}")
    return f"{name}[{','.join(parts)}]"
