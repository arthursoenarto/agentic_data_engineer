"""Standalone ERA5 pressure-level family adapter.

Public entry point:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The adapter is deterministic and offline-first.  If cache_dir contains
source_fixture_manifest.json, every listed raw file is size/hash verified before
any acquisition/provider path is considered, and those verified files are used as
source-of-truth inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p000-baseline-r2"
EXPECTED_DATASET_SLUG = "reanalysis_era5_pressure_levels"
EXPECTED_DATASET_ID = "reanalysis-era5-pressure-levels"
FIXTURE_MANIFEST = "source_fixture_manifest.json"
STORE_NAME = "dataset.zarr"

SHORT_NAME_ALIASES = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "vo": "vorticity",
    "d": "divergence",
    "r": "relative_humidity",
    "q": "specific_humidity",
    "cc": "fraction_of_cloud_cover",
    "pv": "potential_vorticity",
    "o3": "ozone_mass_mixing_ratio",
    "ciwc": "specific_cloud_ice_water_content",
    "clwc": "specific_cloud_liquid_water_content",
    "crwc": "specific_rain_water_content",
    "cswc": "specific_snow_water_content",
}

DIM_ALIASES = {
    "valid_time": "time",
    "initial_time0_hours": "time",
    "forecast_time0": "time",
    "lat": "latitude",
    "lon": "longitude",
    "isobaricInhPa": "pressure_level",
    "isobaricInPa": "pressure_level_pa",
    "level": "pressure_level",
    "plev": "pressure_level",
}


class ContractError(ValueError):
    """Raised when the runtime contract is not valid for the inventory."""


class SourceFixtureError(RuntimeError):
    """Raised when the local fixture manifest or raw files are invalid."""


@dataclass(frozen=True)
class RequestedField:
    name: str
    selectors: tuple[tuple[str, str], ...]
    field_id: str
    array_name: str


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Materialize a selected ERA5 pressure-level lock into consolidated Zarr v3.

    Parameters are supplied by the framework.  This function writes only beneath
    output_dir and treats cache_dir as external input/cache.  Network acquisition
    is intentionally not implemented for this pilot; a verified local fixture is
    required for deterministic, credential-free execution.
    """
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()

    fixture_files, cache_evidence = _verify_source_fixture(cache_root)
    contract = _find_contract(contract_lock)
    request = _validate_contract(contract, inventory)

    if not fixture_files:
        raise SourceFixtureError(
            "No verified local source fixture was found. Place source_fixture_manifest.json at cache_dir root."
        )

    source = _open_verified_sources(fixture_files)
    try:
        normalized = _normalize_dataset(source)
        filtered = _filter_dataset(normalized, request)
        public = _build_public_dataset(filtered, request, inventory)
    finally:
        source.close()

    _publish_zarr(public, out_root / STORE_NAME)
    _validate_published(out_root / STORE_NAME, request)

    channels = []
    for rf in request["fields"]:
        selector_paths = {}
        for dim, _value in rf.selectors:
            unique_dim = _selector_dim_name(rf.array_name, dim)
            selector_paths[dim] = unique_dim
        channels.append(
            {
                "field_id": rf.field_id,
                "array_path": rf.array_name,
                "selectors": {k: v for k, v in rf.selectors},
                "selector_coordinate_paths": selector_paths,
            }
        )

    return {
        "cache": cache_evidence,
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


def _find_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    if isinstance(lock, dict):
        for key in ("contract", "dataset_contract", "selected_contract", "runtime_contract"):
            value = lock.get(key)
            if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
                return value
        for value in lock.values():
            if isinstance(value, (dict, list)):
                try:
                    return _find_contract(value)
                except ContractError:
                    pass
    if isinstance(lock, list):
        for value in lock:
            try:
                return _find_contract(value)
            except ContractError:
                pass
    raise ContractError("Could not locate a dataset_contract.v1 object in contract_lock")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ContractError("inventory.schema_version must be dataset_inventory.v1")
    if inventory.get("dataset_slug") != EXPECTED_DATASET_SLUG:
        raise ContractError("inventory dataset_slug is not supported by this adapter")
    if inventory.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ContractError("inventory dataset_id is not supported by this adapter")
    if contract.get("dataset_slug") != inventory.get("dataset_slug"):
        raise ContractError("contract dataset_slug does not match inventory")
    if contract.get("human_confirmed") is not True:
        raise ContractError("contract must be human_confirmed")

    options = inventory.get("options", {})
    advanced = contract.get("advanced_options") or {}
    dataset_id = advanced.get("dataset_id", inventory.get("dataset_id"))
    if dataset_id != inventory.get("dataset_id"):
        raise ContractError("advanced_options.dataset_id does not match inventory")

    data_format = advanced.get("data_format", inventory.get("defaults", {}).get("data_format"))
    download_format = advanced.get("download_format", inventory.get("defaults", {}).get("download_format"))
    product_type = _as_list(advanced.get("product_type", contract.get("scope", {}).get("product_type", inventory.get("defaults", {}).get("product_type"))))
    if data_format not in options.get("data_format", []):
        raise ContractError(f"unsupported data_format: {data_format!r}")
    if download_format not in options.get("download_format", []):
        raise ContractError(f"unsupported download_format: {download_format!r}")
    _require_options("product_type", product_type, options)

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "scope.date_range.start_date")
    end = _parse_date(dr.get("end_date"), "scope.date_range.end_date")
    if end < start:
        raise ContractError("scope.date_range.end_date must not precede start_date")
    years = [f"{y:04d}" for y in range(start.year, end.year + 1)]
    months = sorted({f"{d.month:02d}" for d in pd.date_range(start, end, freq="D").date})
    days = sorted({f"{d.day:02d}" for d in pd.date_range(start, end, freq="D").date})
    _require_options("year", years, options)
    _require_options("month", months, options)
    _require_options("day", days, options)

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone", "UTC") != "UTC":
        raise ContractError("only UTC time selections are valid for this inventory")
    selected_times = time_scope.get("selected_times") or []
    if not selected_times:
        raise ContractError("scope.time.selected_times must not be empty")
    _require_options("time", selected_times, options)
    for t in selected_times:
        _parse_hhmm(t)

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise ContractError("scope.geography.cds_area must be four numeric values [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90):
        raise ContractError("latitude bounds must satisfy -90 <= south <= north <= 90")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ContractError("longitude bounds must be within [-360, 360]")

    fields = []
    seen = set()
    for field in contract.get("fields") or []:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise ContractError(f"unsupported variable: {name!r}")
        selectors = []
        for sel in field.get("selectors") or []:
            dim = sel.get("dimension")
            value = str(sel.get("value"))
            if dim != "pressure_level":
                raise ContractError(f"unsupported selector dimension for ERA5 pressure levels: {dim!r}")
            if value not in options.get("pressure_level", []):
                raise ContractError(f"unsupported pressure_level: {value!r}")
            selectors.append((dim, value))
        field_id = _canonical_field_id(name, selectors)
        if field_id in seen:
            raise ContractError(f"duplicate requested field selector: {field_id}")
        seen.add(field_id)
        fields.append(RequestedField(name=name, selectors=tuple(selectors), field_id=field_id, array_name=_safe_array_name(field_id)))
    if not fields:
        raise ContractError("contract.fields must contain at least one field")

    timestamps = _requested_timestamps(start, end, bool(dr.get("inclusive", True)), selected_times)
    if not timestamps:
        raise ContractError("runtime date/time selectors produced no timestamps")

    return {
        "data_format": data_format,
        "download_format": download_format,
        "product_type": product_type,
        "start_date": start,
        "end_date": end,
        "inclusive_end": bool(dr.get("inclusive", True)),
        "selected_times": list(selected_times),
        "timestamps": timestamps,
        "area": [north, west, south, east],
        "fields": fields,
    }


def _verify_source_fixture(cache_root: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest_path = cache_root / FIXTURE_MANIFEST
    if not manifest_path.exists():
        return [], {"hits": 0, "misses": 1, "acquired": 0, "reused_keys": [], "acquired_keys": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise SourceFixtureError("source_fixture_manifest.json has unsupported schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SourceFixtureError("source fixture manifest must contain at least one entry")

    files: list[Path] = []
    keys: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise SourceFixtureError("fixture relative_path must be a relative string")
        path = (cache_root / rel).resolve()
        if not _is_relative_to(path, cache_root):
            raise SourceFixtureError(f"fixture path escapes cache_dir: {rel}")
        if not path.is_file():
            raise SourceFixtureError(f"fixture file is missing: {rel}")
        expected_size = int(entry.get("size_bytes"))
        if expected_size <= 0:
            raise SourceFixtureError(f"fixture size_bytes must be positive for {rel}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise SourceFixtureError(f"fixture size mismatch for {rel}: {actual_size} != {expected_size}")
        expected_hash = str(entry.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise SourceFixtureError(f"fixture sha256 is invalid for {rel}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise SourceFixtureError(f"fixture sha256 mismatch for {rel}")
        entry_id = str(entry.get("entry_id") or rel)
        keys.append(f"fixture:{entry_id}:sha256:{actual_hash[:16]}")
        files.append(path)
    return files, {"hits": len(files), "misses": 0, "acquired": 0, "reused_keys": keys, "acquired_keys": []}


def _open_verified_sources(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    opened: list[xr.Dataset] = []
    try:
        for path in paths:
            suffixes = "".join(path.suffixes).lower()
            if suffixes.endswith((".nc", ".nc4", ".netcdf")):
                ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
            elif suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
                ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
            else:
                # Try CF NetCDF first, then GRIB.  Errors include the path for diagnosis.
                try:
                    ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
                except Exception:
                    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
            opened.append(ds)
            datasets.append(ds.load())
        if len(datasets) == 1:
            return datasets[0]
        try:
            return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
        except Exception:
            return xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts")
    finally:
        for ds in opened:
            ds.close()


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {name: DIM_ALIASES[name] for name in list(ds.dims) + list(ds.coords) if name in DIM_ALIASES and DIM_ALIASES[name] not in ds}
    out = ds.rename(rename) if rename else ds.copy()
    var_rename = {}
    for name in out.data_vars:
        canonical = SHORT_NAME_ALIASES.get(name, name)
        if canonical != name and canonical not in out.data_vars:
            var_rename[name] = canonical
    if var_rename:
        out = out.rename(var_rename)
    if "pressure_level_pa" in out.coords and "pressure_level" not in out.coords:
        out = out.assign_coords(pressure_level=("pressure_level_pa", np.asarray(out["pressure_level_pa"].values, dtype=float) / 100.0))
        out = out.swap_dims({"pressure_level_pa": "pressure_level"})
    if "time" in out.coords:
        out = out.assign_coords(time=pd.to_datetime(out["time"].values).tz_localize(None).to_numpy(dtype="datetime64[ns]"))
    return out


def _filter_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    required_dims = ["time", "latitude", "longitude"]
    for dim in required_dims:
        if dim not in ds.coords and dim not in ds.dims:
            raise SourceFixtureError(f"source fixture is missing required coordinate/dimension {dim!r}")

    requested_times = pd.DatetimeIndex(request["timestamps"]).tz_localize(None)
    source_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).tz_localize(None)
    missing = requested_times.difference(source_times)
    if len(missing):
        raise SourceFixtureError(f"source fixture does not cover requested timestamps; first missing={missing[0].isoformat()}")
    out = ds.sel(time=requested_times.to_numpy(dtype="datetime64[ns]"))

    north, west, south, east = request["area"]
    lat = out["latitude"]
    lat_mask = (lat >= south) & (lat <= north)
    out = out.where(lat_mask, drop=True)

    lon = out["longitude"]
    lon_values = np.asarray(lon.values)
    if lon_values.size:
        if np.nanmin(lon_values) >= 0 and (west < 0 or east < 0):
            west_n = west % 360
            east_n = east % 360
            lon_cmp = lon % 360
        else:
            west_n = west
            east_n = east
            lon_cmp = lon
        if abs((east - west)) >= 360 or (west <= -180 and east >= 180):
            lon_mask = xr.ones_like(lon, dtype=bool)
        elif west_n <= east_n:
            lon_mask = (lon_cmp >= west_n) & (lon_cmp <= east_n)
        else:
            lon_mask = (lon_cmp >= west_n) | (lon_cmp <= east_n)
        out = out.where(lon_mask, drop=True)
    if out.sizes.get("latitude", 0) == 0 or out.sizes.get("longitude", 0) == 0:
        raise SourceFixtureError("geographic filter produced an empty grid")
    return out


def _build_public_dataset(ds: xr.Dataset, request: dict[str, Any], inventory: dict[str, Any]) -> xr.Dataset:
    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    }
    variable_meta = inventory.get("option_metadata", {}).get("variable", {})

    for rf in request["fields"]:
        if rf.name not in ds.data_vars:
            raise SourceFixtureError(f"source fixture is missing requested variable {rf.name!r}")
        da = ds[rf.name]
        for dim, value in rf.selectors:
            native_dim = dim if dim in da.dims or dim in da.coords else None
            if native_dim is None:
                for candidate, canonical in DIM_ALIASES.items():
                    if canonical == dim and (candidate in da.dims or candidate in da.coords):
                        native_dim = candidate
                        break
            unique_dim = _selector_dim_name(rf.array_name, dim)
            if native_dim is None:
                requested_values = {
                    selected
                    for other in request["fields"]
                    if other.name == rf.name
                    for selected_dim, selected in other.selectors
                    if selected_dim == dim
                }
                if dim != "pressure_level" or requested_values != {value}:
                    raise SourceFixtureError(f"source variable {rf.name!r} is missing selector coordinate {dim!r}")
                native_value = value
                da = da.expand_dims({unique_dim: [native_value]})
                coord = da[unique_dim].copy()
            else:
                coord_source = da[native_dim] if native_dim in da.coords else ds[native_dim]
                native_values = np.atleast_1d(np.asarray(coord_source.values))
                match_index = next((i for i, native in enumerate(native_values.tolist()) if _selector_value_matches(native, value)), None)
                if match_index is None:
                    raise SourceFixtureError(f"source variable {rf.name!r} is missing {dim}={value}")
                native_value = native_values[match_index]
                if native_dim in da.dims:
                    da = da.sel({native_dim: [native_value]})
                    da = da.rename({native_dim: unique_dim})
                else:
                    da = da.drop_vars(native_dim)
                    da = da.expand_dims({unique_dim: [native_value]})
                coord = da[unique_dim].copy()
                coord.attrs.update(coord_source.attrs)
            coord.attrs.setdefault("units", inventory.get("option_units", {}).get(dim, ""))
            coord.attrs["contract_selector_dimension"] = dim
            coords[unique_dim] = coord
        da = da.transpose(...)
        da.name = rf.array_name
        da.attrs = dict(da.attrs)
        da.attrs["field_id"] = rf.field_id
        da.attrs["canonical_variable"] = rf.name
        da.attrs["contract_selectors_json"] = json.dumps({k: v for k, v in rf.selectors}, sort_keys=True)
        meta = variable_meta.get(rf.name, {})
        if meta.get("units") and not da.attrs.get("units"):
            da.attrs["units"] = _strip_html_units(meta["units"])
        if meta.get("description") and not da.attrs.get("description"):
            da.attrs["description"] = meta["description"]
        data_vars[rf.array_name] = da

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        "dataset_slug": EXPECTED_DATASET_SLUG,
        "dataset_id": EXPECTED_DATASET_ID,
        "provider": "ECMWF",
        "publication_format": "zarr",
        "zarr_format": "3",
        "consolidated_metadata": "true",
    })
    # CF packing/missing-value conventions have already been decoded by xarray.
    # Clear source encodings so public Zarr cannot silently repack values.
    for name in list(out.data_vars) + list(out.coords):
        out[name].encoding = {}
    return out


def _publish_zarr(ds: xr.Dataset, final_store: Path) -> None:
    final_store.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = final_store.parent
    tmp = Path(tempfile.mkdtemp(prefix=f".{final_store.name}.", suffix=".tmp", dir=tmp_parent))
    try:
        ds.to_zarr(tmp, mode="w", zarr_format=3, consolidated=True)
        _replace_dir(tmp, final_store)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _validate_published(store: Path, request: dict[str, Any]) -> None:
    if not store.is_dir():
        raise RuntimeError("published Zarr store was not created")
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        for coord in ("time", "latitude", "longitude"):
            if coord not in reopened.coords:
                raise RuntimeError(f"published Zarr is missing coordinate {coord}")
        expected_times = pd.DatetimeIndex(request["timestamps"]).tz_localize(None)
        actual_times = pd.DatetimeIndex(pd.to_datetime(reopened["time"].values)).tz_localize(None)
        if not actual_times.equals(expected_times):
            raise RuntimeError("published Zarr time coordinate does not match runtime lock")
        for rf in request["fields"]:
            if rf.array_name not in reopened.data_vars:
                raise RuntimeError(f"published Zarr is missing array {rf.array_name}")
            for dim, value in rf.selectors:
                unique_dim = _selector_dim_name(rf.array_name, dim)
                if unique_dim not in reopened[rf.array_name].dims:
                    raise RuntimeError(f"published array {rf.array_name} lost selector dimension {dim}")
                vals = np.atleast_1d(np.asarray(reopened[unique_dim].values)).tolist()
                if len(vals) != 1 or not _selector_value_matches(vals[0], value):
                    raise RuntimeError(f"published selector coordinate {unique_dim} does not match contract")
    finally:
        reopened.close()


def _requested_timestamps(start: date, end: date, inclusive: bool, selected_times: Iterable[str]) -> list[pd.Timestamp]:
    end_for_days = end if inclusive else (pd.Timestamp(end) - pd.Timedelta(days=1)).date()
    if end_for_days < start:
        return []
    out: list[pd.Timestamp] = []
    for day in pd.date_range(start, end_for_days, freq="D"):
        for hhmm in selected_times:
            h, m = _parse_hhmm(hhmm)
            out.append(pd.Timestamp(datetime.combine(day.date(), time(h, m), tzinfo=timezone.utc)).tz_convert(None))
    return sorted(out)


def _canonical_field_id(name: str, selectors: Iterable[tuple[str, str]]) -> str:
    selectors = list(selectors)
    if not selectors:
        return name
    body = ",".join(f"{dim}={json.dumps(str(value), ensure_ascii=False)}" for dim, value in selectors)
    return f"{name}[{body}]"


def _safe_array_name(field_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", field_id).strip("_")
    if not safe or safe[0].isdigit():
        safe = "field_" + safe
    return safe


def _selector_dim_name(array_name: str, dim: str) -> str:
    return f"{dim}__{array_name}"


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{label} must be YYYY-MM-DD") from exc


def _parse_hhmm(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", value):
        raise ContractError(f"invalid UTC time selector: {value!r}")
    h, m = map(int, value.split(":"))
    if h > 23:
        raise ContractError(f"invalid UTC time selector: {value!r}")
    return h, m


def _require_options(name: str, values: Iterable[str], options: dict[str, Any]) -> None:
    allowed = set(options.get(name, []))
    for value in values:
        if str(value) not in allowed:
            raise ContractError(f"unsupported {name}: {value!r}")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _selector_value_matches(native: Any, requested: str) -> bool:
    if str(native) == str(requested):
        return True
    try:
        return bool(np.isclose(float(native), float(requested), rtol=0.0, atol=1e-9))
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _replace_dir(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.replace(src, dst)
        return
    backup = dst.with_name(dst.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    try:
        os.replace(dst, backup)
        os.replace(src, dst)
    except OSError:
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(src, dst)
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _strip_html_units(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("sup", "").strip()
