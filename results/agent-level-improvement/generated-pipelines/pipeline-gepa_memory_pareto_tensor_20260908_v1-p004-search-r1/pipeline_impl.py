"""ERA5 pressure-level family adapter.

Entry point: run_pipeline(contract_lock, inventory, cache_dir, output_dir)

This module is intentionally standalone: it performs no network access and never
constructs provider clients when a complete source_fixture_manifest.json is
present. The local fixture is the authoritative raw input for evaluator runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p004-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "era5_pressure_levels.zarr"
SOURCE_MANIFEST = "source_fixture_manifest.json"
BALANCED_BLOSC_CLEVEL = 3

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
}
_TIME_ALIASES = ("time", "valid_time")
_LAT_ALIASES = ("latitude", "lat")
_LON_ALIASES = ("longitude", "lon")
_PRESSURE_ALIASES = ("pressure_level", "isobaricInhPa", "level", "plev")


class PipelineError(ValueError):
    """Secret-safe validation/materialization error."""


@dataclass(frozen=True)
class RequestedField:
    name: str
    selectors: tuple[tuple[str, str], ...]
    field_id: str


@dataclass(frozen=True)
class VerifiedFixture:
    files: tuple[Path, ...]
    evidence: tuple[str, ...]


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a runtime lock, filter verified local ERA5 fixture files, and publish Zarr v3.

    Parameters are supplied by the framework. The function returns the
    family_pipeline_interface.v3 implementation_return object.
    """
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    contract = _extract_contract(contract_lock)
    _validate_inventory(inventory)
    requested = _validate_contract(contract, inventory)

    fixture = _verify_source_fixture(cache_root)
    source = _open_fixture_dataset(fixture.files)
    try:
        filtered, channel_meta = _filter_dataset(source, contract, requested, inventory)
        artifact = _publish_zarr(filtered, channel_meta, out_root)
    finally:
        source.close()

    return {
        "cache": {
            "hits": len(fixture.files),
            "misses": 0,
            "acquired": 0,
            "reused_keys": list(fixture.evidence),
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise PipelineError("contract_lock must be a mapping")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
            return value
    nested = lock.get("lock")
    if isinstance(nested, dict):
        for key in ("contract", "dataset_contract", "selected_contract"):
            value = nested.get(key)
            if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
                return value
    contracts = lock.get("contracts")
    if isinstance(contracts, list):
        active_id = lock.get("active_contract_id") or lock.get("selected_contract_id")
        candidates = [c for c in contracts if isinstance(c, dict) and c.get("schema_version") == "dataset_contract.v1"]
        if active_id is not None:
            for c in candidates:
                if c.get("contract_id") == active_id or c.get("id") == active_id:
                    return c
        if len(candidates) == 1:
            return candidates[0]
    raise PipelineError("no dataset_contract.v1 found in contract_lock envelope")


def _validate_inventory(inventory: dict[str, Any]) -> None:
    if not isinstance(inventory, dict) or inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("inventory must be dataset_inventory.v1")
    if inventory.get("dataset_slug") != DATASET_SLUG:
        raise PipelineError("inventory dataset_slug does not match adapter family")
    if inventory.get("dataset_id") not in (None, DATASET_ID):
        raise PipelineError("inventory dataset_id does not match ERA5 pressure-level dataset")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> list[RequestedField]:
    if contract.get("dataset_slug") != inventory.get("dataset_slug"):
        raise PipelineError("contract dataset_slug is not supported by this inventory")
    if not contract.get("human_confirmed", False):
        raise PipelineError("contract must be human_confirmed")

    options = inventory.get("options", {})
    advanced = contract.get("advanced_options") or {}
    if advanced.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise PipelineError("advanced_options.dataset_id is not supported")
    _validate_option("data_format", advanced.get("data_format", inventory.get("defaults", {}).get("data_format", "grib")), options)
    _validate_option("download_format", advanced.get("download_format", inventory.get("defaults", {}).get("download_format", "unarchived")), options)

    product_type = contract.get("scope", {}).get("product_type") or (advanced.get("product_type") or ["reanalysis"])
    product_values = product_type if isinstance(product_type, list) else [product_type]
    if product_values != ["reanalysis"]:
        raise PipelineError("this publication policy supports only product_type=reanalysis")
    for value in product_values:
        _validate_option("product_type", value, options)

    selected_times = contract.get("scope", {}).get("time", {}).get("selected_times")
    if not selected_times:
        raise PipelineError("scope.time.selected_times is required")
    for t in selected_times:
        _validate_option("time", t, options)
        _parse_hhmm(t)

    start, end, _inclusive = _contract_dates(contract)
    if end < start:
        raise PipelineError("date_range end_date precedes start_date")
    for d in pd.date_range(start=start, end=end, freq="D"):
        _validate_option("year", f"{d.year:04d}", options)
        _validate_option("month", f"{d.month:02d}", options)
        _validate_option("day", f"{d.day:02d}", options)

    area = contract.get("scope", {}).get("geography", {}).get("cds_area", inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)):
        raise PipelineError("scope.geography.cds_area must be [north, west, south, east]")
    north, west, south, east = map(float, area)
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("scope.geography.cds_area is outside supported geographic bounds")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise PipelineError("contract.fields must contain at least one field")
    requested: list[RequestedField] = []
    seen: set[str] = set()
    for field in fields:
        if not isinstance(field, dict):
            raise PipelineError("each field must be an object")
        name = field.get("name")
        _validate_option("variable", name, options)
        selectors_raw = field.get("selectors") or []
        selectors: list[tuple[str, str]] = []
        for sel in selectors_raw:
            dim = sel.get("dimension")
            value = str(sel.get("value"))
            if dim != "pressure_level":
                raise PipelineError("only pressure_level selectors are supported for this inventory")
            _validate_option("pressure_level", value, options)
            unit = sel.get("unit")
            expected_unit = inventory.get("option_units", {}).get("pressure_level")
            if unit is not None and expected_unit is not None and unit != expected_unit:
                raise PipelineError("pressure_level selector unit does not match inventory")
            selectors.append((dim, value))
        if not selectors:
            raise PipelineError("ERA5 pressure-level fields require a pressure_level selector")
        field_id = _canonical_field_id(name, selectors)
        if field_id in seen:
            raise PipelineError(f"duplicate requested field {field_id}")
        seen.add(field_id)
        requested.append(RequestedField(name=name, selectors=tuple(selectors), field_id=field_id))
    return requested


def _validate_option(name: str, value: Any, options: dict[str, Any]) -> None:
    vals = options.get(name)
    if vals is None:
        return
    if value not in vals:
        raise PipelineError(f"invalid {name} option: {value!r}")


def _canonical_field_id(name: str, selectors: Iterable[tuple[str, str]]) -> str:
    selectors = tuple(selectors)
    if not selectors:
        return name
    body = ",".join(f"{dim}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}" for dim, value in selectors)
    return f"{name}[{body}]"


def _parse_hhmm(value: str) -> time:
    try:
        h, m = value.split(":", 1)
        return time(int(h), int(m), tzinfo=timezone.utc)
    except Exception as exc:  # pragma: no cover - defensive message normalization
        raise PipelineError(f"invalid time option: {value!r}") from exc


def _contract_dates(contract: dict[str, Any]) -> tuple[date, date, bool]:
    dr = contract.get("scope", {}).get("date_range", {})
    try:
        start = date.fromisoformat(dr["start_date"])
        end = date.fromisoformat(dr["end_date"])
    except Exception as exc:
        raise PipelineError("scope.date_range.start_date and end_date must be ISO dates") from exc
    return start, end, bool(dr.get("inclusive", True))


def _requested_datetimes(contract: dict[str, Any]) -> pd.DatetimeIndex:
    start, end, inclusive = _contract_dates(contract)
    if not inclusive:
        end = end - pd.Timedelta(days=1)
    selected_times = [_parse_hhmm(t) for t in contract["scope"]["time"]["selected_times"]]
    values: list[pd.Timestamp] = []
    for day in pd.date_range(start=start, end=end, freq="D"):
        for hhmm in selected_times:
            values.append(pd.Timestamp(datetime(day.year, day.month, day.day, hhmm.hour, hhmm.minute, tzinfo=timezone.utc)).tz_convert(None))
    return pd.DatetimeIndex(values).sort_values()


def _verify_source_fixture(cache_root: Path) -> VerifiedFixture:
    manifest_path = cache_root / SOURCE_MANIFEST
    if not manifest_path.exists():
        raise PipelineError("complete local source fixture is required; no network acquisition is implemented")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("source fixture manifest schema_version is not supported")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("source fixture manifest must contain entries")
    files: list[Path] = []
    evidence: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise PipelineError("manifest entry relative_path must be relative")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineError("manifest entry escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineError("manifest entry file is missing")
        expected_size = int(entry.get("size_bytes"))
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise PipelineError("manifest entry size_bytes verification failed")
        expected_sha = str(entry.get("sha256", ""))
        actual_sha = _sha256(path)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or actual_sha != expected_sha:
            raise PipelineError("manifest entry sha256 verification failed")
        files.append(path)
        evidence.append(f"fixture:{entry.get('entry_id', rel)}:{actual_sha[:16]}:{actual_size}")
    return VerifiedFixture(files=tuple(files), evidence=tuple(evidence))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_fixture_dataset(files: tuple[Path, ...]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in files:
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
            ds = xr.open_dataset(path, engine="cfgrib", mask_and_scale=True, decode_times=True, backend_kwargs={"indexpath": ""})
        elif suffixes.endswith((".nc", ".nc4", ".netcdf")):
            try:
                ds = xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True, decode_times=True)
            except Exception:
                ds = xr.open_dataset(path, mask_and_scale=True, decode_times=True)
        else:
            try:
                ds = xr.open_dataset(path, engine="cfgrib", mask_and_scale=True, decode_times=True, backend_kwargs={"indexpath": ""})
            except Exception:
                ds = xr.open_dataset(path, mask_and_scale=True, decode_times=True)
        datasets.append(ds.load())
        ds.close()
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts")


def _filter_dataset(source: xr.Dataset, contract: dict[str, Any], requested: list[RequestedField], inventory: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    time_dim = _find_name(source.variables.keys() | source.dims.keys(), _TIME_ALIASES, "time")
    lat_dim = _find_name(source.variables.keys() | source.dims.keys(), _LAT_ALIASES, "latitude")
    lon_dim = _find_name(source.variables.keys() | source.dims.keys(), _LON_ALIASES, "longitude")
    pressure_dim = _find_name(source.variables.keys() | source.dims.keys(), _PRESSURE_ALIASES, "pressure_level")

    req_times = _requested_datetimes(contract)
    time_index = pd.DatetimeIndex(pd.to_datetime(source[time_dim].values))
    time_positions = _positions_for_values(time_index, req_times, "time")

    area = contract.get("scope", {}).get("geography", {}).get("cds_area", [90, -180, -90, 180])
    lat_positions = _latitude_positions(source[lat_dim].values, float(area[0]), float(area[2]))
    lon_positions = _longitude_positions(source[lon_dim].values, float(area[1]), float(area[3]))

    by_variable: dict[str, list[RequestedField]] = {}
    for rf in requested:
        by_variable.setdefault(rf.name, []).append(rf)

    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": source[time_dim].values[time_positions],
        "latitude": source[lat_dim].values[lat_positions],
        "longitude": source[lon_dim].values[lon_positions],
    }
    channel_meta: list[dict[str, Any]] = []

    # Use per-variable pressure dimensions when selector sets differ. This avoids
    # publishing unrequested pressure planes while keeping selector coordinates as
    # real Zarr arrays/dimensions.
    selected_level_sets = {name: tuple(_selector_values(fields)) for name, fields in by_variable.items()}
    shared_pressure_dim = len(set(selected_level_sets.values())) == 1

    for variable_name, fields in by_variable.items():
        src_var = _find_source_variable(source, variable_name)
        da = source[src_var]
        if time_dim in da.dims:
            da = da.isel({time_dim: time_positions})
        if lat_dim in da.dims:
            da = da.isel({lat_dim: lat_positions})
        if lon_dim in da.dims:
            da = da.isel({lon_dim: lon_positions})
        levels = _selector_values(fields)
        level_positions = _positions_for_selector(source[pressure_dim].values, levels, "pressure_level")
        if pressure_dim in da.dims:
            da = da.isel({pressure_dim: level_positions})
        elif len(levels) == 1:
            scalar_level = da.coords.get(pressure_dim)
            if scalar_level is not None and getattr(scalar_level, "ndim", 0) == 0 and _normalize_selector_value(scalar_level.item()) != _normalize_selector_value(levels[0]):
                raise PipelineError(f"source variable {src_var!r} pressure_level does not match requested selector")
            da = da.expand_dims({pressure_dim: source[pressure_dim].values[level_positions]})
        else:
            raise PipelineError(f"source variable {src_var!r} has no pressure_level dimension")

        out_pressure_dim = "pressure_level" if shared_pressure_dim else f"pressure_level__{_safe_name(variable_name)}"
        rename: dict[str, str] = {}
        if time_dim in da.dims:
            rename[time_dim] = "time"
        if lat_dim in da.dims:
            rename[lat_dim] = "latitude"
        if lon_dim in da.dims:
            rename[lon_dim] = "longitude"
        rename[pressure_dim] = out_pressure_dim
        da = da.rename(rename)
        da = da.transpose("time", out_pressure_dim, "latitude", "longitude", missing_dims="ignore")
        da.name = variable_name
        da.attrs = _merged_variable_attrs(da.attrs, variable_name, inventory)
        da.encoding = {}
        data_vars[variable_name] = da
        coords[out_pressure_dim] = source[pressure_dim].values[level_positions]

        for rf in fields:
            channel_meta.append({
                "field_id": rf.field_id,
                "array_path": variable_name,
                "selectors": {dim: value for dim, value in rf.selectors},
                "selector_coordinate_paths": {"pressure_level": out_pressure_dim},
            })

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=_dataset_attrs(contract, inventory))
    for cname in ds.coords:
        ds[cname].encoding = {}
    for c in [c for c in ds.coords if c.startswith("pressure_level")]:
        ds[c].attrs.setdefault("units", inventory.get("option_units", {}).get("pressure_level", "hPa"))
        ds[c].attrs.setdefault("long_name", "pressure level")
    ds["time"].attrs.setdefault("timezone", "UTC")
    ds["latitude"].attrs.setdefault("units", "degrees_north")
    ds["longitude"].attrs.setdefault("units", "degrees_east")
    _assert_filtered_semantics(source, ds, contract, requested, channel_meta, time_dim, lat_dim, lon_dim, pressure_dim, time_positions, lat_positions, lon_positions)
    return ds, channel_meta


def _find_name(names: Iterable[str], aliases: Iterable[str], logical: str) -> str:
    names_list = list(names)
    for alias in aliases:
        if alias in names_list:
            return alias
    lower = {n.lower(): n for n in names_list}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise PipelineError(f"source fixture is missing {logical} coordinate/dimension")


def _find_source_variable(ds: xr.Dataset, variable_name: str) -> str:
    candidates = _FIELD_ALIASES.get(variable_name, (variable_name,))
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if attrs.get("standard_name") == variable_name or attrs.get("long_name") == variable_name:
            return var
    raise PipelineError(f"source fixture is missing requested variable {variable_name!r}")


def _positions_for_values(source_index: pd.DatetimeIndex, requested_index: pd.DatetimeIndex, logical: str) -> np.ndarray:
    positions: list[int] = []
    source_map = {pd.Timestamp(v).to_datetime64(): i for i, v in enumerate(source_index)}
    missing: list[str] = []
    for value in requested_index:
        key = pd.Timestamp(value).to_datetime64()
        if key not in source_map:
            missing.append(str(value))
        else:
            positions.append(source_map[key])
    if missing:
        raise PipelineError(f"source fixture is missing requested {logical} values")
    return np.asarray(positions, dtype=np.int64)


def _selector_values(fields: list[RequestedField]) -> list[str]:
    values: list[str] = []
    for field in fields:
        for dim, value in field.selectors:
            if dim == "pressure_level" and value not in values:
                values.append(value)
    return values


def _positions_for_selector(source_values: np.ndarray, requested: list[str], logical: str) -> np.ndarray:
    norm: dict[str, int] = {}
    for i, value in enumerate(source_values):
        norm[_normalize_selector_value(value)] = i
    positions: list[int] = []
    for value in requested:
        key = _normalize_selector_value(value)
        if key not in norm:
            raise PipelineError(f"source fixture is missing requested {logical} value {value!r}")
        positions.append(norm[key])
    return np.asarray(positions, dtype=np.int64)


def _normalize_selector_value(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return format(f, "g")
    except Exception:
        return str(value)


def _latitude_positions(values: np.ndarray, north: float, south: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = (arr <= north) & (arr >= south)
    pos = np.nonzero(mask)[0]
    if pos.size == 0:
        raise PipelineError("source fixture has no latitude values inside requested area")
    return pos.astype(np.int64)


def _longitude_positions(values: np.ndarray, west: float, east: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if west == -180 and east == 180:
        return np.arange(arr.size, dtype=np.int64)
    comparable = ((arr + 180.0) % 360.0) - 180.0
    w = ((west + 180.0) % 360.0) - 180.0
    e = ((east + 180.0) % 360.0) - 180.0
    if w <= e:
        mask = (comparable >= w) & (comparable <= e)
    else:
        mask = (comparable >= w) | (comparable <= e)
    pos = np.nonzero(mask)[0]
    if pos.size == 0:
        raise PipelineError("source fixture has no longitude values inside requested area")
    return pos.astype(np.int64)


def _merged_variable_attrs(attrs: dict[str, Any], name: str, inventory: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    meta = inventory.get("option_metadata", {}).get("variable", {}).get(name, {})
    if meta.get("units") and not out.get("units"):
        out["units"] = meta["units"]
    if meta.get("description") and not out.get("description"):
        out["description"] = meta["description"]
    if meta.get("label") and not out.get("long_name"):
        out["long_name"] = meta["label"]
    return _json_safe_attrs(out)


def _dataset_attrs(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": contract.get("title") or inventory.get("title", "ERA5 pressure-level subset"),
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "provider": inventory.get("provider", "ECMWF"),
        "source_url": inventory.get("source_url", ""),
        "publication_format": "zarr_v3_consolidated",
        "pipeline_id": PIPELINE_ID,
        "semantic_note": "Decoded source values filtered by runtime contract; no lossy transforms, repacking, quantization, or selector scalarization applied.",
    }


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in {"scale_factor", "add_offset", "_FillValue", "missing_value"}:
            continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, np.ndarray):
            v = v.tolist()
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "selector"


def _balanced_blosc_codec() -> Any:
    from zarr.codecs import BloscCodec, BloscCname, BloscShuffle
    return BloscCodec(cname=BloscCname.zstd, clevel=BALANCED_BLOSC_CLEVEL, shuffle=BloscShuffle.bitshuffle)


def _publish_zarr(ds: xr.Dataset, channel_meta: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_store = output_dir / STORE_NAME
    tmp_store = output_dir / f".{STORE_NAME}.tmp"
    if tmp_store.exists():
        shutil.rmtree(tmp_store)
    codec = _balanced_blosc_codec()
    encoding: dict[str, dict[str, Any]] = {}
    lat_count = int(ds.sizes["latitude"])
    lon_count = int(ds.sizes["longitude"])
    for name, da in ds.data_vars.items():
        chunks: list[int] = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim.startswith("pressure_level"):
                chunks.append(1)
            elif dim == "latitude":
                chunks.append(lat_count)
            elif dim == "longitude":
                chunks.append(lon_count)
            else:
                chunks.append(int(ds.sizes[dim]))
        encoding[name] = {"chunks": tuple(chunks), "compressors": (codec,)}
    for coord in ds.coords:
        encoding[coord] = {"chunks": tuple(int(ds.sizes[d]) for d in ds[coord].dims), "compressors": (codec,)}

    write_ds = ds.copy(deep=False)
    for name in list(write_ds.data_vars) + list(write_ds.coords):
        write_ds[name].encoding = {}

    write_ds.to_zarr(tmp_store, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    _validate_published_store(tmp_store, ds, channel_meta)
    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)
    _validate_published_store(final_store, ds, channel_meta)

    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": channel_meta,
    }


def _validate_published_store(store: Path, expected: xr.Dataset, channel_meta: list[dict[str, Any]]) -> None:
    root_meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    if root_meta.get("zarr_format") != 3:
        raise PipelineError("published store is not Zarr v3")
    if not root_meta.get("consolidated_metadata"):
        raise PipelineError("published store is missing consolidated Zarr metadata")
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3, chunks=None)
    try:
        if set(reopened.data_vars) != set(expected.data_vars):
            raise PipelineError("published data variable names differ from filtered source")
        for coord in expected.coords:
            if coord not in reopened.coords:
                raise PipelineError(f"published coordinate {coord!r} is missing")
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
        for var in expected.data_vars:
            if reopened[var].dims != expected[var].dims:
                raise PipelineError(f"published dimensions for {var!r} changed")
            np.testing.assert_array_equal(reopened[var].values, expected[var].values)
            meta = json.loads((store / var / "zarr.json").read_text(encoding="utf-8"))
            chunks = tuple(meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", ()))
            expected_chunks = _expected_chunk_shape(expected[var])
            if chunks != expected_chunks:
                raise PipelineError(f"published chunks for {var!r} are {chunks}, expected {expected_chunks}")
            if not _metadata_has_balanced_blosc(meta):
                raise PipelineError(f"published compressor for {var!r} is not balanced lossless Blosc/Zstd clevel 3")
        for ch in channel_meta:
            if ch["array_path"] not in reopened.data_vars:
                raise PipelineError("artifact channel references missing data array")
            for _selector_dim, coord_path in ch.get("selector_coordinate_paths", {}).items():
                if coord_path not in reopened.coords:
                    raise PipelineError("artifact channel references missing selector coordinate")
    finally:
        reopened.close()


def _expected_chunk_shape(da: xr.DataArray) -> tuple[int, ...]:
    out: list[int] = []
    for dim in da.dims:
        if dim == "time" or dim.startswith("pressure_level"):
            out.append(1)
        else:
            out.append(int(da.sizes[dim]))
    return tuple(out)


def _metadata_has_balanced_blosc(meta: dict[str, Any]) -> bool:
    def walk(value: Any) -> Iterable[dict[str, Any]]:
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for node in walk(meta):
        node_text = json.dumps(node, sort_keys=True).lower()
        name = str(node.get("name") or node.get("codec") or node.get("id") or "").lower()
        if "blosc" not in name and "blosc" not in node_text:
            continue
        configuration = node.get("configuration")
        merged = dict(node)
        if isinstance(configuration, dict):
            merged.update(configuration)
        try:
            clevel = int(merged.get("clevel"))
        except (TypeError, ValueError):
            continue
        cname = str(merged.get("cname") or "").lower()
        shuffle = str(merged.get("shuffle") or "").lower()
        if cname == "zstd" and clevel == BALANCED_BLOSC_CLEVEL and shuffle == "bitshuffle":
            return True
    return False


def _assert_filtered_semantics(source: xr.Dataset, filtered: xr.Dataset, contract: dict[str, Any], requested: list[RequestedField], channel_meta: list[dict[str, Any]], time_dim: str, lat_dim: str, lon_dim: str, pressure_dim: str, time_positions: np.ndarray, lat_positions: np.ndarray, lon_positions: np.ndarray) -> None:
    if "time" not in filtered.dims or "latitude" not in filtered.dims or "longitude" not in filtered.dims:
        raise PipelineError("filtered dataset is missing required grid dimensions")
    if filtered.sizes["time"] != len(_requested_datetimes(contract)):
        raise PipelineError("filtered dataset time cardinality does not match runtime lock")
    for rf, ch in zip(requested, channel_meta):
        arr = ch["array_path"]
        coord = ch["selector_coordinate_paths"].get("pressure_level")
        if arr not in filtered.data_vars or coord not in filtered.coords:
            raise PipelineError("channel declaration does not match filtered dataset")
        if coord not in filtered[arr].dims:
            raise PipelineError("selector coordinate is not an actual data-array dimension")
    # Exact value preservation is also checked after publication. Here we ensure no
    # source encoding keys that can cause repacking remain on public variables.
    forbidden = {"scale_factor", "add_offset", "_FillValue", "missing_value"}
    for var in filtered.data_vars:
        if forbidden.intersection(filtered[var].encoding):
            raise PipelineError("source packing encoding leaked into filtered dataset")
