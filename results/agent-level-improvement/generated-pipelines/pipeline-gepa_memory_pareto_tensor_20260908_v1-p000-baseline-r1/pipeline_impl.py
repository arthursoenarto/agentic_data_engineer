"""Standalone ERA5 pressure-level family adapter.

Entry point required by the framework:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The implementation is intentionally offline-first.  If cache_dir contains a
source_fixture_manifest.json, every listed source object is verified by size and
SHA-256 before any other acquisition path is considered.  This pilot adapter does
not construct a CDS client or perform network activity; a complete local fixture
is the authoritative raw input.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p000-baseline-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
PUBLIC_STORE_NAME = "dataset.zarr"
SOURCE_MANIFEST_NAME = "source_fixture_manifest.json"

# Common ERA5 GRIB short names.  Exact inventory variable names remain preferred.
ERA5_SHORT_NAMES = {
    "divergence": "d",
    "fraction_of_cloud_cover": "cc",
    "geopotential": "z",
    "ozone_mass_mixing_ratio": "o3",
    "potential_vorticity": "pv",
    "relative_humidity": "r",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc",
    "specific_humidity": "q",
    "specific_rain_water_content": "crwc",
    "specific_snow_water_content": "cswc",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "vorticity": "vo",
}

COORD_ALIASES = {
    "time": ("time", "valid_time"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "level", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}


class PipelineValidationError(ValueError):
    """Raised for invalid contracts, inventory mismatches, or incomplete fixtures."""


@dataclass(frozen=True)
class VerifiedSource:
    entry_id: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuntimeSelection:
    contract: dict[str, Any]
    selected_times: pd.DatetimeIndex
    variables: list[str]
    pressure_levels: list[str]
    fields: list[dict[str, Any]]
    area: list[float]


def run_pipeline(contract_lock: Any, inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Materialize a decoded, consolidated Zarr v3 grid artifact.

    Parameters are supplied by the framework.  The function writes only beneath
    output_dir, treats cache_dir as external input/cache, and returns a
    secret-safe artifact manifest compatible with family_pipeline_interface.v3.
    """
    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    contract = _extract_contract(contract_lock)
    selection = _validate_and_plan(contract, inventory)

    # Fixture verification is deliberately first: no provider/client/credential
    # work can happen before this branch.  The adapter has no network fallback.
    sources = _verify_source_fixture(cache_root)
    if not sources:
        raise PipelineValidationError(
            f"No verified local source fixture found. Provide {SOURCE_MANIFEST_NAME} at cache_dir root."
        )

    source_ds = _open_and_combine_sources(sources)
    normalized = _normalize_dataset(source_ds)
    filtered = _filter_dataset(normalized, selection)
    public_ds, channels = _build_public_dataset(filtered, selection, inventory)

    _publish_zarr_atomically(public_ds, output_root / PUBLIC_STORE_NAME)
    _validate_public_zarr(output_root / PUBLIC_STORE_NAME, selection, channels)

    reused_keys = [f"fixture:{src.entry_id}:sha256:{src.sha256[:16]}" for src in sources]
    return {
        "cache": {
            "hits": len(sources),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused_keys,
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": PUBLIC_STORE_NAME,
            "dimensions": {
                "sample": "time",
                "y": "latitude",
                "x": "longitude",
            },
            "coordinates": {
                "sample": "time",
                "y": "latitude",
                "x": "longitude",
            },
            "channels": channels,
        },
        "warnings": [],
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise PipelineValidationError("contract_lock must be a JSON object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "runtime_contract"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
            return value
    contracts = lock.get("contracts")
    if isinstance(contracts, list):
        matches = [c for c in contracts if isinstance(c, dict) and c.get("schema_version") == "dataset_contract.v1"]
        if len(matches) == 1:
            return matches[0]
        slug_matches = [c for c in matches if c.get("dataset_slug") == DATASET_SLUG]
        if len(slug_matches) == 1:
            return slug_matches[0]
    raise PipelineValidationError("Unable to locate a dataset_contract.v1 object in contract_lock")


def _validate_and_plan(contract: dict[str, Any], inventory: dict[str, Any]) -> RuntimeSelection:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineValidationError("inventory schema_version must be dataset_inventory.v1")
    if contract.get("dataset_slug") != inventory.get("dataset_slug") or contract.get("dataset_slug") != DATASET_SLUG:
        raise PipelineValidationError("contract and inventory dataset_slug must match ERA5 pressure-level inventory")
    if inventory.get("dataset_id") != DATASET_ID:
        raise PipelineValidationError("inventory dataset_id does not match fixed policy")
    if not contract.get("human_confirmed", False):
        raise PipelineValidationError("contract must be human_confirmed")

    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise PipelineValidationError("advanced_options.dataset_id conflicts with fixed policy")
    if adv.get("data_format", "grib") != "grib":
        raise PipelineValidationError("fixed policy requires acquisition data_format='grib'")
    if adv.get("download_format", "unarchived") not in _options(inventory, "download_format"):
        raise PipelineValidationError("unsupported download_format")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise PipelineValidationError("contract.fields must be a non-empty list")

    variables: list[str] = []
    pressure_levels: list[str] = []
    for field in fields:
        name = field.get("name")
        if name not in _options(inventory, "variable"):
            raise PipelineValidationError(f"unsupported variable: {name!r}")
        if name not in variables:
            variables.append(name)
        selectors = field.get("selectors") or []
        if not isinstance(selectors, list):
            raise PipelineValidationError("field.selectors must be a list")
        pl_selectors = [s for s in selectors if isinstance(s, dict) and s.get("dimension") == "pressure_level"]
        if len(pl_selectors) != 1:
            raise PipelineValidationError("each ERA5 pressure-level field must declare exactly one pressure_level selector")
        pl = str(pl_selectors[0].get("value"))
        if pl not in _options(inventory, "pressure_level"):
            raise PipelineValidationError(f"unsupported pressure_level: {pl!r}")
        if pl not in pressure_levels:
            pressure_levels.append(pl)
        for selector in selectors:
            dim = selector.get("dimension")
            if dim != "pressure_level":
                raise PipelineValidationError(f"unsupported selector dimension for this inventory: {dim!r}")

    scope = contract.get("scope") or {}
    product_type = scope.get("product_type")
    product_values = adv.get("product_type", [product_type] if product_type else [])
    if product_type != "reanalysis" or product_values != ["reanalysis"]:
        raise PipelineValidationError("fixed policy supports product_type=['reanalysis'] only")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone", "UTC") != "UTC":
        raise PipelineValidationError("scope.time.timezone must be UTC")
    selected_clock_times = time_scope.get("selected_times")
    if not isinstance(selected_clock_times, list) or not selected_clock_times:
        raise PipelineValidationError("scope.time.selected_times must be a non-empty list")
    for t in selected_clock_times:
        if t not in _options(inventory, "time"):
            raise PipelineValidationError(f"unsupported selected time: {t!r}")

    date_range = scope.get("date_range") or {}
    start = _parse_date(date_range.get("start_date"), "start_date")
    end = _parse_date(date_range.get("end_date"), "end_date")
    if end < start:
        raise PipelineValidationError("date_range.end_date must be >= start_date")
    inclusive = bool(date_range.get("inclusive", True))
    selected_times = _selected_datetimes(start, end, inclusive, selected_clock_times)
    _validate_temporal_options(selected_times, inventory)
    _validate_inventory_temporal_extent(selected_times, inventory)

    geography = scope.get("geography") or {}
    area = geography.get("cds_area") or geography.get("area")
    if area == "global" or area is None:
        area = inventory.get("defaults", {}).get("area", [90, -180, -90, 180])
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)):
        raise PipelineValidationError("scope.geography.cds_area must be [north, west, south, east]")
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineValidationError("geographic area is outside valid latitude/longitude bounds")

    return RuntimeSelection(
        contract=contract,
        selected_times=selected_times,
        variables=variables,
        pressure_levels=pressure_levels,
        fields=fields,
        area=[north, west, south, east],
    )


def _options(inventory: dict[str, Any], field: str) -> list[Any]:
    values = inventory.get("options", {}).get(field)
    if not isinstance(values, list):
        raise PipelineValidationError(f"inventory.options.{field} is missing")
    return values


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise PipelineValidationError(f"date_range.{name} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineValidationError(f"invalid date_range.{name}: {value!r}") from exc


def _selected_datetimes(start: date, end: date, inclusive: bool, selected_clock_times: Iterable[str]) -> pd.DatetimeIndex:
    # Date-only inclusive end means the complete final UTC calendar day.  We
    # therefore enumerate days first and then apply the requested clock times.
    final_day = end if inclusive else end - timedelta(days=1)
    if final_day < start:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    stamps: list[pd.Timestamp] = []
    day = start
    parsed_times = [_parse_clock_time(t) for t in selected_clock_times]
    while day <= final_day:
        for clock in parsed_times:
            stamps.append(pd.Timestamp(datetime.combine(day, clock, tzinfo=timezone.utc)).tz_convert(None))
        day += timedelta(days=1)
    if not stamps:
        raise PipelineValidationError("selected date/time range is empty")
    return pd.DatetimeIndex(stamps).sort_values()


def _parse_clock_time(value: str) -> time:
    if not re.fullmatch(r"\d{2}:\d{2}", value):
        raise PipelineValidationError(f"invalid UTC time selector: {value!r}")
    hh, mm = map(int, value.split(":"))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise PipelineValidationError(f"invalid UTC time selector: {value!r}")
    return time(hh, mm)


def _validate_temporal_options(stamps: pd.DatetimeIndex, inventory: dict[str, Any]) -> None:
    years = set(_options(inventory, "year"))
    months = set(_options(inventory, "month"))
    days = set(_options(inventory, "day"))
    times = set(_options(inventory, "time"))
    for stamp in stamps:
        if f"{stamp.year:04d}" not in years:
            raise PipelineValidationError(f"year {stamp.year:04d} is not available in inventory")
        if f"{stamp.month:02d}" not in months:
            raise PipelineValidationError(f"month {stamp.month:02d} is not available in inventory")
        if f"{stamp.day:02d}" not in days:
            raise PipelineValidationError(f"day {stamp.day:02d} is not available in inventory")
        if f"{stamp.hour:02d}:{stamp.minute:02d}" not in times:
            raise PipelineValidationError(f"time {stamp.hour:02d}:{stamp.minute:02d} is not available in inventory")


def _validate_inventory_temporal_extent(stamps: pd.DatetimeIndex, inventory: dict[str, Any]) -> None:
    intervals = inventory.get("catalogue_metadata", {}).get("extent", {}).get("temporal", {}).get("interval", [])
    if not intervals:
        return
    start_raw, end_raw = intervals[0]
    start = pd.Timestamp(start_raw).tz_convert(None) if pd.Timestamp(start_raw).tzinfo else pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw).tz_convert(None) if pd.Timestamp(end_raw).tzinfo else pd.Timestamp(end_raw)
    if stamps.min() < start or stamps.max() > end:
        raise PipelineValidationError("requested timestamps are outside inventory temporal extent")


def _verify_source_fixture(cache_root: Path) -> list[VerifiedSource]:
    manifest_path = cache_root / SOURCE_MANIFEST_NAME
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineValidationError("source fixture manifest schema_version is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineValidationError("source fixture manifest must contain non-empty entries")

    verified: list[VerifiedSource] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PipelineValidationError(f"manifest entry {i} must be an object")
        rel = entry.get("relative_path")
        entry_id = entry.get("entry_id") or f"entry-{i}"
        expected_size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
        if not isinstance(rel, str) or rel.startswith("/") or "\x00" in rel:
            raise PipelineValidationError(f"manifest entry {entry_id!r} has invalid relative_path")
        path = (cache_root / rel).resolve()
        if not _is_relative_to(path, cache_root):
            raise PipelineValidationError(f"manifest entry {entry_id!r} escapes cache_dir")
        if not path.is_file():
            raise PipelineValidationError(f"manifest entry {entry_id!r} file is missing")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise PipelineValidationError(f"manifest entry {entry_id!r} has invalid size_bytes")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise PipelineValidationError(f"manifest entry {entry_id!r} size mismatch")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise PipelineValidationError(f"manifest entry {entry_id!r} has invalid sha256")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise PipelineValidationError(f"manifest entry {entry_id!r} sha256 mismatch")
        verified.append(VerifiedSource(str(entry_id), path, actual_sha, actual_size))
    return verified


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_and_combine_sources(sources: list[VerifiedSource]) -> xr.Dataset:
    datasets = [_open_source(src.path) for src in sources]
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
    except Exception as exc:
        raise PipelineValidationError("verified source files could not be combined by coordinates") from exc


def _open_source(path: Path) -> xr.Dataset:
    suffixes = {s.lower() for s in path.suffixes}
    try:
        if suffixes & {".grib", ".grb", ".grb2"}:
            try:
                return xr.open_dataset(path, engine="cfgrib", mask_and_scale=True, decode_times=True)
            except ValueError as exc:
                raise PipelineValidationError(
                    "GRIB fixture decoding requires cfgrib/eccodes and a single compatible GRIB hypercube"
                ) from exc
        if suffixes & {".nc", ".nc4", ".netcdf"}:
            return xr.open_dataset(path, mask_and_scale=True, decode_times=True)
        # Last resort: let xarray inspect the backend.  This still applies CF
        # scale/offset and fill-value masking for NetCDF inputs.
        return xr.open_dataset(path, mask_and_scale=True, decode_times=True)
    except PipelineValidationError:
        raise
    except Exception as exc:
        raise PipelineValidationError(f"unable to decode source fixture {path.name!r}") from exc


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    all_names = set(ds.dims) | set(ds.coords) | set(ds.variables)
    for canonical, aliases in COORD_ALIASES.items():
        if canonical in all_names:
            continue
        for alias in aliases:
            if alias in all_names:
                rename[alias] = canonical
                break
    if rename:
        ds = ds.rename(rename)
    missing = [c for c in ("time", "pressure_level", "latitude", "longitude") if c not in ds.coords and c not in ds.dims]
    if missing:
        raise PipelineValidationError(f"source dataset is missing required coordinates/dimensions: {missing}")
    if "time" not in ds.indexes:
        ds = ds.assign_coords(time=pd.DatetimeIndex(pd.to_datetime(ds["time"].values)))
    return ds


def _filter_dataset(ds: xr.Dataset, selection: RuntimeSelection) -> xr.Dataset:
    missing_times = selection.selected_times.difference(pd.DatetimeIndex(pd.to_datetime(ds["time"].values)))
    if len(missing_times):
        preview = ", ".join(str(x) for x in missing_times[:5])
        raise PipelineValidationError(f"source fixture does not contain requested timestamps: {preview}")
    ds = ds.sel(time=selection.selected_times)

    ds = _select_pressure_levels(ds, selection.pressure_levels)
    ds = _select_area(ds, selection.area)
    return ds


def _select_pressure_levels(ds: xr.Dataset, levels: list[str]) -> xr.Dataset:
    coord = ds["pressure_level"]
    source_values = list(coord.values)
    if np.issubdtype(coord.dtype, np.number):
        desired = np.array([float(v) if "." in v else int(v) for v in levels], dtype=coord.dtype)
        source_set = {float(v) for v in source_values}
        missing = [str(v) for v in desired if float(v) not in source_set]
    else:
        desired = np.array(levels, dtype=coord.dtype)
        source_set = {str(v) for v in source_values}
        missing = [str(v) for v in levels if str(v) not in source_set]
    if missing:
        raise PipelineValidationError(f"source fixture lacks requested pressure levels: {missing}")
    return ds.sel(pressure_level=desired)


def _select_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    # Global request: preserve native coordinates and ordering exactly.
    if north == 90 and south == -90 and west in (-180, 0) and east in (180, 360):
        return ds
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_mask = (lat >= south) & (lat <= north)
    lon_mask = (lon >= west) & (lon <= east)
    if not bool(np.any(lat_mask)) or not bool(np.any(lon_mask)):
        raise PipelineValidationError("geographic subset selects no source grid cells")
    return ds.sel(latitude=ds["latitude"].values[lat_mask], longitude=ds["longitude"].values[lon_mask])


def _build_public_dataset(ds: xr.Dataset, selection: RuntimeSelection, inventory: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    data_vars: dict[str, xr.DataArray] = {}
    channels: list[dict[str, Any]] = []

    # Shared selected pressure coordinate keeps the selector dimension explicit in
    # the public regular-grid store, including cardinality-one selections.
    selected_pressure = ds["pressure_level"]

    for field in selection.fields:
        field_id = _canonical_field_id(field)
        variable_name = str(field["name"])
        pressure_value = _field_pressure_value(field)
        source_var = _find_source_variable(ds, variable_name)
        da = ds[source_var]
        coerced_pressure = _coerce_level_for_coord(pressure_value, ds["pressure_level"])
        if "pressure_level" in da.dims:
            da = da.sel(pressure_level=coerced_pressure)
            pressure_scalar = da["pressure_level"].item()
        else:
            if "pressure_level" in da.coords:
                pressure_scalar = da["pressure_level"].item()
                if str(pressure_scalar) not in {str(coerced_pressure), str(pressure_value)}:
                    raise PipelineValidationError(
                        f"source variable {source_var!r} is tagged with pressure_level {pressure_scalar!r}, not requested {pressure_value!r}"
                    )
            else:
                available = {str(v) for v in np.asarray(selected_pressure.values).tolist()}
                if str(coerced_pressure) not in available and str(pressure_value) not in available:
                    raise PipelineValidationError(
                        f"source variable {source_var!r} lacks requested pressure_level {pressure_value!r}"
                    )
                pressure_scalar = coerced_pressure
        # Keep pressure_level as an actual dimension, not a scalar coordinate.
        da = da.expand_dims(pressure_level=[pressure_scalar])
        da = da.reindex(pressure_level=selected_pressure.values)
        da = da.transpose("time", "pressure_level", "latitude", "longitude", ...)
        da.name = field_id
        da.attrs = dict(da.attrs)
        meta = inventory.get("option_metadata", {}).get("variable", {}).get(variable_name, {})
        if "units" not in da.attrs and meta.get("units"):
            da.attrs["units"] = meta["units"]
        if meta.get("label"):
            da.attrs.setdefault("long_name", meta["label"])
        if meta.get("description"):
            da.attrs.setdefault("description", meta["description"])
        da.attrs["source_variable"] = variable_name
        da.attrs["canonical_field_id"] = field_id
        da.attrs["selector_pressure_level"] = pressure_value
        da.encoding = _clean_encoding(da.encoding)
        data_vars[field_id] = da
        channels.append(
            {
                "field_id": field_id,
                "array_path": field_id,
                "selectors": {"pressure_level": pressure_value},
                "selector_coordinate_paths": {"pressure_level": "pressure_level"},
            }
        )

    public = xr.Dataset(data_vars=data_vars, coords={
        "time": ds["time"],
        "pressure_level": selected_pressure,
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    })
    public.attrs.update(
        {
            "dataset_slug": DATASET_SLUG,
            "dataset_id": DATASET_ID,
            "provider": "ECMWF",
            "publication_format": "zarr_v3_consolidated",
            "pipeline_id": PIPELINE_ID,
        }
    )
    for coord in public.coords:
        public[coord].encoding = _clean_encoding(public[coord].encoding)
    return public, channels


def _find_source_variable(ds: xr.Dataset, variable_name: str) -> str:
    if variable_name in ds.data_vars:
        return variable_name
    short = ERA5_SHORT_NAMES.get(variable_name)
    if short and short in ds.data_vars:
        return short
    for candidate in ds.data_vars:
        attrs = ds[candidate].attrs
        if attrs.get("long_name") == variable_name or attrs.get("standard_name") == variable_name:
            return candidate
    raise PipelineValidationError(f"source fixture lacks requested variable {variable_name!r}")


def _field_pressure_value(field: dict[str, Any]) -> str:
    for selector in field.get("selectors") or []:
        if selector.get("dimension") == "pressure_level":
            return str(selector.get("value"))
    raise PipelineValidationError("field is missing pressure_level selector")


def _coerce_level_for_coord(value: str, coord: xr.DataArray) -> Any:
    if np.issubdtype(coord.dtype, np.integer):
        return int(value)
    if np.issubdtype(coord.dtype, np.floating):
        return float(value)
    return value


def _canonical_field_id(field: dict[str, Any]) -> str:
    name = str(field.get("name"))
    selectors = field.get("selectors") or []
    if not selectors:
        return name
    pieces = []
    for selector in selectors:
        dim = str(selector.get("dimension"))
        value = json.dumps(str(selector.get("value")), ensure_ascii=False, separators=(",", ":"))
        pieces.append(f"{dim}={value}")
    return f"{name}[{','.join(pieces)}]"


def _clean_encoding(encoding: dict[str, Any]) -> dict[str, Any]:
    # Do not propagate source packing, dtype coercion, or fill-value conventions
    # into the public Zarr.  Values have already been decoded/masked by xarray.
    blocked = {
        "scale_factor",
        "add_offset",
        "_FillValue",
        "missing_value",
        "dtype",
        "source",
        "original_shape",
        "preferred_chunks",
        "coordinates",
        "filter_by_keys",
        "encode_cf",
    }
    return {k: v for k, v in dict(encoding).items() if k not in blocked}


def _publish_zarr_atomically(ds: xr.Dataset, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{target.name}-{uuid.uuid4().hex}"
    backup = target.parent / f".old-{target.name}-{uuid.uuid4().hex}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        encoding: dict[str, dict[str, Any]] = {}
        for name, da in ds.data_vars.items():
            chunks = tuple(max(1, min(int(size), preferred)) for size, preferred in zip(da.shape, (8, 4, 180, 360)))
            encoding[name] = {**_clean_encoding(da.encoding), "chunks": chunks}
        ds.to_zarr(tmp, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
        _validate_basic_zarr(tmp)
        if target.exists():
            target.rename(backup)
        tmp.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if backup.exists() and not target.exists():
            backup.rename(target)
        elif backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        raise


def _validate_basic_zarr(path: Path) -> None:
    if not path.is_dir():
        raise PipelineValidationError("published Zarr target is not a directory store")
    if not (path / "zarr.json").is_file():
        raise PipelineValidationError("published store is not a Zarr v3 store")
    xr.open_zarr(path, consolidated=True).close()


def _validate_public_zarr(path: Path, selection: RuntimeSelection, channels: list[dict[str, Any]]) -> None:
    try:
        reopened = xr.open_zarr(path, consolidated=True)
        for dim in ("time", "pressure_level", "latitude", "longitude"):
            if dim not in reopened.dims:
                raise PipelineValidationError(f"published Zarr is missing dimension {dim!r}")
        if reopened.sizes["time"] != len(selection.selected_times):
            raise PipelineValidationError("published Zarr time dimension has wrong length")
        if reopened.sizes["pressure_level"] != len(selection.pressure_levels):
            raise PipelineValidationError("published Zarr pressure_level dimension has wrong length")
        for channel in channels:
            if channel["array_path"] not in reopened.data_vars:
                raise PipelineValidationError(f"published Zarr is missing channel {channel['field_id']!r}")
            if "pressure_level" not in reopened[channel["array_path"]].dims:
                raise PipelineValidationError(f"channel {channel['field_id']!r} lost pressure_level dimension")
    finally:
        try:
            reopened.close()  # type: ignore[name-defined]
        except Exception:
            pass
