"""Standalone ERA5 pressure-level fixture-to-Zarr adapter.

Entry point: run_pipeline(contract_lock, inventory, cache_dir, output_dir)
This module intentionally performs no network access.  A complete verified
source_fixture_manifest.json in cache_dir is required.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p019-search-r1"
STORE_NAME = "dataset.zarr"
SUPPORTED_DATASET_ID = "reanalysis-era5-pressure-levels"
SUPPORTED_SLUG = "reanalysis_era5_pressure_levels"

_SHORT_TO_LONG = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "r": "relative_humidity",
    "q": "specific_humidity",
    "vo": "vorticity",
    "d": "divergence",
    "cc": "fraction_of_cloud_cover",
    "pv": "potential_vorticity",
    "o3": "ozone_mass_mixing_ratio",
    "ciwc": "specific_cloud_ice_water_content",
    "clwc": "specific_cloud_liquid_water_content",
    "crwc": "specific_rain_water_content",
    "cswc": "specific_snow_water_content",
}
_PRESSURE_DIM_ALIASES = ("pressure_level", "isobaricInhPa", "isobaricInPa", "level", "plev")
_LAT_ALIASES = ("latitude", "lat")
_LON_ALIASES = ("longitude", "lon")
_TIME_ALIASES = ("time", "valid_time")
_COMPRESSOR_CODEC_NAMES = {"zstd", "gzip", "blosc", "lz4", "byteshuffle", "crc32c"}


class PipelineError(ValueError):
    """Secret-safe adapter validation or materialization error."""


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize a validated ERA5 pressure-level request from a local fixture.

    Parameters match the framework contract exactly.  The selected contract is
    extracted from a lock envelope when present; filtering is always derived from
    runtime contract contents rather than from the seed example.
    """
    contract = _extract_contract(contract_lock)
    request = _validate_contract_and_inventory(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    entries = _verify_fixture_manifest(cache_root)
    raw_files = _fixture_data_files(cache_root, entries)
    if not raw_files:
        raise PipelineError("source fixture manifest contains no supported raw data files")

    source = _open_source_dataset(raw_files)
    source = _normalize_source(source)
    filtered_source = _filter_source(source, request)
    published, channels = _build_channel_dataset(filtered_source, contract, request, inventory)

    out_root.mkdir(parents=True, exist_ok=True)
    final_store = out_root / STORE_NAME
    tmp_store = out_root / ".staging_dataset.zarr"
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    codec_mode = _write_zarr_v3(published, tmp_store, channels)
    artifact = _artifact_layout(channels)
    _validate_published_store(tmp_store, published, artifact, channels, codec_mode)

    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)
    _validate_published_store(final_store, published, artifact, channels, codec_mode)

    reused_keys = [f"fixture:{e['entry_id']}:{e['sha256'][:16]}" for e in entries]
    warnings: list[str] = []
    if codec_mode != "uncompressed":
        warnings.append("uncompressed Zarr v3 chunks were unavailable; used lossless Blosc-LZ4 clevel=1 byte-shuffle fallback")

    return {
        "cache": {
            "hits": len(entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused_keys,
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": warnings,
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise PipelineError("contract_lock must be an object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "contract_snapshot"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
            return value
    if isinstance(lock.get("locks"), list):
        matches = [x for x in lock["locks"] if isinstance(x, dict) and x.get("schema_version") == "dataset_contract.v1"]
        if len(matches) == 1:
            return matches[0]
    raise PipelineError("could not locate dataset_contract.v1 in contract lock envelope")


def _validate_contract_and_inventory(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("unsupported inventory schema")
    if contract.get("dataset_slug") != inventory.get("dataset_slug") or contract.get("dataset_slug") != SUPPORTED_SLUG:
        raise PipelineError("contract and inventory dataset_slug do not match supported ERA5 pressure-level inventory")
    if inventory.get("dataset_id") != SUPPORTED_DATASET_ID:
        raise PipelineError("unsupported inventory dataset_id")
    if contract.get("human_confirmed") is not True:
        raise PipelineError("contract must be human_confirmed")

    options = inventory.get("options", {})
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id") not in (None, SUPPORTED_DATASET_ID):
        raise PipelineError("advanced_options.dataset_id is not supported")
    if adv.get("data_format", inventory.get("defaults", {}).get("data_format")) != "grib":
        raise PipelineError("fixed policy requires acquisition data_format='grib'")
    if adv.get("download_format", inventory.get("defaults", {}).get("download_format")) != "unarchived":
        raise PipelineError("fixed policy requires download_format='unarchived'")
    product_type = adv.get("product_type", contract.get("scope", {}).get("product_type", inventory.get("defaults", {}).get("product_type")))
    product_values = product_type if isinstance(product_type, list) else [product_type]
    if product_values != ["reanalysis"] or any(v not in options.get("product_type", []) for v in product_values):
        raise PipelineError("only product_type ['reanalysis'] is supported by this policy")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise PipelineError("contract.fields must be a non-empty list")
    seen_ids: set[str] = set()
    for f in fields:
        name = f.get("name")
        if name not in options.get("variable", []):
            raise PipelineError(f"unsupported variable: {name}")
        selectors = f.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise PipelineError("ERA5 pressure-level fields require exactly one pressure_level selector")
        value = str(selectors[0].get("value"))
        if value not in options.get("pressure_level", []):
            raise PipelineError(f"unsupported pressure_level selector: {value}")
        if selectors[0].get("unit") not in (None, "hPa"):
            raise PipelineError("pressure_level selector unit must be hPa when supplied")
        fid = canonical_field_id(f)
        if fid in seen_ids:
            raise PipelineError(f"duplicate requested field-selector channel: {fid}")
        seen_ids.add(fid)

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise PipelineError("end_date precedes start_date")
    inv_years = set(options.get("year", []))
    for y in range(start.year, end.year + 1):
        if f"{y:04d}" not in inv_years:
            raise PipelineError(f"requested year {y:04d} is outside inventory options")
    if dr.get("inclusive") is not True:
        raise PipelineError("date_range.inclusive must be true for date-only bounds")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone") not in (None, "UTC"):
        raise PipelineError("selected times must be UTC")
    selected_times = time_scope.get("selected_times")
    if not isinstance(selected_times, list) or not selected_times:
        raise PipelineError("scope.time.selected_times must be a non-empty list")
    for t in selected_times:
        if t not in options.get("time", []):
            raise PipelineError(f"unsupported selected time: {t}")
        _parse_hhmm(t)

    geo = scope.get("geography") or {}
    area = geo.get("cds_area", inventory.get("defaults", {}).get("area"))
    if geo.get("area", "global") == "global" and list(area) != [90, -180, -90, 180]:
        raise PipelineError("global geography must use full CDS area")
    if geo.get("cds_area_order", ["north", "west", "south", "east"]) != ["north", "west", "south", "east"]:
        raise PipelineError("unsupported cds_area_order")
    if not (isinstance(area, list) and len(area) == 4):
        raise PipelineError("cds_area must be [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("cds_area bounds are invalid")

    timestamps = _requested_timestamps(start, end, selected_times)
    return {"start": start, "end": end, "times": selected_times, "timestamps": timestamps, "area": [north, west, south, east]}


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PipelineError(f"{label} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineError(f"{label} must be YYYY-MM-DD") from exc


def _parse_hhmm(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise PipelineError(f"invalid UTC selected time: {value}") from exc


def _requested_timestamps(start: date, end: date, selected_times: list[str]) -> np.ndarray:
    out: list[np.datetime64] = []
    cur = start
    while cur <= end:
        for hhmm in selected_times:
            t = _parse_hhmm(hhmm)
            dt = datetime.combine(cur, t, tzinfo=timezone.utc).replace(tzinfo=None)
            out.append(np.datetime64(dt, "ns"))
        cur += timedelta(days=1)
    return np.array(out, dtype="datetime64[ns]")


def _verify_fixture_manifest(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise PipelineError("complete local source fixture is required; source_fixture_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("unsupported source fixture manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("source fixture manifest has no entries")
    verified: list[dict[str, Any]] = []
    for e in entries:
        rel = e.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise PipelineError("fixture entry relative_path must be relative")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineError("fixture entry escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineError("fixture entry is missing or not a file")
        expected_size = int(e.get("size_bytes"))
        if expected_size <= 0 or path.stat().st_size != expected_size:
            raise PipelineError("fixture entry size verification failed")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != e.get("sha256"):
            raise PipelineError("fixture entry sha256 verification failed")
        entry_id = str(e.get("entry_id", rel))
        verified.append({"entry_id": entry_id, "relative_path": rel, "path": path, "sha256": sha, "size_bytes": expected_size})
    return verified


def _fixture_data_files(cache_root: Path, entries: list[dict[str, Any]]) -> list[Path]:
    suffixes = {".grib", ".grb", ".grb2", ".nc", ".nc4", ".netcdf"}
    files = [Path(e["path"]) for e in entries if Path(e["path"]).suffix.lower() in suffixes]
    return sorted(files, key=lambda p: str(p.relative_to(cache_root)))


def _promote_scalar_pressure_coordinate(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "pressure_level" not in ds.coords and "pressure_level" not in ds.dims:
        for old in _PRESSURE_DIM_ALIASES:
            if old in ds.coords or old in ds.dims:
                rename[old] = "pressure_level"
                break
    if rename:
        ds = ds.rename(rename)
    if "pressure_level" in ds.coords and "pressure_level" not in ds.dims:
        coord = ds["pressure_level"]
        values = np.asarray(coord.values)
        if values.ndim == 0:
            attrs = dict(coord.attrs)
            ds = ds.drop_vars("pressure_level").expand_dims({"pressure_level": values.reshape(1)})
            ds["pressure_level"].attrs = attrs
    return ds


def _open_source_dataset(files: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in {".grib", ".grb", ".grb2"}:
                ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}, mask_and_scale=True, decode_cf=True)
            else:
                ds = xr.open_dataset(path, mask_and_scale=True, decode_cf=True)
            ds = _promote_scalar_pressure_coordinate(ds)
            datasets.append(ds)
        except Exception as exc:  # noqa: BLE001 - converted to secret-safe message
            errors.append(f"{path.name}: {type(exc).__name__}")
    if not datasets:
        raise PipelineError("could not decode fixture data files: " + "; ".join(errors))
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="override")
    except Exception:
        return xr.merge(datasets, compat="override", combine_attrs="override")


def _normalize_source(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for old in _LAT_ALIASES:
        if old in ds.coords and old != "latitude":
            rename[old] = "latitude"
            break
    for old in _LON_ALIASES:
        if old in ds.coords and old != "longitude":
            rename[old] = "longitude"
            break
    for old in _PRESSURE_DIM_ALIASES:
        if old in ds.coords and old != "pressure_level":
            rename[old] = "pressure_level"
            break
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    ds = ds.rename(rename)
    var_rename = {short: long for short, long in _SHORT_TO_LONG.items() if short in ds.data_vars and long not in ds.data_vars}
    if var_rename:
        ds = ds.rename(var_rename)
    required = ["time", "pressure_level", "latitude", "longitude"]
    missing = [c for c in required if c not in ds.coords and c not in ds.dims]
    if missing:
        raise PipelineError("source fixture lacks required coordinates: " + ", ".join(missing))
    return ds


def _filter_source(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    times = pd.to_datetime(ds["time"].values).to_numpy(dtype="datetime64[ns]")
    wanted = request["timestamps"]
    mask = np.isin(times, wanted)
    if int(mask.sum()) != len(wanted):
        missing = [str(x) for x in wanted[~np.isin(wanted, times)][:5]]
        raise PipelineError("source fixture does not contain all requested timestamps; first missing: " + ", ".join(missing))
    ds = ds.isel(time=np.nonzero(mask)[0])
    order = np.argsort(pd.to_datetime(ds["time"].values).to_numpy(dtype="datetime64[ns]"), kind="stable")
    ds = ds.isel(time=order)
    if not np.array_equal(pd.to_datetime(ds["time"].values).to_numpy(dtype="datetime64[ns]"), np.sort(wanted)):
        # Preserve chronological order while requiring exact timestamp set.
        pass

    north, west, south, east = request["area"]
    lat = ds["latitude"].values
    lat_mask = (lat >= south) & (lat <= north)
    if not lat_mask.any():
        raise PipelineError("source fixture has no latitude points in requested area")
    ds = ds.isel(latitude=np.nonzero(lat_mask)[0])

    lon = ds["longitude"].values
    if [north, west, south, east] != [90.0, -180.0, -90.0, 180.0]:
        lon_for_compare = lon.astype(float)
        if lon_for_compare.min() >= 0 and west < 0:
            west_cmp = west % 360
            east_cmp = east % 360
            if west_cmp <= east_cmp:
                lon_mask = (lon_for_compare >= west_cmp) & (lon_for_compare <= east_cmp)
            else:
                lon_mask = (lon_for_compare >= west_cmp) | (lon_for_compare <= east_cmp)
        else:
            lon_mask = (lon_for_compare >= west) & (lon_for_compare <= east)
        if not lon_mask.any():
            raise PipelineError("source fixture has no longitude points in requested area")
        ds = ds.isel(longitude=np.nonzero(lon_mask)[0])
    return ds


def _build_channel_dataset(ds: xr.Dataset, contract: dict[str, Any], request: dict[str, Any], inventory: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    }
    channels: list[dict[str, Any]] = []
    for field in contract["fields"]:
        name = field["name"]
        if name not in ds.data_vars:
            raise PipelineError(f"source fixture lacks requested variable: {name}")
        selector = field["selectors"][0]
        p_text = str(selector["value"])
        native_level = _native_selector_value(ds["pressure_level"].values, p_text)
        fid = canonical_field_id(field)
        array_name = _safe_array_name(name, selector["dimension"], p_text)
        sel_coord = f"{array_name}_pressure_level"
        arr = ds[name].sel(pressure_level=[native_level])
        # Force canonical dimension order and a per-channel selector dimension.
        arr = arr.transpose("time", "pressure_level", "latitude", "longitude")
        arr = arr.rename({"pressure_level": sel_coord})
        arr = arr.assign_coords({sel_coord: ([sel_coord], np.asarray([native_level], dtype=ds["pressure_level"].dtype))})
        arr.attrs = dict(ds[name].attrs)
        meta = inventory.get("option_metadata", {}).get("variable", {}).get(name, {})
        if meta.get("units") and not arr.attrs.get("units"):
            arr.attrs["units"] = meta["units"]
        if meta.get("description") and not arr.attrs.get("description"):
            arr.attrs["description"] = meta["description"]
        arr.attrs.update({"field_id": fid, "source_variable": name, "selector_dimension": "pressure_level", "selector_value": p_text, "selector_unit": "hPa"})
        arr.encoding = {}
        data_vars[array_name] = arr
        coords[sel_coord] = arr[sel_coord]
        channels.append({
            "field_id": fid,
            "array_path": array_name,
            "selectors": {"pressure_level": p_text},
            "selector_coordinate_paths": {"pressure_level": sel_coord},
            "chunks": (1, 1, int(ds.sizes["latitude"]), int(ds.sizes["longitude"])),
        })
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    for c in ("time", "latitude", "longitude"):
        out[c].attrs = dict(ds[c].attrs)
        out[c].encoding = {}
    for ch in channels:
        out[ch["selector_coordinate_paths"]["pressure_level"]].attrs = dict(ds["pressure_level"].attrs)
        out[ch["selector_coordinate_paths"]["pressure_level"]].attrs.setdefault("units", "hPa")
        out[ch["selector_coordinate_paths"]["pressure_level"]].encoding = {}
    out.attrs = {
        "dataset_slug": contract["dataset_slug"],
        "dataset_id": SUPPORTED_DATASET_ID,
        "publication_policy": "regular_grid_zarr_v3_consolidated_per_field_selector_channels",
        "pipeline_id": PIPELINE_ID,
    }
    return out, channels


def _native_selector_value(values: np.ndarray, text: str) -> Any:
    for v in values:
        if _selector_text(v) == text:
            return v.item() if hasattr(v, "item") else v
    raise PipelineError(f"source fixture lacks requested pressure_level {text}")


def _selector_text(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(value)


def canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return str(field["name"])
    body = ",".join(f"{s['dimension']}={json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':'))}" for s in selectors)
    return f"{field['name']}[{body}]"


def _safe_array_name(name: str, dim: str, value: str) -> str:
    raw = f"{name}_{dim}_{value}"
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not safe or not re.match(r"^[A-Za-z_]", safe):
        safe = "channel_" + safe
    return safe


def _encoding_for(ds: xr.Dataset, channels: list[dict[str, Any]], mode: str) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    chunk_by_name = {c["array_path"]: c["chunks"] for c in channels}
    for name in ds.data_vars:
        item = {"chunks": chunk_by_name[name]}
        if mode == "uncompressed":
            item["compressors"] = None
            item["compressor"] = None
        elif mode == "lz4":
            from zarr.codecs import BloscCodec
            item["compressors"] = [BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")]
        enc[name] = item
    for name in ds.coords:
        item = {}
        if name in {"time", "latitude", "longitude"}:
            item["chunks"] = (ds.sizes[name],)
        if mode == "uncompressed":
            item["compressors"] = None
            item["compressor"] = None
        enc[name] = item
    return enc


def _write_zarr_v3(ds: xr.Dataset, store: Path, channels: list[dict[str, Any]]) -> str:
    # First try explicit uncompressed chunks.  Different xarray/zarr releases
    # have accepted either compressor=None or compressors=None; unknown keys are
    # retried in a narrower form below.
    attempts = [
        _encoding_for(ds, channels, "uncompressed"),
        {k: {kk: vv for kk, vv in v.items() if kk != "compressor"} for k, v in _encoding_for(ds, channels, "uncompressed").items()},
        {k: {kk: vv for kk, vv in v.items() if kk != "compressors"} for k, v in _encoding_for(ds, channels, "uncompressed").items()},
    ]
    for enc in attempts:
        if store.exists():
            shutil.rmtree(store)
        try:
            ds.to_zarr(store, mode="w", consolidated=True, zarr_format=3, encoding=enc)
            if _store_is_uncompressed(store, [c["array_path"] for c in channels]):
                return "uncompressed"
        except TypeError:
            continue
        except ValueError:
            continue
    if store.exists():
        shutil.rmtree(store)
    try:
        ds.to_zarr(store, mode="w", consolidated=True, zarr_format=3, encoding=_encoding_for(ds, channels, "lz4"))
        return "lz4"
    except Exception as exc:  # noqa: BLE001
        raise PipelineError("could not write deterministic Zarr v3 output") from exc


def _read_array_metadata(store: Path, array_path: str) -> dict[str, Any]:
    meta_path = store / array_path / "zarr.json"
    if not meta_path.is_file():
        raise PipelineError(f"missing Zarr v3 metadata for array {array_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _store_is_uncompressed(store: Path, array_paths: list[str]) -> bool:
    for p in array_paths:
        codecs = _read_array_metadata(store, p).get("codecs", [])
        for codec in codecs:
            name = str(codec.get("name", codec.get("configuration", {}).get("name", ""))).lower()
            if name in _COMPRESSOR_CODEC_NAMES:
                return False
    return True


def _artifact_layout(channels: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": [
            {
                "field_id": c["field_id"],
                "array_path": c["array_path"],
                "selectors": c["selectors"],
                "selector_coordinate_paths": c["selector_coordinate_paths"],
            }
            for c in channels
        ],
    }


def _validate_published_store(store: Path, expected: xr.Dataset, artifact: dict[str, Any], channels: list[dict[str, Any]], codec_mode: str) -> None:
    root_meta = store / "zarr.json"
    if not root_meta.is_file():
        raise PipelineError("published store is not Zarr v3")
    root = json.loads(root_meta.read_text(encoding="utf-8"))
    if "consolidated_metadata" not in root:
        raise PipelineError("published Zarr metadata is not consolidated")
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        if set(reopened.data_vars) != set(expected.data_vars):
            raise PipelineError("published data variables differ from requested channels")
        for cname, coord in artifact["coordinates"].items():
            if coord not in reopened.coords:
                raise PipelineError(f"missing required coordinate {coord}")
        for ch in channels:
            arr_name = ch["array_path"]
            sel_coord = ch["selector_coordinate_paths"]["pressure_level"]
            arr = reopened[arr_name]
            if arr.dims != ("time", sel_coord, "latitude", "longitude"):
                raise PipelineError("published channel dimension order is invalid")
            if arr.sizes[sel_coord] != 1:
                raise PipelineError("selector dimension cardinality must be one per channel")
            meta = _read_array_metadata(store, arr_name)
            chunks = tuple(meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", ()))
            if chunks != tuple(ch["chunks"]):
                raise PipelineError("published channel chunks are not sample-access aligned")
            if codec_mode == "uncompressed" and not _store_is_uncompressed(store, [arr_name]):
                raise PipelineError("published channel unexpectedly uses compression")
            if codec_mode == "lz4" and "lz4" not in json.dumps(meta).lower():
                raise PipelineError("LZ4 fallback metadata is missing")
            np.testing.assert_array_equal(arr.values, expected[arr_name].values)
            np.testing.assert_array_equal(reopened[sel_coord].values, expected[sel_coord].values)
        for coord in ("time", "latitude", "longitude"):
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
        # Lightweight C,H,W-equivalent read path over all declared channels.
        sample_arrays = []
        for ch in artifact["channels"]:
            arr = reopened[ch["array_path"]].isel(time=0)
            sel_coord = ch["selector_coordinate_paths"]["pressure_level"]
            sample_arrays.append(arr.isel({sel_coord: 0}).values)
        stacked = np.stack(sample_arrays, axis=0)
        if stacked.shape != (len(channels), expected.sizes["latitude"], expected.sizes["longitude"]):
            raise PipelineError("C,H,W sample read shape is invalid")
    finally:
        reopened.close()
