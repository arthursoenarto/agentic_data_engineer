"""Standalone ERA5 pressure-level fixture adapter.

Runtime entry point: run_pipeline(contract_lock, inventory, cache_dir, output_dir)
This module performs no network access and has no repository-local runtime dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p013-search-r1"
CONTRACT_SCHEMA = "dataset_contract.v1"
INVENTORY_SCHEMA = "dataset_inventory.v1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
FIXTURE_MANIFEST = "source_fixture_manifest.json"
STORE_NAME = "dataset.zarr"

_DIM_ALIASES = {
    "valid_time": "time",
    "forecast_reference_time": "time",
    "isobaricInhPa": "pressure_level",
    "isobaricinhpa": "pressure_level",
    "level": "pressure_level",
    "plev": "pressure_level",
    "lat": "latitude",
    "lon": "longitude",
}
_VAR_ALIASES = {
    "t": "temperature",
    "z": "geopotential",
}


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a runtime lock, consume a verified local source fixture, and publish Zarr v3.

    The cache fixture is authoritative for this pilot. If no complete fixture manifest is
    present, this adapter raises rather than probing external services.
    """
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    contract = _extract_contract(contract_lock)
    request = _validate_contract_and_inventory(contract, inventory)

    fixture_entries = _verify_fixture(cache_root)
    source_ds = _open_source_fixture(cache_root, fixture_entries)
    source_ds = _normalize_source_dataset(source_ds)

    filtered = _build_public_dataset(source_ds, contract, inventory, request)
    filtered = _sanitize_for_publication(filtered)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root = out_root / ".tmp-pipeline-publish"
    tmp_store = tmp_root / STORE_NAME
    final_store = out_root / STORE_NAME
    _clean_path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    encoding, codec_mode = _encoding_for(filtered)
    try:
        _write_zarr_v3(filtered, tmp_store, encoding)
    except TypeError:
        # Older maintained xarray/zarr combinations may spell the no-compressor
        # option differently. Retry with conservative compressor=None syntax.
        encoding = _encoding_for(filtered, legacy=True)[0]
        codec_mode = "uncompressed"
        _clean_path(tmp_store)
        _write_zarr_v3(filtered, tmp_store, encoding)
    except Exception as exc:
        # Only fall back for known compression-API incompatibilities. The fallback
        # is read-speed-biased lossless LZ4 and never quantizes or repacks data.
        if not _looks_like_codec_api_issue(exc):
            raise
        encoding, codec_mode = _encoding_for(filtered, lz4_fallback=True)
        _clean_path(tmp_store)
        _write_zarr_v3(filtered, tmp_store, encoding)

    artifact = _artifact_declaration(filtered, contract, request)
    _post_write_validate(tmp_store, filtered, artifact, codec_mode)
    _atomic_replace_dir(tmp_store, final_store)
    _clean_path(tmp_root)

    # Reopen the final path as the consumer will see it.
    _post_write_validate(final_store, filtered, artifact, codec_mode)

    safe_keys = [f"fixture:{e['entry_id']}:sha256:{e['sha256'][:16]}:size:{e['size_bytes']}" for e in fixture_entries]
    return {
        "cache": {
            "hits": len(fixture_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": safe_keys,
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [f"zarr_data_codec={codec_mode}"],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise ValueError("contract_lock must be an object")
    if lock.get("schema_version") == CONTRACT_SCHEMA:
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "DatasetContract"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == CONTRACT_SCHEMA:
            return value
    if isinstance(lock.get("contracts"), list):
        matches = [c for c in lock["contracts"] if isinstance(c, dict) and c.get("schema_version") == CONTRACT_SCHEMA and c.get("dataset_slug") == DATASET_SLUG]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("could not locate selected DatasetContract in lock envelope")


def _validate_contract_and_inventory(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unsupported inventory schema_version")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("unsupported contract schema_version")
    if contract.get("dataset_slug") != inventory.get("dataset_slug") or contract.get("dataset_slug") != DATASET_SLUG:
        raise ValueError("dataset identity does not match frozen inventory")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ValueError("inventory dataset_id mismatch")
    if contract.get("human_confirmed") is not True:
        raise ValueError("contract must be human_confirmed")

    options = inventory.get("options", {})
    advanced = contract.get("advanced_options") or {}
    if advanced.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ValueError("advanced_options.dataset_id mismatch")
    for key in ("data_format", "download_format"):
        value = advanced.get(key, inventory.get("defaults", {}).get(key))
        if value not in options.get(key, []):
            raise ValueError(f"unsupported {key}: {value!r}")
    product_type = advanced.get("product_type", contract.get("scope", {}).get("product_type", ["reanalysis"]))
    if isinstance(product_type, str):
        product_values = [product_type]
    else:
        product_values = list(product_type)
    if not product_values or any(v not in options.get("product_type", []) for v in product_values):
        raise ValueError("unsupported product_type")
    if product_values != ["reanalysis"]:
        raise ValueError("this fixed policy supports ERA5 reanalysis product_type only")

    fields = contract.get("fields") or []
    if not fields:
        raise ValueError("contract must request at least one field")
    channels: list[dict[str, Any]] = []
    for field in fields:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise ValueError(f"unsupported variable: {name!r}")
        selectors = field.get("selectors") or []
        if not selectors:
            channels.append({"name": name, "selectors": []})
            continue
        for sel in selectors:
            if sel.get("dimension") != "pressure_level":
                raise ValueError(f"unsupported selector dimension for {name}: {sel.get('dimension')!r}")
            value = str(sel.get("value"))
            if value not in options.get("pressure_level", []):
                raise ValueError(f"unsupported pressure_level: {value!r}")
            if sel.get("unit") not in (None, "hPa"):
                raise ValueError("pressure_level selector unit must be hPa when provided")
        channels.append({"name": name, "selectors": selectors})

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise ValueError("end_date precedes start_date")
    inv_years = set(options.get("year", []))
    years = {str(y) for y in range(start.year, end.year + 1)}
    if not years.issubset(inv_years):
        raise ValueError("date_range falls outside inventory years")
    cat_interval = (((inventory.get("catalogue_metadata") or {}).get("extent") or {}).get("temporal") or {}).get("interval") or []
    if cat_interval:
        lo = datetime.fromisoformat(cat_interval[0][0].replace("Z", "+00:00")).date()
        hi = datetime.fromisoformat(cat_interval[0][1].replace("Z", "+00:00")).date()
        if start < lo or end > hi:
            raise ValueError("date_range outside catalogue temporal extent")

    t_scope = scope.get("time") or {}
    if t_scope.get("timezone", "UTC") != "UTC":
        raise ValueError("only UTC selected_times are supported")
    selected_times = list(t_scope.get("selected_times") or [])
    if not selected_times:
        raise ValueError("scope.time.selected_times is required")
    for t in selected_times:
        if t not in options.get("time", []):
            raise ValueError(f"unsupported selected time: {t!r}")
        _parse_hhmm(t)

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise ValueError("geography.cds_area must be [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError("geography.cds_area is outside valid latitude/longitude bounds")

    timestamps = _requested_timestamps(start, end, bool(dr.get("inclusive", True)), selected_times)
    return {"channels": channels, "timestamps": timestamps, "area": [north, west, south, east]}


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute), tzinfo=timezone.utc)


def _requested_timestamps(start: date, end: date, inclusive: bool, selected_times: list[str]) -> np.ndarray:
    # Inclusive date-only end means the complete final UTC calendar day; selected
    # times are then applied exactly. Exclusive end omits the end calendar date.
    stop = end if inclusive else (end - timedelta(days=1))
    out: list[np.datetime64] = []
    d = start
    while d <= stop:
        for hhmm in selected_times:
            tm = _parse_hhmm(hhmm)
            out.append(np.datetime64(datetime(d.year, d.month, d.day, tm.hour, tm.minute), "ns"))
        d += timedelta(days=1)
    return np.array(out, dtype="datetime64[ns]")


def _verify_fixture(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / FIXTURE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError("complete source fixture manifest is required; network fallback is disabled")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ValueError("unsupported source fixture manifest schema_version")
    entries = manifest.get("entries") or []
    if not entries:
        raise ValueError("source fixture manifest contains no entries")
    verified: list[dict[str, Any]] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise ValueError("fixture entry relative_path must be relative")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError("fixture entry escapes cache_dir") from exc
        if not path.exists() or not path.is_file():
            raise FileNotFoundError("fixture raw file is missing")
        size = path.stat().st_size
        expected_size = int(entry.get("size_bytes"))
        if size != expected_size or size <= 0:
            raise ValueError("fixture raw file size mismatch")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_digest = str(entry.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest) or digest != expected_digest:
            raise ValueError("fixture raw file SHA-256 mismatch")
        verified.append({
            "entry_id": str(entry.get("entry_id", rel)),
            "relative_path": rel,
            "path": path,
            "size_bytes": size,
            "sha256": digest,
        })
    return verified


def _open_source_fixture(cache_root: Path, entries: list[dict[str, Any]]) -> xr.Dataset:
    paths = [e["path"] for e in entries if Path(e["path"]).name != FIXTURE_MANIFEST]
    zarrs = [p for p in paths if p.suffix == ".zarr" or p.name.endswith(".zarr")]
    netcdfs = [p for p in paths if p.suffix.lower() in {".nc", ".nc4", ".cdf", ".netcdf"}]
    gribs = [p for p in paths if p.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"}]
    if zarrs:
        if len(zarrs) != 1:
            raise ValueError("fixture may contain at most one source Zarr store")
        return xr.open_zarr(zarrs[0], consolidated=False).load()
    if netcdfs:
        datasets = [xr.open_dataset(p, decode_cf=True, mask_and_scale=True).load() for p in sorted(netcdfs, key=lambda x: x.as_posix())]
        return _combine_datasets(datasets)
    if gribs:
        datasets = []
        for p in sorted(gribs, key=lambda x: x.as_posix()):
            datasets.append(xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_cf=True, mask_and_scale=True).load())
        return _combine_datasets(datasets)
    raise ValueError("fixture manifest does not list a supported raw data file")


def _combine_datasets(datasets: list[xr.Dataset]) -> xr.Dataset:
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts").load()
    except Exception:
        return xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts").load()


def _normalize_source_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for name in list(ds.dims) + list(ds.coords):
        key = name if name in _DIM_ALIASES else name.lower()
        if key in _DIM_ALIASES and _DIM_ALIASES[key] not in ds:
            rename[name] = _DIM_ALIASES[key]
    for name in ds.data_vars:
        canonical = _VAR_ALIASES.get(name)
        if canonical and canonical not in ds:
            rename[name] = canonical
        elif ds[name].attrs.get("GRIB_shortName") in _VAR_ALIASES:
            canonical = _VAR_ALIASES[ds[name].attrs["GRIB_shortName"]]
            if canonical not in ds:
                rename[name] = canonical
    if rename:
        ds = ds.rename(rename)
    for required in ("time", "latitude", "longitude"):
        if required not in ds.coords and required not in ds.dims:
            raise ValueError(f"source fixture is missing required coordinate {required!r}")
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].astype("datetime64[ns]"))
    return ds


def _build_public_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> xr.Dataset:
    ds = _filter_time(ds, request["timestamps"])
    ds = _filter_area(ds, request["area"])

    groups: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for field in contract["fields"]:
        groups.setdefault(field["name"], []).append(field)

    out_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    }
    meta = inventory.get("option_metadata", {}).get("variable", {})
    for name, fields in groups.items():
        if name not in ds.data_vars:
            raise ValueError(f"source fixture lacks requested variable {name!r}")
        da = ds[name]
        if "time" not in da.dims or "latitude" not in da.dims or "longitude" not in da.dims:
            raise ValueError(f"source variable {name!r} lacks required grid dimensions")
        selectors = [sel for f in fields for sel in (f.get("selectors") or [])]
        if selectors:
            requested = _unique_preserve([str(sel["value"]) for sel in selectors])
            if "pressure_level" not in da.dims and "pressure_level" not in da.coords:
                if len(requested) != 1:
                    raise ValueError(f"source variable {name!r} lacks pressure_level selector dimension")
                try:
                    native_value = int(requested[0])
                except ValueError:
                    try:
                        native_value = float(requested[0])
                    except ValueError:
                        native_value = requested[0]
                da = da.expand_dims(pressure_level=[native_value])
            else:
                native_values = _native_coord_values(ds["pressure_level"].values, requested)
                da = da.sel(pressure_level=native_values)
            selector_dim = f"{name}_pressure_level"
            da = da.rename({"pressure_level": selector_dim})
            coords[selector_dim] = da[selector_dim]
            da[selector_dim].attrs.update({"units": "hPa", "standard_name": "air_pressure", "selector_dimension": "pressure_level"})
        for extra_dim in list(da.dims):
            if extra_dim not in {"time", "latitude", "longitude", f"{name}_pressure_level"}:
                if da.sizes[extra_dim] == 1:
                    da = da.isel({extra_dim: 0}, drop=True)
                else:
                    raise ValueError(f"source variable {name!r} contains unsupported extra dimension {extra_dim!r}")
        order = [d for d in ("time", f"{name}_pressure_level", "latitude", "longitude") if d in da.dims]
        da = da.transpose(*order)
        attrs = dict(da.attrs)
        attrs.setdefault("long_name", meta.get(name, {}).get("label", name))
        if meta.get(name, {}).get("units") and "units" not in attrs:
            attrs["units"] = _strip_html_units(meta[name]["units"])
        if meta.get(name, {}).get("description") and "description" not in attrs:
            attrs["description"] = meta[name]["description"]
        da.attrs = attrs
        out_vars[name] = da
    public = xr.Dataset(out_vars, coords=coords, attrs={
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "provider": "ECMWF",
        "source": "verified local fixture",
        "publication_policy": "regular_grid_zarr_output_policy.v1",
        "zarr_format": "3",
    })
    return public


def _filter_time(ds: xr.Dataset, requested: np.ndarray) -> xr.Dataset:
    source_times = ds["time"].values.astype("datetime64[ns]")
    missing = [str(t) for t in requested if t not in source_times]
    if missing:
        raise ValueError("source fixture does not contain all requested timestamps")
    return ds.sel(time=requested)


def _filter_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    lat = ds["latitude"]
    lon = ds["longitude"]
    # Full global request: keep native longitude convention/order exactly.
    if north == 90 and south == -90 and west in (-180, 0) and east in (180, 360):
        return ds.sel(latitude=_lat_slice(lat, north, south))
    out = ds.sel(latitude=_lat_slice(lat, north, south))
    lon_vals = lon.values.astype(float)
    if lon_vals.min() >= 0 and west < 0:
        west = west % 360
        east = east % 360
    if west <= east:
        out = out.sel(longitude=slice(west, east))
    else:
        left = out.sel(longitude=slice(west, float(lon_vals.max())))
        right = out.sel(longitude=slice(float(lon_vals.min()), east))
        out = xr.concat([left, right], dim="longitude")
    if out.sizes.get("latitude", 0) == 0 or out.sizes.get("longitude", 0) == 0:
        raise ValueError("geography filter selected no grid cells")
    return out


def _lat_slice(lat: xr.DataArray, north: float, south: float) -> slice:
    vals = lat.values.astype(float)
    if vals[0] > vals[-1]:
        return slice(north, south)
    return slice(south, north)


def _native_coord_values(values: np.ndarray, requested: list[str]) -> list[Any]:
    result = []
    str_to_value = {str(v.item() if hasattr(v, "item") else v): v for v in values}
    for req in requested:
        if req not in str_to_value:
            # Numeric string normalization, e.g. 500 vs 500.0.
            found = None
            for v in values:
                scalar = v.item() if hasattr(v, "item") else v
                try:
                    if float(scalar) == float(req):
                        found = v
                        break
                except Exception:
                    pass
            if found is None:
                raise ValueError(f"source fixture lacks requested pressure_level {req}")
            result.append(found)
        else:
            result.append(str_to_value[req])
    return result


def _unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _strip_html_units(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).replace("sup>-", "-")


def _sanitize_for_publication(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=True)
    for name in list(ds.data_vars) + list(ds.coords):
        obj = ds[name]
        for key in ("scale_factor", "add_offset", "_Unsigned"):
            obj.encoding.pop(key, None)
            obj.attrs.pop(key, None)
        obj.encoding.pop("source", None)
        obj.encoding.pop("original_shape", None)
        obj.encoding.pop("chunksizes", None)
        obj.encoding.pop("preferred_chunks", None)
        if obj.dtype.kind in "fc":
            obj.encoding["_FillValue"] = None
        obj.attrs = _json_safe_attrs(obj.attrs)
    ds.attrs = _json_safe_attrs(ds.attrs)
    return ds


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if k.lower() in {"history"}:
            continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, (str, int, float, bool)) or v is None or isinstance(v, (list, tuple, dict)):
            try:
                json.dumps(v)
                out[str(k)] = v
            except TypeError:
                out[str(k)] = str(v)
        else:
            out[str(k)] = str(v)
    return out


def _encoding_for(ds: xr.Dataset, legacy: bool = False, lz4_fallback: bool = False) -> tuple[dict[str, dict[str, Any]], str]:
    enc: dict[str, dict[str, Any]] = {}
    lat_n = int(ds.sizes["latitude"])
    lon_n = int(ds.sizes["longitude"])
    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim.endswith("_pressure_level"):
                chunks.append(1)
            elif dim == "latitude":
                chunks.append(lat_n)
            elif dim == "longitude":
                chunks.append(lon_n)
            else:
                chunks.append(int(da.sizes[dim]))
        entry: dict[str, Any] = {"chunks": tuple(chunks)}
        if lz4_fallback:
            from zarr.codecs import BloscCodec
            entry["compressors"] = [BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")]
        elif legacy:
            entry["compressor"] = None
            entry["compressors"] = None
        else:
            entry["compressors"] = None
        if da.dtype.kind in "fc":
            entry["_FillValue"] = None
        enc[name] = entry
    for name in ds.coords:
        enc[name] = {"chunks": tuple(ds[name].shape) if ds[name].shape else (), "compressors": None}
    return enc, ("blosc_lz4_clevel1_shuffle" if lz4_fallback else "uncompressed")


def _write_zarr_v3(ds: xr.Dataset, store: Path, encoding: dict[str, dict[str, Any]]) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_zarr(store, mode="w", encoding=encoding, consolidated=True, zarr_format=3, compute=True)
    except TypeError as exc:
        if "zarr_format" not in str(exc):
            raise
        ds.to_zarr(store, mode="w", encoding=encoding, consolidated=True, zarr_version=3, compute=True)


def _looks_like_codec_api_issue(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in ("compressor", "compressors", "codec", "bytesbytescodec", "zarr v3"))


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return str(field["name"])
    parts = [f"{s['dimension']}={json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':'))}" for s in selectors]
    return f"{field['name']}[" + ",".join(parts) + "]"


def _artifact_declaration(ds: xr.Dataset, contract: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    channels = []
    for field in contract["fields"]:
        name = field["name"]
        selectors = field.get("selectors") or []
        selector_map: dict[str, str] = {}
        selector_paths: dict[str, str] = {}
        if selectors:
            for sel in selectors:
                selector_map[sel["dimension"]] = str(sel["value"])
                selector_paths[sel["dimension"]] = f"{name}_pressure_level"
        channels.append({
            "field_id": _canonical_field_id(field),
            "array_path": name,
            "selectors": selector_map,
            "selector_coordinate_paths": selector_paths,
        })
    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": channels,
    }


def _post_write_validate(store: Path, expected: xr.Dataset, artifact: dict[str, Any], codec_mode: str) -> None:
    if not (store / "zarr.json").exists():
        raise ValueError("published store is not Zarr v3")
    root_meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    if root_meta.get("zarr_format") != 3:
        raise ValueError("published store is not Zarr format 3")
    if "consolidated_metadata" not in root_meta:
        raise ValueError("Zarr v3 consolidated metadata is missing")
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    if set(reopened.data_vars) != set(expected.data_vars):
        raise ValueError("reopened data variables differ from expected")
    for coord in expected.coords:
        if coord not in reopened.coords:
            raise ValueError(f"reopened coordinate missing: {coord}")
        np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
    for name in expected.data_vars:
        if reopened[name].dims != expected[name].dims:
            raise ValueError(f"reopened dimensions differ for {name}")
        np.testing.assert_array_equal(reopened[name].values, expected[name].values)
        _validate_chunk_metadata(store, name, expected[name], codec_mode)
    _validate_artifact_paths(reopened, artifact)


def _validate_chunk_metadata(store: Path, name: str, da: xr.DataArray, codec_mode: str) -> None:
    meta_path = store / name / "zarr.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
    expected = []
    for dim in da.dims:
        if dim == "time" or dim.endswith("_pressure_level"):
            expected.append(1)
        else:
            expected.append(int(da.sizes[dim]))
    if list(chunk_shape) != expected:
        raise ValueError(f"unexpected chunk shape for {name}: {chunk_shape} != {expected}")
    codecs = json.dumps(meta.get("codecs", []), sort_keys=True).lower()
    if codec_mode == "uncompressed":
        if any(token in codecs for token in ("zstd", "gzip", "blosc", "lz4")):
            raise ValueError(f"data chunks for {name} are unexpectedly compressed")
    else:
        if "blosc" not in codecs or "lz4" not in codecs or "clevel" not in codecs:
            raise ValueError("LZ4 fallback codec metadata is missing")


def _validate_artifact_paths(ds: xr.Dataset, artifact: dict[str, Any]) -> None:
    for ch in artifact["channels"]:
        array_path = ch["array_path"]
        if array_path not in ds.data_vars:
            raise ValueError(f"artifact channel array_path is not a data variable: {array_path}")
        for selector_dim, coord_path in ch.get("selector_coordinate_paths", {}).items():
            if coord_path not in ds.coords:
                raise ValueError(f"artifact selector coordinate path missing: {coord_path}")
            value = str(ch["selectors"][selector_dim])
            coord_values = [str(v.item() if hasattr(v, "item") else v) for v in ds[coord_path].values]
            if value not in coord_values and not any(_float_equal_string(v, value) for v in coord_values):
                raise ValueError("artifact selector value not present in selector coordinate")


def _float_equal_string(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except Exception:
        return False


def _atomic_replace_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError("temporary publication store missing")
    backup = dst.with_name(dst.name + ".old")
    _clean_path(backup)
    if dst.exists():
        os.replace(dst, backup)
    os.replace(src, dst)
    _clean_path(backup)


def _clean_path(path: Path) -> None:
    if path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
