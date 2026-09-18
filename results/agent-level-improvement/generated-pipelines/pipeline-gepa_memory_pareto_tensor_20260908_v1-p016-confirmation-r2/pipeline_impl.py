"""Standalone ERA5 pressure-level fixture adapter.

Entry point required by the framework:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The implementation is intentionally offline-only for this pilot family.  It
verifies a framework-provided source_fixture_manifest.json before opening any raw
source file, filters exclusively from the runtime contract lock, decodes CF
packing/missing conventions through xarray, and publishes consolidated Zarr v3.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p016-confirmation-r2"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "dataset.zarr"

_FIELD_SOURCE_NAMES = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
}

_DIM_ALIASES = {
    "time": ("time", "valid_time"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "isobaricInPa", "level", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}


class PipelineError(ValueError):
    """Secret-safe deterministic pipeline error."""


@dataclass(frozen=True)
class ChannelRequest:
    field_name: str
    field_id: str
    selector_dimension: str | None
    selector_value: str | None
    selector_unit: str | None


@dataclass(frozen=True)
class PublicationCodec:
    name: str
    encoding_value: Any
    warning: str | None = None


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize the selected ERA5 pressure-level request from a frozen fixture.

    Parameters are framework-owned objects.  This function never contacts a
    provider, never reads credentials, and never writes into cache_dir.
    """
    contract = _extract_contract(contract_lock)
    _validate_inventory(inventory)
    requests = _validate_contract(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    fixture_entries = _verify_source_fixture(cache_root)

    requested_field_names = sorted({r.field_name for r in requests})
    source = _open_fixture_dataset([entry["path"] for entry in fixture_entries], requested_field_names)
    source = _normalise_source_dataset(source)

    target_times = _requested_timestamps(contract)
    area = contract["scope"]["geography"]["cds_area"]

    published_vars: dict[str, xr.DataArray] = {}
    selector_coord_paths: dict[str, str] = {}
    grouped: dict[str, list[ChannelRequest]] = {}
    for req in requests:
        grouped.setdefault(req.field_name, []).append(req)

    # Stable output order: contract field first occurrence order, then variable name.
    ordered_field_names: list[str] = []
    for req in requests:
        if req.field_name not in ordered_field_names:
            ordered_field_names.append(req.field_name)

    for field_name in ordered_field_names:
        reqs = grouped[field_name]
        var_name = _resolve_source_variable(source, field_name)
        da = source[var_name]
        da = _ensure_required_grid_dims(da, field_name)
        da = _filter_time(da, target_times)
        da = _filter_area(da, area)

        selector_dim = reqs[0].selector_dimension
        if selector_dim is not None:
            if any(r.selector_dimension != selector_dim for r in reqs):
                raise PipelineError(f"mixed selector dimensions are not supported for {field_name}")
            if selector_dim != "pressure_level":
                raise PipelineError(f"unsupported selector dimension {selector_dim!r}")
            selected_values = [r.selector_value for r in reqs]
            assert all(v is not None for v in selected_values)
            da = _filter_pressure_levels(da, [str(v) for v in selected_values])
            out_level_dim = f"{field_name}_pressure_level"
            da = da.rename({"pressure_level": out_level_dim})
            selector_coord_paths[field_name] = out_level_dim
            da = da.transpose("time", out_level_dim, "latitude", "longitude")
        else:
            da = da.transpose("time", "latitude", "longitude")

        # Publish under canonical long field name and decoded native values.  Strip
        # source encodings so xarray cannot silently re-pack scale/offset/fill values.
        da = da.rename(field_name).load()
        da.encoding.clear()
        da.attrs = dict(da.attrs)
        published_vars[field_name] = da

    published = xr.Dataset(published_vars)
    published.attrs.update({
        "dataset_slug": contract["dataset_slug"],
        "dataset_id": DATASET_ID,
        "provider": inventory.get("provider", "ECMWF"),
        "publication_format": "zarr-v3-consolidated",
        "pipeline_id": PIPELINE_ID,
        "source_fixture_entry_ids": ",".join(entry["entry_id"] for entry in fixture_entries),
    })
    _clear_all_encodings(published)

    artifact_channels = _artifact_channels(requests, selector_coord_paths)
    chunks = _encoding_chunks(published)

    output_root.mkdir(parents=True, exist_ok=True)
    final_store = output_root / STORE_NAME
    tmp_store = output_root / f".{STORE_NAME}.tmp"
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    codec, warnings = _write_zarr_v3_with_preferred_codec(published, tmp_store, chunks)
    reopened = xr.open_zarr(tmp_store, consolidated=True)
    try:
        _assert_dataset_semantics_equal(published, reopened)
        _assert_chunk_layout(tmp_store, published, chunks)
        _assert_codec_policy(tmp_store, published.data_vars, codec.name)
    finally:
        reopened.close()

    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)

    reopened_final = xr.open_zarr(final_store, consolidated=True)
    try:
        _assert_dataset_semantics_equal(published, reopened_final)
    finally:
        reopened_final.close()

    if codec.warning:
        warnings.append(codec.warning)

    return {
        "cache": {
            "hits": len(fixture_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [entry["cache_key"] for entry in fixture_entries],
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": STORE_NAME,
            "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
            "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
            "channels": artifact_channels,
        },
        "warnings": warnings,
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    if isinstance(lock, dict):
        for key in ("contract", "dataset_contract", "selected_contract", "contract_snapshot"):
            value = lock.get(key)
            if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
                return value
        for key in ("contracts", "dataset_contracts"):
            value = lock.get(key)
            if isinstance(value, list):
                matches = [v for v in value if isinstance(v, dict) and v.get("schema_version") == "dataset_contract.v1"]
                if len(matches) == 1:
                    return matches[0]
                selected_id = lock.get("selected_contract_id") or lock.get("contract_id")
                if selected_id is not None:
                    for item in matches:
                        if item.get("contract_id") == selected_id or item.get("id") == selected_id:
                            return item
        for value in lock.values():
            try:
                return _extract_contract(value)
            except PipelineError:
                pass
    if isinstance(lock, list):
        matches = []
        for value in lock:
            try:
                matches.append(_extract_contract(value))
            except PipelineError:
                pass
        if len(matches) == 1:
            return matches[0]
    raise PipelineError("no dataset_contract.v1 object found in contract lock")


def _validate_inventory(inventory: dict[str, Any]) -> None:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("inventory schema_version must be dataset_inventory.v1")
    if inventory.get("dataset_slug") != DATASET_SLUG:
        raise PipelineError("inventory dataset_slug mismatch")
    if inventory.get("dataset_id") != DATASET_ID:
        raise PipelineError("inventory dataset_id mismatch")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> list[ChannelRequest]:
    if contract.get("schema_version") != "dataset_contract.v1":
        raise PipelineError("contract schema_version must be dataset_contract.v1")
    if contract.get("dataset_slug") != inventory.get("dataset_slug"):
        raise PipelineError("contract dataset_slug is not supported by this inventory")
    if not contract.get("human_confirmed", False):
        raise PipelineError("contract must be human_confirmed")

    opts = inventory.get("options", {})
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id") != DATASET_ID:
        raise PipelineError("advanced_options.dataset_id mismatch")
    _validate_option_value("data_format", adv.get("data_format"), opts, scalar=True)
    _validate_option_value("download_format", adv.get("download_format"), opts, scalar=True)
    _validate_option_value("product_type", adv.get("product_type"), opts, scalar=False)

    scope = contract.get("scope") or {}
    if scope.get("product_type") not in opts.get("product_type", []):
        raise PipelineError("unsupported scope.product_type")
    time_scope = scope.get("time") or {}
    if time_scope.get("timezone") != "UTC":
        raise PipelineError("only UTC selected_times are supported")
    selected_times = time_scope.get("selected_times")
    if not isinstance(selected_times, list) or not selected_times:
        raise PipelineError("scope.time.selected_times must be a non-empty list")
    for t in selected_times:
        _validate_option_value("time", t, opts, scalar=True)
        if not re.fullmatch(r"\d{2}:\d{2}", str(t)):
            raise PipelineError("selected time must be HH:MM")

    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise PipelineError("date_range end_date precedes start_date")
    for year in range(start.year, end.year + 1):
        _validate_option_value("year", f"{year:04d}", opts, scalar=True)

    geo = scope.get("geography") or {}
    area = geo.get("cds_area")
    if not isinstance(area, list) or len(area) != 4 or not all(isinstance(x, (int, float)) for x in area):
        raise PipelineError("geography.cds_area must be four numeric values")
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90):
        raise PipelineError("latitude bounds are invalid")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("longitude bounds are invalid")
    if geo.get("cds_area_order") not in (None, ["north", "west", "south", "east"]):
        raise PipelineError("unsupported cds_area_order")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise PipelineError("contract.fields must be non-empty")
    requests: list[ChannelRequest] = []
    for field in fields:
        name = field.get("name")
        _validate_option_value("variable", name, opts, scalar=True)
        selectors = field.get("selectors") or []
        if len(selectors) > 1:
            raise PipelineError("only one selector dimension per ERA5 pressure-level field is supported")
        if selectors:
            sel = selectors[0]
            if sel.get("dimension") != "pressure_level":
                raise PipelineError("only pressure_level selectors are supported")
            val = str(sel.get("value"))
            _validate_option_value("pressure_level", val, opts, scalar=True)
            unit = sel.get("unit")
            if unit not in (None, "hPa"):
                raise PipelineError("pressure_level selector unit must be hPa")
            selector_dimension = "pressure_level"
            selector_value = val
            selector_unit = unit
        else:
            selector_dimension = selector_value = selector_unit = None
        requests.append(ChannelRequest(str(name), _canonical_field_id(str(name), selectors), selector_dimension, selector_value, selector_unit))

    # Allowed combination for this inventory: requested variable x pressure_level.
    for req in requests:
        if req.field_name not in opts.get("variable", []):
            raise PipelineError(f"unsupported variable {req.field_name}")
        if req.selector_dimension != "pressure_level":
            raise PipelineError("ERA5 pressure-level variables require pressure_level selectors")
    return requests


def _validate_option_value(name: str, value: Any, opts: dict[str, Any], *, scalar: bool) -> None:
    allowed = opts.get(name)
    if not isinstance(allowed, list):
        raise PipelineError(f"inventory missing options.{name}")
    if scalar:
        if value not in allowed:
            raise PipelineError(f"unsupported {name} value")
    else:
        if not isinstance(value, list) or not value:
            raise PipelineError(f"{name} must be a non-empty list")
        for item in value:
            if item not in allowed:
                raise PipelineError(f"unsupported {name} value")


def _canonical_field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    parts = []
    for sel in selectors:
        parts.append(f"{sel['dimension']}={json.dumps(str(sel['value']), ensure_ascii=False, sort_keys=True)}")
    return f"{name}[{','.join(parts)}]"


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise PipelineError(f"{name} must be ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineError(f"{name} must be ISO date string") from exc


def _requested_timestamps(contract: dict[str, Any]) -> pd.DatetimeIndex:
    scope = contract["scope"]
    dr = scope["date_range"]
    start = _parse_date(dr["start_date"], "start_date")
    end = _parse_date(dr["end_date"], "end_date")
    inclusive = bool(dr.get("inclusive", True))
    last = end if inclusive else end - timedelta(days=1)
    if last < start:
        raise PipelineError("exclusive date range selects no complete UTC calendar days")
    times = []
    for t in scope["time"]["selected_times"]:
        hour, minute = map(int, str(t).split(":"))
        times.append(time(hour, minute, tzinfo=timezone.utc))
    stamps: list[pd.Timestamp] = []
    cur = start
    while cur <= last:
        for tod in times:
            stamps.append(pd.Timestamp(datetime.combine(cur, tod)).tz_convert(None))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(stamps).sort_values()


def _verify_source_fixture(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise PipelineError("complete source_fixture_manifest.json is required; network fallback is disabled")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("invalid source fixture manifest schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("source fixture manifest must contain at least one entry")
    verified: list[dict[str, Any]] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise PipelineError("fixture entry relative_path must be relative")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineError("fixture entry escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineError("fixture entry file is missing")
        expected_size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise PipelineError("fixture entry size_bytes must be positive integer")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise PipelineError("fixture entry sha256 must be lowercase hex")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise PipelineError("fixture entry size mismatch")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise PipelineError("fixture entry sha256 mismatch")
        entry_id = str(entry.get("entry_id", rel))
        verified.append({
            "entry_id": entry_id,
            "path": path,
            "cache_key": f"fixture:{entry_id}:sha256:{actual_sha[:16]}",
        })
    return sorted(verified, key=lambda e: e["entry_id"])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_fixture_dataset(paths: list[Path], requested_field_names: list[str]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".grib", ".grb", ".grb2"}:
            opened_any = False
            for field in requested_field_names:
                for candidate in _FIELD_SOURCE_NAMES.get(field, (field,)):
                    try:
                        ds = xr.open_dataset(
                            path,
                            engine="cfgrib",
                            decode_cf=True,
                            mask_and_scale=True,
                            backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": candidate}},
                        ).load()
                    except Exception:
                        continue
                    if ds.data_vars:
                        datasets.append(_normalise_source_dataset(ds))
                        opened_any = True
                        break
            if not opened_any:
                raise PipelineError("unable to open requested variables from GRIB fixture")
        else:
            last_error: Exception | None = None
            for kwargs in ({"decode_cf": True, "mask_and_scale": True}, {"engine": "h5netcdf", "decode_cf": True, "mask_and_scale": True}, {"engine": "scipy", "decode_cf": True, "mask_and_scale": True}):
                try:
                    datasets.append(xr.open_dataset(path, **kwargs).load())
                    last_error = None
                    break
                except Exception as exc:  # try next maintained backend
                    last_error = exc
            if last_error is not None:
                raise PipelineError("unable to open fixture file with installed xarray backends") from last_error
    if not datasets:
        raise PipelineError("no fixture datasets opened")
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="override").load()
    except Exception:
        return xr.merge(datasets, compat="override", combine_attrs="override").load()


def _normalise_source_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    names = set(ds.dims) | set(ds.coords) | set(ds.variables)
    for canonical, aliases in _DIM_ALIASES.items():
        if canonical in names:
            continue
        for alias in aliases:
            if alias in names:
                rename[alias] = canonical
                names.add(canonical)
                break
    out = ds.rename(rename) if rename else ds
    # Pa pressure levels are uncommon in ERA5 pressure-level fixtures but can be
    # decoded from some NetCDF exports; convert coordinate values only when the
    # alias made the unit unambiguous.
    if "pressure_level" in out.coords:
        coord = out["pressure_level"]
        units = str(coord.attrs.get("units", "")).lower()
        if units in {"pa", "pascal", "pascals"} and np.nanmax(coord.values.astype(float)) > 2000:
            out = out.assign_coords(pressure_level=(coord.dims, coord.values.astype(float) / 100.0, {**coord.attrs, "units": "hPa"}))
    if "pressure_level" in out.coords and "pressure_level" not in out.dims and out["pressure_level"].ndim == 0:
        out = out.expand_dims("pressure_level")
    for long_name, candidates in _FIELD_SOURCE_NAMES.items():
        for candidate in candidates:
            if candidate in out.data_vars and candidate != long_name and long_name not in out.data_vars:
                out = out.rename({candidate: long_name})
                break
    return out


def _resolve_source_variable(ds: xr.Dataset, field_name: str) -> str:
    for candidate in _FIELD_SOURCE_NAMES.get(field_name, (field_name,)):
        if candidate in ds.data_vars:
            return candidate
    raise PipelineError(f"fixture does not contain requested variable {field_name}")


def _ensure_required_grid_dims(da: xr.DataArray, field_name: str) -> xr.DataArray:
    if "pressure_level" not in da.dims:
        for alias in _DIM_ALIASES["pressure_level"]:
            if alias != "pressure_level" and (alias in da.dims or alias in da.coords):
                da = da.rename({alias: "pressure_level"})
                break
    if "pressure_level" not in da.dims and "pressure_level" in da.coords and da["pressure_level"].ndim == 0:
        da = da.expand_dims("pressure_level")
    required = {"time", "pressure_level", "latitude", "longitude"}
    missing = required.difference(da.dims)
    if missing:
        raise PipelineError(f"source variable {field_name} missing required dimensions: {sorted(missing)}")
    unsupported = [d for d in da.dims if d not in required]
    if unsupported:
        raise PipelineError(f"source variable {field_name} has unsupported extra dimensions: {unsupported}")
    return da


def _filter_time(da: xr.DataArray, target_times: pd.DatetimeIndex) -> xr.DataArray:
    source_times = pd.DatetimeIndex(pd.to_datetime(da["time"].values)).tz_localize(None)
    target = pd.DatetimeIndex(pd.to_datetime(target_times)).tz_localize(None)
    missing = target.difference(source_times)
    if len(missing):
        raise PipelineError("fixture does not contain all requested timestamps")
    return da.sel(time=target.values)


def _filter_pressure_levels(da: xr.DataArray, requested: list[str]) -> xr.DataArray:
    coord = da["pressure_level"].values
    selected_native: list[Any] = []
    for req in requested:
        matches = [v for v in coord if _selector_equal(v, req)]
        if not matches:
            raise PipelineError("fixture does not contain requested pressure level")
        selected_native.append(matches[0])
    return da.sel(pressure_level=selected_native)


def _selector_equal(native: Any, requested: str) -> bool:
    try:
        return float(native) == float(requested)
    except Exception:
        return str(native) == requested


def _filter_area(da: xr.DataArray, area: list[Any]) -> xr.DataArray:
    north, west, south, east = [float(v) for v in area]
    lat = da["latitude"].values.astype(float)
    lon = da["longitude"].values.astype(float)
    lat_mask = (lat >= south) & (lat <= north)
    if not lat_mask.any():
        raise PipelineError("fixture latitude coordinate does not overlap requested area")

    if west == -180 and east == 180:
        lon_mask = np.ones(lon.shape, dtype=bool)
    else:
        lon_cmp = ((lon + 180.0) % 360.0) - 180.0
        west_cmp = ((west + 180.0) % 360.0) - 180.0
        east_cmp = ((east + 180.0) % 360.0) - 180.0
        if west_cmp <= east_cmp:
            lon_mask = (lon_cmp >= west_cmp) & (lon_cmp <= east_cmp)
        else:
            lon_mask = (lon_cmp >= west_cmp) | (lon_cmp <= east_cmp)
    if not lon_mask.any():
        raise PipelineError("fixture longitude coordinate does not overlap requested area")
    return da.isel(latitude=np.flatnonzero(lat_mask), longitude=np.flatnonzero(lon_mask))


def _clear_all_encodings(ds: xr.Dataset) -> None:
    for name in list(ds.variables):
        ds[name].encoding.clear()


def _artifact_channels(requests: list[ChannelRequest], selector_coord_paths: dict[str, str]) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for req in requests:
        selectors: dict[str, str] = {}
        paths: dict[str, str] = {}
        if req.selector_dimension is not None:
            selectors[req.selector_dimension] = str(req.selector_value)
            paths[req.selector_dimension] = selector_coord_paths[req.field_name]
        channels.append({
            "field_id": req.field_id,
            "array_path": req.field_name,
            "selectors": selectors,
            "selector_coordinate_paths": paths,
        })
    return channels


def _encoding_chunks(ds: xr.Dataset) -> dict[str, tuple[int, ...]]:
    chunks: dict[str, tuple[int, ...]] = {}
    for name, da in ds.data_vars.items():
        shape_by_dim = dict(zip(da.dims, da.shape, strict=True))
        if len(da.dims) == 4:
            level_dim = [d for d in da.dims if d.endswith("_pressure_level")][0]
            chunks[name] = (1, 1, int(shape_by_dim["latitude"]), int(shape_by_dim["longitude"]))
            if da.dims != ("time", level_dim, "latitude", "longitude"):
                raise PipelineError(f"unexpected dimension order for {name}")
        elif da.dims == ("time", "latitude", "longitude"):
            chunks[name] = (1, int(shape_by_dim["latitude"]), int(shape_by_dim["longitude"]))
        else:
            raise PipelineError(f"unexpected data variable dimensions for {name}: {da.dims}")
    return chunks


def _write_zarr_v3_with_preferred_codec(ds: xr.Dataset, tmp_store: Path, chunks: dict[str, tuple[int, ...]]) -> tuple[PublicationCodec, list[str]]:
    warnings: list[str] = []
    if tmp_store.exists():
        shutil.rmtree(tmp_store)
    uncompressed = PublicationCodec("uncompressed", ())
    try:
        _write_zarr(ds, tmp_store, chunks, uncompressed)
        _assert_codec_policy(tmp_store, ds.data_vars, "uncompressed")
        return uncompressed, warnings
    except Exception as exc:
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        codec = _lz4_codec()
        fallback = PublicationCodec(
            "blosc_lz4_clevel1_shuffle",
            (codec,),
            "uncompressed Zarr v3 chunks were unavailable in the host stack; used lossless Blosc-LZ4 clevel=1 byte-shuffle fallback",
        )
        _write_zarr(ds, tmp_store, chunks, fallback)
        _assert_codec_policy(tmp_store, ds.data_vars, fallback.name)
        warnings.append("preferred uncompressed Zarr v3 write path failed; lossless read-speed-biased fallback was used")
        return fallback, warnings


def _lz4_codec() -> Any:
    try:
        from zarr.codecs import BloscCodec
        try:
            return BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
        except TypeError:
            return BloscCodec(cname="lz4", clevel=1, shuffle=1)
    except Exception:
        from numcodecs import Blosc
        return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)


def _write_zarr(ds: xr.Dataset, store: Path, chunks: dict[str, tuple[int, ...]], codec: PublicationCodec) -> None:
    encoding: dict[str, dict[str, Any]] = {}
    for name in ds.data_vars:
        encoding[name] = {"chunks": chunks[name], "compressors": codec.encoding_value}
    for name, var in ds.coords.items():
        encoding[name] = {"chunks": tuple(max(1, int(s)) for s in var.shape), "compressors": ()}
    ds.to_zarr(store, mode="w", zarr_format=3, consolidated=True, encoding=encoding)


def _array_metadata(store: Path, array_path: str) -> dict[str, Any]:
    meta_path = store / array_path / "zarr.json"
    if not meta_path.exists():
        raise PipelineError(f"missing zarr metadata for {array_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_chunk_layout(store: Path, ds: xr.Dataset, chunks: dict[str, tuple[int, ...]]) -> None:
    for name, expected in chunks.items():
        meta = _array_metadata(store, name)
        got = tuple(meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", ()))
        if got != expected:
            raise PipelineError(f"unexpected chunk layout for {name}: {got} != {expected}")


def _assert_codec_policy(store: Path, data_vars: Iterable[str], expected: str) -> None:
    for name in data_vars:
        meta = _array_metadata(store, str(name))
        codec_names = [str(c.get("name", "")).lower() for c in meta.get("codecs", []) if isinstance(c, dict)]
        compressed = [c for c in codec_names if c not in {"bytes", "endian", "transpose", "crc32c", "sharding_indexed"}]
        if expected == "uncompressed":
            if compressed:
                raise PipelineError(f"data array {name} is unexpectedly compressed: {compressed}")
        else:
            joined = " ".join(codec_names) + " " + json.dumps(meta.get("codecs", []), sort_keys=True).lower()
            if "blosc" not in joined or "lz4" not in joined:
                raise PipelineError(f"data array {name} does not declare Blosc-LZ4 fallback codec")
            if "clevel" in joined and "1" not in joined:
                raise PipelineError(f"data array {name} fallback codec is not clevel=1")


def _assert_dataset_semantics_equal(expected: xr.Dataset, actual: xr.Dataset) -> None:
    if set(expected.data_vars) != set(actual.data_vars):
        raise PipelineError("reopened data variables differ from published dataset")
    if dict(expected.sizes) != dict(actual.sizes):
        raise PipelineError("reopened dimensions differ from published dataset")
    for coord in expected.coords:
        if coord not in actual.coords:
            raise PipelineError(f"missing coordinate {coord} after reopen")
        np.testing.assert_array_equal(expected[coord].values, actual[coord].values)
    for name in expected.data_vars:
        if expected[name].dims != actual[name].dims:
            raise PipelineError(f"dimension order changed for {name}")
        np.testing.assert_array_equal(expected[name].values, actual[name].values)
        for key, value in expected[name].attrs.items():
            if actual[name].attrs.get(key) != value:
                raise PipelineError(f"attribute {key} changed for {name}")


__all__ = ["run_pipeline"]
