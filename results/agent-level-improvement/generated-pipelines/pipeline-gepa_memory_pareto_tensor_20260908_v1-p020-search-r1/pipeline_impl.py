"""Standalone ERA5 pressure-level family adapter.

Implements pipeline_impl.run_pipeline(contract_lock, inventory, cache_dir, output_dir).
The adapter is deterministic, fixture-first, credential-free when a verified local
source_fixture_manifest.json is present, and never performs network probing.
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
import pandas as pd
import xarray as xr
import zarr

DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
PROVIDER = "ECMWF"
STORE_NAME = "era5_pressure_levels.zarr"
TMP_STORE_NAME = ".tmp_era5_pressure_levels.zarr"

SHORT_NAMES = {
    "temperature": ["temperature", "t"],
    "geopotential": ["geopotential", "z"],
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "relative_humidity": ["relative_humidity", "r"],
    "specific_humidity": ["specific_humidity", "q"],
    "vertical_velocity": ["vertical_velocity", "w"],
    "vorticity": ["vorticity", "vo"],
    "divergence": ["divergence", "d"],
    "ozone_mass_mixing_ratio": ["ozone_mass_mixing_ratio", "o3"],
    "potential_vorticity": ["potential_vorticity", "pv"],
    "fraction_of_cloud_cover": ["fraction_of_cloud_cover", "cc"],
    "specific_cloud_ice_water_content": ["specific_cloud_ice_water_content", "ciwc"],
    "specific_cloud_liquid_water_content": ["specific_cloud_liquid_water_content", "clwc"],
    "specific_rain_water_content": ["specific_rain_water_content", "crwc"],
    "specific_snow_water_content": ["specific_snow_water_content", "cswc"],
}


def run_pipeline(contract_lock: Any, inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract = _extract_contract(contract_lock)
    policy = _extract_policy(contract_lock)
    _validate_inventory_and_policy(contract, inventory, policy)

    cache_root = Path(cache_dir).resolve()
    manifest, verified_entries = _verify_fixture_manifest(cache_root)
    source_roots = _source_roots(cache_root, verified_entries)

    ds_src = _open_sources(source_roots)
    request = _validate_and_normalize_request(contract, inventory)
    ds_out, channels = _materialize_grouped_dataset(ds_src, request, inventory, manifest)

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / TMP_STORE_NAME
    final = out_root / STORE_NAME
    if tmp.exists():
        shutil.rmtree(tmp)

    _write_zarr_v3(ds_out, tmp, channels)
    _reopen_and_validate(tmp, ds_out, channels)

    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _reopen_and_validate(final, ds_out, channels)

    reused = [str(e["entry_id"]) for e in sorted(verified_entries, key=lambda x: str(x["entry_id"]))]
    return {
        "cache": {
            "hits": len(reused),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused,
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


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    found: list[dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("schema_version") == "dataset_contract.v1":
                found.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(lock)
    if len(found) != 1:
        raise ValueError("contract_lock must contain exactly one dataset_contract.v1 object")
    return found[0]


def _extract_policy(lock: Any) -> dict[str, Any] | None:
    if not isinstance(lock, dict):
        return None
    for key in ("fixed_pipeline_policy", "fixed_output_policy", "output_policy", "policy"):
        val = lock.get(key)
        if isinstance(val, dict) and (val.get("publication_format") == "zarr" or val.get("dataset_id") == DATASET_ID):
            return val
    return None


def _validate_inventory_and_policy(contract: dict[str, Any], inventory: dict[str, Any], policy: dict[str, Any] | None) -> None:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ValueError("unsupported inventory schema")
    if inventory.get("dataset_slug") != DATASET_SLUG or contract.get("dataset_slug") != DATASET_SLUG:
        raise ValueError("dataset identity mismatch")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ValueError("inventory dataset_id mismatch")
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id") != DATASET_ID:
        raise ValueError("contract advanced_options.dataset_id mismatch")
    if adv.get("data_format") != "grib" or adv.get("download_format") != "unarchived":
        raise ValueError("this policy requires GRIB acquisition and unarchived download format")
    if adv.get("product_type") != ["reanalysis"]:
        raise ValueError("only reanalysis product_type is valid for this adapter policy")
    if policy is not None:
        if policy.get("provider") != PROVIDER or policy.get("dataset_id") != DATASET_ID:
            raise ValueError("fixed output policy identity mismatch")
        if policy.get("acquisition_format") != "grib" or policy.get("publication_format") != "zarr":
            raise ValueError("fixed output policy format mismatch")
        zpol = policy.get("zarr") or {}
        if zpol.get("format_version") != 3 or zpol.get("consolidated_metadata") is not True:
            raise ValueError("fixed output policy requires consolidated Zarr v3")


def _verify_fixture_manifest(cache_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("no complete local source fixture; network acquisition is not permitted")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ValueError("invalid source fixture manifest JSON") from exc
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ValueError("unsupported source fixture manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source fixture manifest has no entries")
    verified: list[dict[str, Any]] = []
    for entry in entries:
        eid = str(entry.get("entry_id", ""))
        rel = entry.get("relative_path")
        sha = entry.get("sha256")
        size = entry.get("size_bytes")
        if not eid or not isinstance(rel, str) or not isinstance(sha, str) or not isinstance(size, int) or size <= 0:
            raise ValueError("malformed source fixture manifest entry")
        p = (cache_root / rel).resolve()
        try:
            p.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(f"fixture entry failed verification: {eid}") from exc
        if not p.is_file():
            raise ValueError(f"fixture entry failed verification: {eid}")
        if p.stat().st_size != size:
            raise ValueError(f"fixture entry failed verification: {eid}")
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != sha.lower():
            raise ValueError(f"fixture entry failed verification: {eid}")
        verified.append(entry)
    return manifest, verified


def _source_roots(cache_root: Path, entries: list[dict[str, Any]]) -> list[Path]:
    roots: set[Path] = set()
    for e in entries:
        p = (cache_root / e["relative_path"]).resolve()
        zroot = next((parent for parent in [p, *p.parents] if parent.name.endswith(".zarr") and parent.is_dir()), None)
        roots.add(zroot if zroot is not None else p)
    return sorted(roots, key=lambda p: p.as_posix())


def _open_sources(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    for p in paths:
        try:
            if p.is_dir() and p.name.endswith(".zarr"):
                datasets.append(xr.open_zarr(p, consolidated=False).load())
                continue
            suffix = p.suffix.lower()
            if suffix in {".nc", ".nc4", ".cdf", ".netcdf"}:
                datasets.append(xr.open_dataset(p, decode_cf=True, mask_and_scale=True).load())
                continue
            if suffix in {".grib", ".grb", ".grib2", ".grb2"} or suffix == "":
                try:
                    import cfgrib

                    opened = cfgrib.open_datasets(str(p), backend_kwargs={"indexpath": ""})
                    datasets.extend([d.load() for d in opened])
                except Exception:
                    datasets.append(xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}).load())
                continue
            datasets.append(xr.open_dataset(p, decode_cf=True, mask_and_scale=True).load())
        except Exception as exc:  # keep diagnostics secret/path safe
            errors.append(type(exc).__name__)
    if not datasets:
        raise RuntimeError("verified local fixture could not be decoded: " + ",".join(sorted(set(errors))))
    return xr.merge(datasets, compat="override", join="outer")


def _validate_and_normalize_request(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    opts = inventory.get("options") or {}
    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("contract must request at least one field")
    normalized_fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in fields:
        name = f.get("name")
        if name not in opts.get("variable", []):
            raise ValueError("requested variable is not in frozen inventory")
        selectors = f.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ValueError("each pressure-level field must have exactly one pressure_level selector")
        level = str(selectors[0].get("value"))
        if level not in opts.get("pressure_level", []):
            raise ValueError("requested pressure_level is not in frozen inventory")
        if selectors[0].get("unit") not in (None, "hPa"):
            raise ValueError("pressure_level selector unit must be hPa")
        fid = _field_id(name, selectors)
        if fid in seen:
            raise ValueError("duplicate requested field-selector channel")
        seen.add(fid)
        normalized_fields.append({"name": name, "level": level, "selectors": selectors, "field_id": fid})

    scope = contract.get("scope") or {}
    if scope.get("product_type") != "reanalysis":
        raise ValueError("scope.product_type must be reanalysis")
    geo = scope.get("geography") or {}
    if geo.get("cds_area_order") != ["north", "west", "south", "east"]:
        raise ValueError("unsupported geography coordinate order")
    area = geo.get("cds_area")
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise ValueError("invalid geography area")
    if not (90 >= area[0] >= area[2] >= -90 and -360 <= area[1] <= 360 and -360 <= area[3] <= 360):
        raise ValueError("invalid geography bounds")

    tscope = scope.get("time") or {}
    selected_times = tscope.get("selected_times")
    if not isinstance(selected_times, list) or not selected_times:
        raise ValueError("time.selected_times is required")
    for t in selected_times:
        if t not in opts.get("time", []):
            raise ValueError("selected time is not in frozen inventory")
    if tscope.get("timezone") != "UTC":
        raise ValueError("only UTC time selections are supported")
    times = _selected_datetimes(scope.get("date_range") or {}, selected_times, opts)
    return {"fields": normalized_fields, "times": times, "area": area}


def _selected_datetimes(date_range: dict[str, Any], selected_times: list[str], opts: dict[str, Any]) -> pd.DatetimeIndex:
    start_s = date_range.get("start_date")
    end_s = date_range.get("end_date")
    inclusive = date_range.get("inclusive", True)
    if not isinstance(start_s, str) or not isinstance(end_s, str):
        raise ValueError("date_range start_date and end_date are required")
    start_d = pd.Timestamp(start_s).date()
    end_d = pd.Timestamp(end_s).date()
    if end_d < start_d:
        raise ValueError("date_range end precedes start")
    out: list[pd.Timestamp] = []
    d = start_d
    last = end_d if inclusive else end_d - timedelta(days=1)
    while d <= last:
        ys, ms, ds = f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}"
        if ys not in opts.get("year", []) or ms not in opts.get("month", []) or ds not in opts.get("day", []):
            raise ValueError("requested date is outside frozen inventory options")
        for hhmm in selected_times:
            hh, mm = [int(x) for x in hhmm.split(":")]
            out.append(pd.Timestamp(datetime.combine(d, time(hh, mm), tzinfo=timezone.utc)).tz_convert(None))
        d += timedelta(days=1)
    return pd.DatetimeIndex(out, name="time")


def _materialize_grouped_dataset(ds: xr.Dataset, request: dict[str, Any], inventory: dict[str, Any], manifest: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    time_name = _find_coord(ds, ["time", "valid_time"])
    lat_name = _find_coord(ds, ["latitude", "lat"])
    lon_name = _find_coord(ds, ["longitude", "lon"])
    level_name = _find_coord(ds, ["pressure_level", "isobaricInhPa", "level", "plev"])
    if not all([time_name, lat_name, lon_name]):
        raise RuntimeError("source fixture lacks required time/latitude/longitude coordinates")

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for f in request["fields"]:
        groups.setdefault(f["name"], []).append(f)

    out_vars: dict[str, xr.DataArray] = {}
    out_coords: dict[str, Any] = {}
    channels: list[dict[str, Any]] = []

    for var_name, reqs in groups.items():
        src_var = _find_data_var(ds, var_name)
        if src_var is None:
            raise RuntimeError("source fixture lacks a requested variable")
        da = ds[src_var]
        rename: dict[str, str] = {}
        for old, new in ((time_name, "time"), (lat_name, "latitude"), (lon_name, "longitude")):
            if old in da.dims or old in da.coords:
                rename[old] = new
        if rename:
            da = da.rename(rename)
        lvl_coord = level_name
        if lvl_coord in rename:
            lvl_coord = rename[lvl_coord]
        levels = [r["level"] for r in reqs]
        selector_dim = f"{_safe_name(var_name)}_pressure_level"
        if lvl_coord and (lvl_coord in da.dims or lvl_coord in da.coords):
            da = _select_levels(da, lvl_coord, levels)
            if lvl_coord != selector_dim:
                da = da.rename({lvl_coord: selector_dim})
        else:
            if len(levels) != 1:
                raise RuntimeError("source fixture lacks pressure-level coordinate for requested multi-level field")
            da = da.expand_dims({selector_dim: np.array([int(levels[0])], dtype=np.int32)})
        da = _select_times_and_area(da, request["times"], request["area"])
        da = da.transpose("time", selector_dim, "latitude", "longitude")
        da = da.astype(np.float32, copy=False) if np.issubdtype(da.dtype, np.floating) else da
        da.attrs = dict(ds[src_var].attrs)
        da.attrs.setdefault("long_name", (inventory.get("option_metadata", {}).get("variable", {}).get(var_name, {}) or {}).get("label", var_name))
        units = (inventory.get("option_metadata", {}).get("variable", {}).get(var_name, {}) or {}).get("units")
        if units and "units" not in da.attrs:
            da.attrs["units"] = _strip_html(units)
        da.encoding.clear()
        out_vars[var_name] = da
        out_coords[selector_dim] = da[selector_dim]
        out_coords[selector_dim].attrs.update({"units": "hPa", "positive": "down", "selector_dimension": "pressure_level"})
        for r in reqs:
            channels.append({
                "field_id": r["field_id"],
                "array_path": var_name,
                "selectors": {"pressure_level": r["level"]},
                "selector_coordinate_paths": {"pressure_level": selector_dim},
            })

    ds_out = xr.Dataset(out_vars)
    ds_out.attrs = {
        "dataset_id": DATASET_ID,
        "dataset_slug": DATASET_SLUG,
        "provider": PROVIDER,
        "publication_format": "zarr_v3_consolidated",
        "source_fixture_schema_version": manifest.get("schema_version", "source_fixture_manifest.v1"),
        "semantic_layout": "grouped_by_native_variable_with_per_variable_pressure_level_selector_dimension",
    }
    for c in ("time", "latitude", "longitude"):
        if c in ds_out.coords:
            ds_out[c].encoding.clear()
    ds_out["time"].attrs.setdefault("standard_name", "time")
    ds_out["latitude"].attrs.setdefault("standard_name", "latitude")
    ds_out["latitude"].attrs.setdefault("units", "degrees_north")
    ds_out["longitude"].attrs.setdefault("standard_name", "longitude")
    ds_out["longitude"].attrs.setdefault("units", "degrees_east")
    return ds_out, channels


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    names = list(ds.coords) + list(ds.dims)
    lower = {str(n).lower(): str(n) for n in names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for n in names:
        ln = str(n).lower()
        if any(c.lower() in ln for c in candidates):
            return str(n)
    return None


def _find_data_var(ds: xr.Dataset, requested: str) -> str | None:
    candidates = [c.lower() for c in SHORT_NAMES.get(requested, [requested])]
    for name, da in ds.data_vars.items():
        vals = [str(name).lower()]
        vals.extend(str(da.attrs.get(k, "")).lower() for k in ("GRIB_shortName", "GRIB_cfVarName", "shortName", "standard_name", "long_name"))
        if any(v in candidates for v in vals):
            return str(name)
    for name in ds.data_vars:
        if str(name).lower() == requested.lower():
            return str(name)
    return None


def _select_levels(da: xr.DataArray, level_name: str, levels: list[str]) -> xr.DataArray:
    coord = da[level_name]
    mapping = {_level_key(v): v for v in coord.values.tolist()}
    missing = [lv for lv in levels if lv not in mapping]
    if missing:
        raise RuntimeError("source fixture lacks a requested pressure level")
    selected_native = [mapping[lv] for lv in levels]
    return da.sel({level_name: selected_native})


def _level_key(v: Any) -> str:
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(v)


def _select_times_and_area(da: xr.DataArray, times: pd.DatetimeIndex, area: list[float]) -> xr.DataArray:
    src_times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    lookup = {pd.Timestamp(t).to_datetime64(): t for t in src_times}
    wanted = [t.to_datetime64() for t in times]
    if any(t not in lookup for t in wanted):
        raise RuntimeError("source fixture lacks one or more selected timestamps")
    da = da.sel(time=list(times.to_numpy()))
    north, west, south, east = area
    lat = da["latitude"]
    if lat.size > 0:
        if float(lat[0]) >= float(lat[-1]):
            da = da.sel(latitude=slice(north, south))
        else:
            da = da.sel(latitude=slice(south, north))
    lon = da["longitude"]
    if not (west <= -180 and east >= 180):
        vals = lon.values
        if np.nanmin(vals) >= 0 and west < 0:
            west2 = west % 360
            east2 = east % 360
            mask = (lon >= west2) & (lon <= east2) if west2 <= east2 else ((lon >= west2) | (lon <= east2))
            da = da.sel(longitude=lon.where(mask, drop=True))
        else:
            da = da.sel(longitude=lon.where((lon >= west) & (lon <= east), drop=True))
    return da


def _write_zarr_v3(ds: xr.Dataset, store: Path, channels: list[dict[str, Any]]) -> None:
    codec = _blosc_codec()
    encoding: dict[str, dict[str, Any]] = {}
    for name, da in ds.data_vars.items():
        selector_dim = [d for d in da.dims if d.endswith("_pressure_level")][0]
        encoding[name] = {
            "chunks": (1, 1, int(da.sizes["latitude"]), int(da.sizes["longitude"])),
            "compressors": [codec],
        }
    for cname, coord in ds.coords.items():
        encoding[cname] = {"chunks": tuple(int(coord.sizes[d]) for d in coord.dims)} if coord.dims else {}
    for v in ds.variables:
        ds[v].encoding.clear()
    try:
        ds.to_zarr(store, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    except TypeError:
        ds.to_zarr(store, mode="w", zarr_version=3, consolidated=True, encoding=encoding)
    zarr.consolidate_metadata(str(store))


def _blosc_codec() -> Any:
    from zarr.codecs import BloscCodec
    try:
        from zarr.codecs import BloscShuffle
        return BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle)
    except Exception:
        return BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")


def _reopen_and_validate(store: Path, expected: xr.Dataset, channels: list[dict[str, Any]]) -> None:
    reopened = xr.open_zarr(store, consolidated=True).load()
    if set(reopened.data_vars) != set(expected.data_vars):
        raise RuntimeError("published Zarr data variable set mismatch")
    for name in expected.data_vars:
        xr.testing.assert_identical(reopened[name], expected[name])
        da = expected[name]
        selector_dim = [d for d in da.dims if d.endswith("_pressure_level")][0]
        arr = zarr.open(str(store / name), mode="r")
        want_chunks = (1, 1, int(da.sizes["latitude"]), int(da.sizes["longitude"]))
        if tuple(arr.chunks) != want_chunks:
            raise RuntimeError("published Zarr chunk topology mismatch")
        if not _has_required_blosc(arr):
            raise RuntimeError("published Zarr codec is not Blosc-Zstd clevel=9 bitshuffle")
        if selector_dim not in reopened.dims or reopened.sizes[selector_dim] < 1:
            raise RuntimeError("selector dimension was not preserved as a real dimension")
    root_meta = json.loads((store / "zarr.json").read_text())
    if "consolidated_metadata" not in root_meta:
        raise RuntimeError("published Zarr v3 metadata is not consolidated")
    seen = set()
    for ch in channels:
        key = (ch["field_id"], ch["array_path"], json.dumps(ch["selectors"], sort_keys=True))
        if key in seen:
            raise RuntimeError("duplicate artifact channel declaration")
        seen.add(key)
        if ch["array_path"] not in expected.data_vars:
            raise RuntimeError("artifact channel points to missing array")
        for coord_path in ch["selector_coordinate_paths"].values():
            if coord_path not in expected.coords:
                raise RuntimeError("artifact selector coordinate path is missing")


def _has_required_blosc(arr: Any) -> bool:
    codecs = getattr(getattr(arr, "metadata", None), "codecs", []) or []
    text = " ".join([repr(c).lower() for c in codecs])
    return "blosc" in text and "zstd" in text and "clevel=9" in text and "bitshuffle" in text


def _field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    inner = ",".join(f'{s["dimension"]}={json.dumps(str(s["value"]))}' for s in selectors)
    return f"{name}[{inner}]"


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).replace("  ", " ").strip()
