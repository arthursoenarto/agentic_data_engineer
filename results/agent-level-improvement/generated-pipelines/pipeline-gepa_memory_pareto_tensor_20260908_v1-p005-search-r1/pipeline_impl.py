"""Reusable ERA5 pressure-level regular-grid adapter.

Entry point required by the framework:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The implementation is deterministic and fixture-first.  If cache_dir contains
source_fixture_manifest.json, every listed raw file is verified and consumed
before any provider/client/credential logic.  This adapter intentionally does
not implement network acquisition; incomplete local fixtures fail closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p005-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "dataset.zarr"
MANIFEST_NAME = "source_fixture_manifest.json"

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "relative_humidity": ("relative_humidity", "r"),
    "specific_humidity": ("specific_humidity", "q"),
    "vertical_velocity": ("vertical_velocity", "w"),
    "vorticity": ("vorticity", "vo"),
    "divergence": ("divergence", "d"),
    "ozone_mass_mixing_ratio": ("ozone_mass_mixing_ratio", "o3"),
    "potential_vorticity": ("potential_vorticity", "pv"),
    "fraction_of_cloud_cover": ("fraction_of_cloud_cover", "cc"),
    "specific_cloud_ice_water_content": ("specific_cloud_ice_water_content", "ciwc"),
    "specific_cloud_liquid_water_content": ("specific_cloud_liquid_water_content", "clwc"),
    "specific_rain_water_content": ("specific_rain_water_content", "crwc"),
    "specific_snow_water_content": ("specific_snow_water_content", "cswc"),
}

CANONICAL_RENAMES: dict[str, tuple[str, ...]] = {
    "time": ("time", "valid_time"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "level", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}


def run_pipeline(contract_lock: Any, inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Materialize a validated ERA5 pressure-level contract to consolidated Zarr v3.

    Parameters are supplied by the framework.  Paths returned in the artifact are
    relative to output_dir and contain no secrets.
    """
    contract = _extract_contract(contract_lock)
    _validate_contract(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_files, reused_keys = _verify_and_resolve_fixture(cache_root)
    if not source_files:
        raise RuntimeError("No complete local source fixture was found; network acquisition is intentionally disabled for this adapter.")

    raw = _open_source_dataset(source_files)
    canonical = _canonicalize_dataset(raw)
    expected = _filter_dataset(canonical, contract, inventory)
    expected = _strip_source_encodings(expected)

    store_path = _publish_zarr_atomic(expected, output_root)
    chunk_map = _expected_chunk_map(expected)
    codec_summary = _reopen_and_validate_zarr(store_path, expected, chunk_map)

    channels = _build_channels(contract, expected)
    return {
        "cache": {
            "hits": len(reused_keys),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused_keys,
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
        "warnings": [f"Published Zarr v3 with throughput-oriented lossless codec(s): {codec_summary}"],
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, (str, os.PathLike)):
        with open(lock, "r", encoding="utf-8") as f:
            lock = json.load(f)
    if not isinstance(lock, dict):
        raise ValueError("contract_lock must be a mapping or JSON file path")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "resolved_contract"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
            return value
    nested = lock.get("lock")
    if isinstance(nested, dict):
        try:
            return _extract_contract(nested)
        except ValueError:
            pass
    contracts = lock.get("contracts")
    if isinstance(contracts, list):
        selected_id = lock.get("selected_contract_id") or lock.get("contract_id")
        if selected_id:
            for c in contracts:
                if isinstance(c, dict) and c.get("id") == selected_id and c.get("schema_version") == "dataset_contract.v1":
                    return c
        dataset_contracts = [c for c in contracts if isinstance(c, dict) and c.get("schema_version") == "dataset_contract.v1"]
        if len(dataset_contracts) == 1:
            return dataset_contracts[0]
        for c in dataset_contracts:
            if c.get("human_confirmed") is True:
                return c
    raise ValueError("Could not find dataset_contract.v1 inside contract lock envelope")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if contract.get("schema_version") != "dataset_contract.v1":
        raise ValueError("Unsupported contract schema_version")
    if contract.get("dataset_slug") != inventory.get("dataset_slug") or contract.get("dataset_slug") != DATASET_SLUG:
        raise ValueError("Contract dataset_slug does not match frozen inventory")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ValueError("Inventory dataset_id does not match this adapter")

    options = inventory.get("options", {})
    fields = contract.get("fields") or []
    if not fields:
        raise ValueError("Contract must request at least one field")
    for field in fields:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise ValueError(f"Unsupported variable requested: {name!r}")
        selectors = field.get("selectors") or []
        for sel in selectors:
            dim = sel.get("dimension")
            val = str(sel.get("value"))
            if dim != "pressure_level":
                raise ValueError(f"Unsupported selector dimension for ERA5 pressure-level grid: {dim!r}")
            if val not in options.get("pressure_level", []):
                raise ValueError(f"Unsupported pressure_level requested: {val!r}")
            unit = sel.get("unit")
            if unit not in (None, "hPa"):
                raise ValueError("pressure_level selectors must use hPa when a unit is supplied")

    scope = contract.get("scope") or {}
    product_type = scope.get("product_type") or (contract.get("advanced_options") or {}).get("product_type")
    if isinstance(product_type, list):
        product_values = product_type
    else:
        product_values = [product_type]
    for p in product_values:
        if p and p not in options.get("product_type", []):
            raise ValueError(f"Unsupported product_type requested: {p!r}")
    if product_values and any(p != "reanalysis" for p in product_values if p):
        raise ValueError("This regular-grid adapter supports the reanalysis product_type only")

    selected_times = ((scope.get("time") or {}).get("selected_times") or [])
    if not selected_times:
        raise ValueError("scope.time.selected_times is required")
    for t in selected_times:
        if t not in options.get("time", []):
            raise ValueError(f"Unsupported selected time: {t!r}")

    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise ValueError("date_range.end_date must be on or after start_date")
    available_years = set(options.get("year", []))
    d = start
    while d <= end:
        if f"{d.year:04d}" not in available_years or f"{d.month:02d}" not in options.get("month", []) or f"{d.day:02d}" not in options.get("day", []):
            raise ValueError(f"Requested date is outside inventory options: {d.isoformat()}")
        d += timedelta(days=1)

    area = ((scope.get("geography") or {}).get("cds_area") or (scope.get("geography") or {}).get("area"))
    if area != "global":
        if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
            raise ValueError("geography.cds_area must be [north, west, south, east] or geography.area must be global")
        north, west, south, east = [float(x) for x in area]
        if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
            raise ValueError("Requested geographic area is invalid")

    adv = contract.get("advanced_options") or {}
    dataset_id = adv.get("dataset_id", DATASET_ID)
    if dataset_id != DATASET_ID:
        raise ValueError("advanced_options.dataset_id does not match inventory")
    if adv.get("data_format", "grib") not in inventory.get("options", {}).get("data_format", []):
        raise ValueError("Unsupported data_format")
    if adv.get("data_format", "grib") != "grib":
        raise ValueError("Fixed acquisition policy for this adapter requires data_format='grib'")
    if adv.get("download_format", "unarchived") not in inventory.get("options", {}).get("download_format", []):
        raise ValueError("Unsupported download_format")


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"date_range.{name} must be an ISO date string")
    return datetime.strptime(value, "%Y-%m-%d").date()


def _verify_and_resolve_fixture(cache_root: Path) -> tuple[list[Path], list[str]]:
    manifest_path = cache_root / MANIFEST_NAME
    if not manifest_path.exists():
        return [], []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ValueError("Unsupported source fixture manifest schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source fixture manifest must contain at least one entry")
    files: list[Path] = []
    keys: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise ValueError("Fixture relative_path must be a relative path")
        path = (cache_root / rel).resolve()
        if cache_root not in path.parents and path != cache_root:
            raise ValueError("Fixture path escapes cache_dir")
        if not path.is_file():
            raise FileNotFoundError(f"Fixture file is missing: {rel}")
        expected_size = int(entry.get("size_bytes"))
        if expected_size <= 0:
            raise ValueError("Fixture size_bytes must be positive")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Fixture size mismatch for {rel}")
        expected_sha = entry.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != expected_sha:
            raise ValueError("Fixture sha256 must be a lowercase SHA-256 hex digest")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Fixture checksum mismatch for {rel}")
        entry_id = entry.get("entry_id") or rel
        safe_id = "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in str(entry_id))[:160]
        keys.append(f"fixture:{safe_id}:sha256:{actual_sha[:16]}")
        files.append(path)
    return files, keys


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _open_source_dataset(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in sorted(paths, key=lambda p: str(p)):
        suffixes = "".join(path.suffixes).lower()
        errors: list[str] = []
        engine_plan: list[tuple[str | None, dict[str, Any]]] = []
        if suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
            engine_plan.append(("cfgrib", {"backend_kwargs": {"indexpath": ""}}))
        engine_plan.extend([
            ("h5netcdf", {}),
            ("netcdf4", {}),
            ("scipy", {}),
            (None, {}),
        ])
        ds = None
        for engine, kwargs in engine_plan:
            try:
                open_kwargs = dict(decode_cf=True, mask_and_scale=True)
                open_kwargs.update(kwargs)
                if engine is not None:
                    open_kwargs["engine"] = engine
                ds = xr.open_dataset(path, **open_kwargs)
                break
            except Exception as exc:  # try next engine; final error is sanitized below
                errors.append(f"{engine or 'auto'}:{type(exc).__name__}")
        if ds is None:
            raise RuntimeError(f"Could not decode fixture file {path.name}; tried engines {', '.join(errors)}")
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, data_vars="minimal", coords="minimal", compat="override", combine_attrs="override")
    except Exception:
        return xr.merge(datasets, compat="override", combine_attrs="override")


def _canonicalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    renames: dict[str, str] = {}
    names = set(ds.dims) | set(ds.coords) | set(ds.variables)
    for canonical, candidates in CANONICAL_RENAMES.items():
        if canonical in names:
            continue
        for candidate in candidates:
            if candidate in names:
                renames[candidate] = canonical
                break
    if renames:
        ds = ds.rename(renames)
    required = {"time", "latitude", "longitude"}
    missing = sorted(required - set(ds.coords) - set(ds.dims))
    if missing:
        raise ValueError(f"Decoded source is missing required coordinate(s): {missing}")
    return ds


def _filter_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any]) -> xr.Dataset:
    times = _requested_timestamps(contract)
    ds = _select_time(ds, times)
    ds = _select_area(ds, contract)

    fields = contract.get("fields") or []
    by_var: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for field in fields:
        name = field["name"]
        entry = by_var.setdefault(name, {"levels": [], "fields": []})
        selectors = field.get("selectors") or []
        level = None
        for sel in selectors:
            if sel.get("dimension") == "pressure_level":
                level = str(sel.get("value"))
        if level is not None and level not in entry["levels"]:
            entry["levels"].append(level)
        entry["fields"].append(field)

    data_vars: OrderedDict[str, xr.DataArray] = OrderedDict()
    for requested_name, spec in by_var.items():
        source_name = _find_source_variable(ds, requested_name)
        da = ds[source_name]
        if source_name != requested_name:
            da = da.rename(requested_name)
        levels = spec["levels"]
        if levels:
            if "pressure_level" not in da.dims and "pressure_level" not in da.coords:
                if len(levels) == 1 and "pressure_level" in ds.coords:
                    coerced_levels = _coerce_coord_values(ds["pressure_level"].values, levels)
                    da = da.expand_dims(pressure_level=coerced_levels)
                else:
                    raise ValueError(f"Requested pressure_level selector for variable {requested_name}, but decoded source lacks pressure_level")
            coerced_levels = _coerce_coord_values(ds["pressure_level"].values, levels)
            da = da.sel(pressure_level=coerced_levels)
            # xarray preserves a dimension for list selection, including cardinality one.
        dims = [d for d in ("time", "pressure_level", "latitude", "longitude") if d in da.dims]
        extra_dims = [d for d in da.dims if d not in dims]
        if extra_dims:
            raise ValueError(f"Variable {requested_name} contains unsupported extra dimension(s): {extra_dims}")
        da = da.transpose(*dims)
        da.attrs = dict(da.attrs)
        meta = ((inventory.get("option_metadata") or {}).get("variable") or {}).get(requested_name, {})
        if "units" not in da.attrs and meta.get("units"):
            da.attrs["units"] = meta["units"]
        if "long_name" not in da.attrs and meta.get("label"):
            da.attrs["long_name"] = meta["label"]
        data_vars[requested_name] = da

    coords: dict[str, Any] = {}
    for coord in ("time", "pressure_level", "latitude", "longitude"):
        if coord in ds.coords:
            coords[coord] = ds[coord]
    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(ds.attrs))
    out.attrs.update({
        "dataset_slug": contract.get("dataset_slug"),
        "dataset_id": DATASET_ID,
        "source_provider": "ECMWF",
        "publication_format": "zarr_v3_consolidated",
    })
    return out


def _requested_timestamps(contract: dict[str, Any]) -> pd.DatetimeIndex:
    scope = contract["scope"]
    dr = scope["date_range"]
    start = _parse_date(dr["start_date"], "start_date")
    end = _parse_date(dr["end_date"], "end_date")
    selected_times = (scope.get("time") or {}).get("selected_times") or []
    inclusive = bool(dr.get("inclusive", True))
    final = end if inclusive else end - timedelta(days=1)
    if final < start:
        return pd.DatetimeIndex([], tz="UTC")
    values: list[pd.Timestamp] = []
    d = start
    while d <= final:
        for t_str in selected_times:
            hh, mm = [int(x) for x in t_str.split(":")]
            values.append(pd.Timestamp(datetime.combine(d, time(hh, mm), tzinfo=timezone.utc)))
        d += timedelta(days=1)
    return pd.DatetimeIndex(values).tz_convert(None)


def _select_time(ds: xr.Dataset, times: pd.DatetimeIndex) -> xr.Dataset:
    if len(times) == 0:
        raise ValueError("Requested time selection is empty")
    if "time" not in ds.coords:
        raise ValueError("Decoded source lacks time coordinate")
    target = np.asarray(times.values, dtype="datetime64[ns]")
    source = np.asarray(pd.to_datetime(ds["time"].values).values, dtype="datetime64[ns]")
    missing = [str(pd.Timestamp(t)) for t in target if t not in set(source)]
    if missing:
        raise ValueError(f"Source fixture does not contain requested timestamp(s), first missing: {missing[:3]}")
    return ds.assign_coords(time=source).sel(time=target)


def _select_area(ds: xr.Dataset, contract: dict[str, Any]) -> xr.Dataset:
    geography = (contract.get("scope") or {}).get("geography") or {}
    area = geography.get("cds_area")
    if area is None and geography.get("area") == "global":
        return ds
    if area is None:
        return ds
    north, west, south, east = [float(x) for x in area]
    if north == 90 and south == -90 and ((west == -180 and east == 180) or (west == 0 and east == 360)):
        return ds
    lat = ds["latitude"].values
    if lat[0] > lat[-1]:
        ds = ds.sel(latitude=slice(north, south))
    else:
        ds = ds.sel(latitude=slice(south, north))
    lon = ds["longitude"].values
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    req_west, req_east = west, east
    if lon_min >= 0 and req_west < 0:
        req_west = req_west % 360
        req_east = req_east % 360
    if req_west > req_east:
        raise ValueError("Longitude selections crossing the dateline are not materialized by this deterministic adapter")
    return ds.sel(longitude=slice(req_west, req_east))


def _find_source_variable(ds: xr.Dataset, requested_name: str) -> str:
    for candidate in FIELD_ALIASES.get(requested_name, (requested_name,)):
        if candidate in ds.data_vars:
            return candidate
    raise ValueError(f"Decoded source fixture lacks requested variable {requested_name!r}")


def _coerce_coord_values(coord_values: np.ndarray, string_values: list[str]) -> list[Any]:
    dtype = np.asarray(coord_values).dtype
    out: list[Any] = []
    for v in string_values:
        if np.issubdtype(dtype, np.integer):
            out.append(int(v))
        elif np.issubdtype(dtype, np.floating):
            out.append(float(v))
        else:
            out.append(v)
    existing = set(np.asarray(coord_values).tolist())
    missing = [v for v in out if v not in existing]
    if missing:
        raise ValueError(f"Source fixture does not contain requested pressure_level(s): {missing}")
    return out


def _strip_source_encodings(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    forbidden = {"scale_factor", "add_offset", "_FillValue", "missing_value", "dtype"}
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {k: v for k, v in ds[name].encoding.items() if k not in forbidden}
    return ds


def _expected_chunk_map(ds: xr.Dataset) -> dict[str, tuple[int, ...]]:
    lat_n = int(ds.sizes["latitude"])
    lon_n = int(ds.sizes["longitude"])
    mapping: dict[str, tuple[int, ...]] = {}
    for name, da in ds.data_vars.items():
        chunks: list[int] = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim == "pressure_level":
                chunks.append(1)
            elif dim == "latitude":
                chunks.append(lat_n)
            elif dim == "longitude":
                chunks.append(lon_n)
            else:
                chunks.append(int(ds.sizes[dim]))
        mapping[name] = tuple(chunks)
    return mapping


def _publish_zarr_atomic(ds: xr.Dataset, output_root: Path) -> Path:
    final = output_root / STORE_NAME
    tmp = output_root / f".{STORE_NAME}.tmp"
    old = output_root / f".{STORE_NAME}.old"
    for p in (tmp, old):
        if p.exists():
            shutil.rmtree(p)
    encoding = _build_encoding(ds)
    write_errors: list[str] = []
    for style in ("v3", "legacy"):
        trial_encoding = _with_codec_style(encoding, style)
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            try:
                ds.to_zarr(tmp, mode="w", consolidated=True, zarr_format=3, encoding=trial_encoding)
            except TypeError:
                ds.to_zarr(tmp, mode="w", consolidated=True, zarr_version=3, encoding=trial_encoding)
            _ensure_consolidated(tmp)
            _assert_data_codecs_present(tmp, ds.data_vars.keys())
            break
        except Exception as exc:
            write_errors.append(f"{style}:{type(exc).__name__}:{exc}")
            if tmp.exists():
                shutil.rmtree(tmp)
    else:
        raise RuntimeError("Could not write compressed Zarr v3 store: " + " | ".join(write_errors))

    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)
    return final


def _build_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    chunks = _expected_chunk_map(ds)
    encoding: dict[str, dict[str, Any]] = {}
    for name, ch in chunks.items():
        encoding[name] = {"chunks": ch}
    for coord in ds.coords:
        if coord in ds.dims:
            encoding.setdefault(coord, {})["chunks"] = (int(ds.sizes[coord]),)
    return encoding


def _with_codec_style(base: dict[str, dict[str, Any]], style: str) -> dict[str, dict[str, Any]]:
    enc = {k: dict(v) for k, v in base.items()}
    if style == "v3":
        codec = _zarr_v3_blosc_codec()
        key = "compressors"
        value = [codec]
    else:
        codec = _zarr_v3_blosc_codec()
        key = "compressor"
        value = codec
    for name in list(enc.keys()):
        # Only data variables receive throughput-critical compression here; coordinate arrays remain compact chunks.
        if name not in ("time", "pressure_level", "latitude", "longitude"):
            enc[name][key] = value
    return enc


def _zarr_v3_blosc_codec() -> Any:
    try:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
    except Exception:
        return _numcodecs_blosc_codec()


def _numcodecs_blosc_codec() -> Any:
    from numcodecs import Blosc
    return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)


def _ensure_consolidated(store: Path) -> None:
    try:
        import zarr
        zarr.consolidate_metadata(str(store))
    except Exception:
        pass
    root_meta = store / "zarr.json"
    if not root_meta.exists():
        raise RuntimeError("Zarr v3 root zarr.json is missing")
    with open(root_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("zarr_format", 0)) != 3:
        raise RuntimeError("Published store is not Zarr format 3")
    if "consolidated_metadata" not in meta:
        raise RuntimeError("Published Zarr v3 store lacks consolidated metadata")


def _zarr_codec_text(arr: Any) -> str:
    parts: list[str] = []
    for attr in ("metadata", "compressors", "compressor"):
        try:
            parts.append(str(getattr(arr, attr, "")))
        except TypeError:
            # Zarr v3 arrays intentionally do not expose the legacy .compressor property.
            continue
    return "".join(parts)


def _assert_data_codecs_present(store: Path, data_var_names: Any) -> None:
    import zarr
    group = zarr.open_group(str(store), mode="r")
    for name in data_var_names:
        arr = group[name]
        codec_text = _zarr_codec_text(arr)
        lower = codec_text.lower()
        if "blosc" not in lower:
            raise RuntimeError(f"Data array {name} is missing explicit Blosc compression")
        if "lz4" not in lower and "zstd" not in lower:
            raise RuntimeError(f"Data array {name} codec is not a fast lossless Blosc codec")


def _reopen_and_validate_zarr(store: Path, expected: xr.Dataset, chunks: dict[str, tuple[int, ...]]) -> str:
    _ensure_consolidated(store)
    _assert_data_codecs_present(store, expected.data_vars.keys())
    import zarr
    group = zarr.open_group(str(store), mode="r")
    summaries: list[str] = []
    for name, expected_chunks in chunks.items():
        arr = group[name]
        actual_chunks = tuple(int(x) for x in arr.chunks)
        if actual_chunks != expected_chunks:
            raise RuntimeError(f"Unexpected chunk layout for {name}: {actual_chunks} != {expected_chunks}")
        codec_text = _zarr_codec_text(arr)
        summaries.append(f"{name}:chunks={actual_chunks}:codec={'lz4' if 'lz4' in codec_text.lower() else 'blosc'}")
    try:
        reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    except TypeError:
        reopened = xr.open_zarr(store, consolidated=True, zarr_version=3)
    reopened = reopened[expected.data_vars.keys()]
    expected_cmp = expected[expected.data_vars.keys()]
    xr.testing.assert_identical(reopened.load(), expected_cmp.load())
    for coord in ("time", "pressure_level", "latitude", "longitude"):
        if coord in expected.coords:
            if coord not in reopened.coords:
                raise RuntimeError(f"Reopened store is missing coordinate {coord}")
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
    return "; ".join(summaries)


def _build_channels(contract: dict[str, Any], ds: xr.Dataset) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for field in contract.get("fields") or []:
        name = field["name"]
        if name not in ds.data_vars:
            continue
        selectors_obj: OrderedDict[str, str] = OrderedDict()
        selector_paths: OrderedDict[str, str] = OrderedDict()
        for sel in field.get("selectors") or []:
            dim = sel["dimension"]
            selectors_obj[dim] = str(sel["value"])
            selector_paths[dim] = dim
        channels.append({
            "field_id": _canonical_field_id(field),
            "array_path": name,
            "selectors": dict(selectors_obj),
            "selector_coordinate_paths": dict(selector_paths),
        })
    return channels


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return field["name"]
    parts = []
    for sel in selectors:
        parts.append(f"{sel['dimension']}={json.dumps(str(sel['value']))}")
    return f"{field['name']}[" + ",".join(parts) + "]"
