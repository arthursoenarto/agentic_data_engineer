from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p018-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "dataset.zarr"
SHORT_NAME_ALIASES = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
}
DIM_ALIASES = {
    "time": ("time", "valid_time"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "level", "plev"),
}
LOSSLESS_CODEC_DECLARATION = {"family": "blosc", "cname": "zstd", "clevel": 9, "shuffle": "bitshuffle"}


class ContractError(ValueError):
    pass


class FixtureError(RuntimeError):
    pass


class PublicationError(RuntimeError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize an ERA5 pressure-level fixture as consolidated Zarr v3.

    The function is deterministic and offline-only. A complete fixture manifest at
    cache_dir/source_fixture_manifest.json is mandatory for this pilot family.
    """
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    contract = _extract_contract(contract_lock)
    spec = _validate_contract(contract, inventory)
    fixture_entries = _verify_source_fixture(cache_root)

    source = _open_fixture_dataset([e["path"] for e in fixture_entries])
    normalized = _normalize_dataset(source)
    filtered = _filter_and_group_dataset(normalized, spec, inventory)
    artifact = _build_artifact(spec, filtered)

    out_root.mkdir(parents=True, exist_ok=True)
    final_store = out_root / STORE_NAME
    tmp_parent = out_root / ".tmp"
    tmp_parent.mkdir(exist_ok=True)
    tmp_store = Path(tempfile.mkdtemp(prefix="publish-", dir=tmp_parent)) / STORE_NAME
    try:
        _write_zarr_v3(filtered, tmp_store)
        _validate_publication(tmp_store, filtered, artifact)
        if final_store.exists():
            if final_store.is_dir():
                shutil.rmtree(final_store)
            else:
                final_store.unlink()
        os.replace(tmp_store, final_store)
        _validate_publication(final_store, filtered, artifact)
    finally:
        try:
            if tmp_store.exists():
                shutil.rmtree(tmp_store)
            if tmp_store.parent.exists():
                shutil.rmtree(tmp_store.parent)
        except FileNotFoundError:
            pass

    reused = [f"fixture:{e['entry_id']}:{e['sha256'][:16]}" for e in fixture_entries]
    return {
        "cache": {
            "hits": len(reused),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused,
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [],
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    if isinstance(lock, dict):
        for key in ("contract", "dataset_contract", "selected_contract", "request", "lock"):
            val = lock.get(key)
            if isinstance(val, dict):
                try:
                    return _extract_contract(val)
                except ContractError:
                    pass
        for val in lock.values():
            if isinstance(val, dict):
                try:
                    return _extract_contract(val)
                except ContractError:
                    pass
    raise ContractError("contract_lock does not contain a dataset_contract.v1 object")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ContractError("unsupported inventory schema_version")
    if contract.get("dataset_slug") != DATASET_SLUG or inventory.get("dataset_slug") != DATASET_SLUG:
        raise ContractError("contract and inventory must target reanalysis_era5_pressure_levels")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ContractError("inventory dataset_id mismatch")
    opts = inventory.get("options", {})
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id") != DATASET_ID:
        raise ContractError("advanced_options.dataset_id mismatch")
    if adv.get("data_format") != "grib":
        raise ContractError("fixed policy requires grib acquisition_format")
    if adv.get("download_format") not in opts.get("download_format", []):
        raise ContractError("download_format is not allowed by inventory")
    product_type = adv.get("product_type", contract.get("scope", {}).get("product_type", ["reanalysis"]))
    if isinstance(product_type, str):
        product_type = [product_type]
    if not product_type or any(p not in opts.get("product_type", []) for p in product_type):
        raise ContractError("product_type is not allowed by inventory")

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise ContractError("end_date precedes start_date")
    if not dr.get("inclusive", True):
        end = end - timedelta(days=1)
        if end < start:
            raise ContractError("exclusive date range is empty")
    years = {str(y) for y in range(start.year, end.year + 1)}
    if any(y not in opts.get("year", []) for y in years):
        raise ContractError("requested year is outside inventory options")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone", "UTC") != "UTC":
        raise ContractError("only UTC time selections are valid for ERA5 inventory")
    selected_times = time_scope.get("selected_times") or []
    if not selected_times:
        raise ContractError("at least one selected time is required")
    for t in selected_times:
        _parse_hhmm(t)
        if t not in opts.get("time", []):
            raise ContractError(f"time {t!r} is not allowed by inventory")

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area"))
    if not isinstance(area, list) or len(area) != 4:
        raise ContractError("cds_area must be [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90) or not (-360 <= west <= 360) or not (-360 <= east <= 360):
        raise ContractError("cds_area values are outside valid latitude/longitude bounds")
    if geography.get("cds_area_order", ["north", "west", "south", "east"]) != ["north", "west", "south", "east"]:
        raise ContractError("unsupported cds_area_order")

    groups: dict[str, list[str]] = defaultdict(list)
    fields = contract.get("fields") or []
    if not fields:
        raise ContractError("at least one field is required")
    channels: list[dict[str, Any]] = []
    for f in fields:
        name = f.get("name")
        if name not in opts.get("variable", []):
            raise ContractError(f"variable {name!r} is not allowed by inventory")
        selectors = f.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ContractError("ERA5 pressure-level fields require exactly one pressure_level selector")
        level = str(selectors[0].get("value"))
        if level not in opts.get("pressure_level", []):
            raise ContractError(f"pressure level {level!r} is not allowed by inventory")
        if level not in groups[name]:
            groups[name].append(level)
        channels.append({"field_id": _field_id(name, selectors), "name": name, "pressure_level": level})

    expected_times = _expected_timestamps(start, end, selected_times)
    return {
        "start": start,
        "end": end,
        "selected_times": selected_times,
        "expected_times": expected_times,
        "area": [north, west, south, east],
        "groups": dict(groups),
        "channels": channels,
        "product_type": product_type,
    }


def _verify_source_fixture(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise FixtureError("complete local fixture is required: source_fixture_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise FixtureError("unsupported source fixture manifest schema_version")
    entries = manifest.get("entries") or []
    if not entries:
        raise FixtureError("source fixture manifest has no entries")
    verified: list[dict[str, Any]] = []
    for raw in entries:
        rel = raw.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise FixtureError("fixture relative_path must be a relative path")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise FixtureError("fixture path escapes cache_dir") from exc
        if not path.is_file():
            raise FixtureError(f"fixture file is missing: {rel}")
        size = path.stat().st_size
        if size != int(raw.get("size_bytes")):
            raise FixtureError(f"fixture size mismatch: {rel}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != str(raw.get("sha256", "")).lower():
            raise FixtureError(f"fixture sha256 mismatch: {rel}")
        verified.append({"entry_id": str(raw.get("entry_id", rel)), "path": path, "sha256": sha, "size_bytes": size})
    verified.sort(key=lambda e: e["entry_id"])
    return verified


def _open_fixture_dataset(paths: list[Path]) -> xr.Dataset:
    datasets = []
    for p in paths:
        suffixes = "".join(p.suffixes).lower()
        if suffixes.endswith((".nc", ".nc4", ".netcdf")):
            ds = xr.open_dataset(p, decode_cf=True, mask_and_scale=True).load()
        elif suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
            try:
                ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_cf=True).load()
            except Exception as exc:  # pragma: no cover - depends on external fixture format
                raise FixtureError(f"could not decode GRIB fixture {p.name}: {exc}") from exc
        else:
            try:
                ds = xr.open_dataset(p, decode_cf=True, mask_and_scale=True).load()
            except Exception:
                try:
                    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_cf=True).load()
                except Exception as exc:
                    raise FixtureError(f"could not decode fixture file {p.name}: {exc}") from exc
        datasets.append(ds)
    if len(datasets) == 1:
        out = datasets[0]
    else:
        out = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
    return out.load()


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for std, aliases in DIM_ALIASES.items():
        for a in aliases:
            if a in ds.dims or a in ds.coords:
                rename[a] = std
                break
    for long_name, aliases in SHORT_NAME_ALIASES.items():
        for a in aliases:
            if a in ds.data_vars:
                rename[a] = long_name
                break
    out = ds.rename({k: v for k, v in rename.items() if k != v})
    required_dims = ["time", "latitude", "longitude"]
    for d in required_dims:
        if d not in out.coords and d not in out.dims:
            raise FixtureError(f"source fixture is missing required coordinate/dimension {d}")
    if "time" in out.coords:
        out = out.assign_coords(time=pd.to_datetime(out["time"].values).to_numpy(dtype="datetime64[ns]"))
    return out


def _filter_and_group_dataset(ds: xr.Dataset, spec: dict[str, Any], inventory: dict[str, Any]) -> xr.Dataset:
    if "time" not in ds.coords:
        raise FixtureError("fixture dataset has no time coordinate")
    source_times = pd.to_datetime(ds["time"].values).to_numpy(dtype="datetime64[ns]")
    expected = spec["expected_times"]
    missing = [str(pd.Timestamp(t)) for t in expected if t not in set(source_times)]
    if missing:
        raise FixtureError("source fixture is incomplete for requested times: " + ", ".join(missing[:5]))
    ds = ds.sel(time=expected)
    ds = _apply_area(ds, spec["area"])

    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {"time": ds["time"], "latitude": ds["latitude"], "longitude": ds["longitude"]}
    for var_name in sorted(spec["groups"].keys()):
        if var_name not in ds.data_vars:
            raise FixtureError(f"source fixture is missing requested variable {var_name}")
        da = ds[var_name].load()
        selected_levels = spec["groups"][var_name]
        if "pressure_level" not in da.dims:
            if len(selected_levels) != 1:
                raise FixtureError(f"requested pressure-level variable {var_name} lacks pressure_level dimension")
            scalar_level = da.coords.get("pressure_level")
            if scalar_level is not None:
                _indices_for_levels(np.asarray(scalar_level.values).reshape(-1), selected_levels)
                da = da.expand_dims("pressure_level")
            else:
                level = selected_levels[0]
                level_value: Any = int(level) if level.isdigit() else level
                da = da.expand_dims({"pressure_level": np.array([level_value])})
        indices = _indices_for_levels(da["pressure_level"].values, selected_levels)
        da = da.isel(pressure_level=indices)
        selector_dim = f"pressure_level_{var_name}"
        da = da.rename({"pressure_level": selector_dim})
        da = da.transpose("time", selector_dim, "latitude", "longitude")
        coords[selector_dim] = (selector_dim, da[selector_dim].values, _coord_attrs(ds.get("pressure_level"), inventory))
        attrs = dict(da.attrs)
        meta = inventory.get("option_metadata", {}).get("variable", {}).get(var_name, {})
        attrs.setdefault("long_name", meta.get("label", var_name))
        if meta.get("units"):
            attrs.setdefault("units", _strip_html_units(meta["units"]))
        if meta.get("description"):
            attrs.setdefault("description", meta["description"])
        attrs["source_native_variable"] = var_name
        da.attrs = attrs
        for bad in ("scale_factor", "add_offset", "_FillValue", "missing_value"):
            da.encoding.pop(bad, None)
        data_vars[var_name] = da
    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "provider": "ECMWF",
        "publication_format": "zarr_v3_consolidated",
        "codec": json.dumps(LOSSLESS_CODEC_DECLARATION, sort_keys=True),
    })
    for name in out.variables:
        out[name].encoding.clear()
    return out


def _apply_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_mask = (lat >= south) & (lat <= north)
    if not lat_mask.any():
        raise FixtureError("requested latitude area selects no source cells")
    if west <= -180 and east >= 180:
        lon_mask = np.ones(lon.shape, dtype=bool)
    else:
        lon_vals = lon.astype(float)
        if lon_vals.min() >= 0 and west < 0:
            west = west % 360
            east = east % 360
        if west <= east:
            lon_mask = (lon_vals >= west) & (lon_vals <= east)
        else:
            lon_mask = (lon_vals >= west) | (lon_vals <= east)
    if not lon_mask.any():
        raise FixtureError("requested longitude area selects no source cells")
    return ds.isel(latitude=np.where(lat_mask)[0], longitude=np.where(lon_mask)[0])


def _indices_for_levels(values: np.ndarray, selected: list[str]) -> list[int]:
    index: dict[str, int] = {}
    for i, v in enumerate(values):
        if isinstance(v, (np.integer, int)):
            key = str(int(v))
        elif isinstance(v, (np.floating, float)) and float(v).is_integer():
            key = str(int(v))
        else:
            key = str(v)
        index[key] = i
    missing = [lvl for lvl in selected if lvl not in index]
    if missing:
        raise FixtureError("source fixture is missing requested pressure levels: " + ", ".join(missing))
    return [index[lvl] for lvl in selected]


def _build_artifact(spec: dict[str, Any], ds: xr.Dataset) -> dict[str, Any]:
    channels = []
    for ch in spec["channels"]:
        selector_dim = f"pressure_level_{ch['name']}"
        channels.append({
            "field_id": ch["field_id"],
            "array_path": ch["name"],
            "selectors": {"pressure_level": ch["pressure_level"]},
            "selector_coordinate_paths": {"pressure_level": selector_dim},
        })
    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": channels,
    }


def _write_zarr_v3(ds: xr.Dataset, store_path: Path) -> None:
    lat_count = int(ds.sizes["latitude"])
    lon_count = int(ds.sizes["longitude"])
    encoding: dict[str, dict[str, Any]] = {}
    codec = _make_blosc_zstd_codec()
    for name, da in ds.data_vars.items():
        selector_dim = [d for d in da.dims if d.startswith("pressure_level_")][0]
        encoding[name] = {"chunks": (1, 1, lat_count, lon_count), "compressors": [codec]}
    for cname in ds.coords:
        if cname not in encoding:
            encoding[cname] = {"compressors": [codec]}
    for enc in encoding.values():
        enc.pop("scale_factor", None)
        enc.pop("add_offset", None)
    try:
        ds.to_zarr(store_path, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    except TypeError:
        # Compatibility for maintained stacks during the zarr-v3 transition.
        for enc in encoding.values():
            enc["compressor"] = enc.pop("compressors")[0]
        ds.to_zarr(store_path, mode="w", zarr_version=3, consolidated=True, encoding=encoding)


def _make_blosc_zstd_codec() -> Any:
    try:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")
    except Exception:
        from numcodecs import Blosc
        return Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)


def _validate_publication(store_path: Path, expected: xr.Dataset, artifact: dict[str, Any]) -> None:
    if not (store_path / "zarr.json").exists():
        raise PublicationError("Zarr v3 store lacks root zarr.json")
    root_meta = json.loads((store_path / "zarr.json").read_text(encoding="utf-8"))
    if root_meta.get("zarr_format") != 3:
        raise PublicationError("published store is not Zarr v3")
    if "consolidated_metadata" not in root_meta:
        raise PublicationError("published Zarr v3 metadata is not consolidated")
    reopened = _open_published_zarr(store_path).load()
    if set(reopened.data_vars) != set(expected.data_vars):
        raise PublicationError("reopened data variables differ from expected variables")
    for name in expected.coords:
        if name not in reopened.coords:
            raise PublicationError(f"missing coordinate after reopen: {name}")
        if not np.array_equal(reopened[name].values, expected[name].values, equal_nan=True):
            raise PublicationError(f"coordinate values changed after publication: {name}")
    for name in expected.data_vars:
        if reopened[name].dims != expected[name].dims:
            raise PublicationError(f"dimensions changed for {name}")
        if reopened[name].shape != expected[name].shape:
            raise PublicationError(f"shape changed for {name}")
        if not np.array_equal(reopened[name].values, expected[name].values, equal_nan=True):
            raise PublicationError(f"data values changed for {name}")
        _validate_array_chunks_and_codec(store_path, name, expected[name])
    for ch in artifact["channels"]:
        if ch["array_path"] not in expected.data_vars:
            raise PublicationError("channel array_path does not name a published data array")
        coord_path = ch["selector_coordinate_paths"].get("pressure_level")
        if coord_path not in expected.coords:
            raise PublicationError("channel selector coordinate path is missing")
        coord_vals = {str(int(v)) if isinstance(v, (np.integer, int, float, np.floating)) and float(v).is_integer() else str(v) for v in expected[coord_path].values}
        if ch["selectors"].get("pressure_level") not in coord_vals:
            raise PublicationError("channel selector is not present in selector coordinate")


def _open_published_zarr(store_path: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(store_path, consolidated=True, zarr_format=3)
    except TypeError:
        return xr.open_zarr(store_path, consolidated=True)


def _validate_array_chunks_and_codec(store_path: Path, name: str, da: xr.DataArray) -> None:
    meta_path = store_path / name / "zarr.json"
    if not meta_path.exists():
        raise PublicationError(f"array metadata missing for {name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_grid = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape") or meta.get("chunks")
    full = [1, 1, int(da.sizes["latitude"]), int(da.sizes["longitude"])]
    if list(chunk_grid) != full:
        raise PublicationError(f"unexpected chunk shape for {name}: {chunk_grid}")
    text = json.dumps(meta, sort_keys=True).lower()
    if "zstd" not in text or "bitshuffle" not in text or "scale_factor" in text or "add_offset" in text:
        raise PublicationError(f"codec metadata for {name} is not the required lossless Zstd/bitshuffle configuration")
    if "clevel" in text and "9" not in text:
        raise PublicationError(f"codec metadata for {name} does not record high compression level")


def _field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    parts = []
    for s in selectors:
        parts.append(f"{s['dimension']}={json.dumps(str(s['value']), ensure_ascii=False)}")
    return f"{name}[{','.join(parts)}]"


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"invalid {label}") from exc


def _parse_hhmm(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ContractError(f"invalid selected time {value!r}") from exc


def _expected_timestamps(start: date, end: date, selected_times: list[str]) -> np.ndarray:
    stamps = []
    d = start
    parsed = [_parse_hhmm(t) for t in selected_times]
    while d <= end:
        for tt in parsed:
            stamps.append(datetime.combine(d, tt, tzinfo=timezone.utc).replace(tzinfo=None))
        d += timedelta(days=1)
    return pd.to_datetime(stamps).to_numpy(dtype="datetime64[ns]")


def _coord_attrs(source_coord: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    attrs = dict(getattr(source_coord, "attrs", {}) or {})
    attrs.setdefault("units", inventory.get("option_units", {}).get("pressure_level", "hPa"))
    attrs.setdefault("long_name", "pressure level")
    return attrs


def _strip_html_units(units: str) -> str:
    return units.replace("<sup>", "^").replace("</sup>", "").replace("&nbsp;", " ")
