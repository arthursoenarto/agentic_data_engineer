from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p014-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
PROVIDER = "ECMWF"
STORE_NAME = "dataset.zarr"

_VARIABLE_ALIASES = {
    "temperature": ["temperature", "t"],
    "geopotential": ["geopotential", "z"],
}
_PRESSURE_NAMES = ("pressure_level", "isobaricInhPa", "isobaricInPa", "level")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_TIME_NAMES = ("time", "valid_time")


class PipelineError(ValueError):
    pass


def run_pipeline(contract_lock, inventory, cache_dir, output_dir):
    """Materialize a validated ERA5 pressure-level regular-grid request from a local fixture."""
    lock = _load_jsonish(contract_lock)
    inv = _load_jsonish(inventory)
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    contract = _extract_contract(lock)
    policy = _extract_optional_policy(lock)

    _validate_inventory(inv)
    _validate_policy(policy)
    request = _validate_contract(contract, inv)

    fixture = _verify_fixture(cache_root)
    if not fixture["complete"]:
        raise PipelineError("A complete source_fixture_manifest.json is required; network acquisition is intentionally disabled.")

    ds = _open_fixture_dataset(fixture["dataset_roots"])
    materialized = _filter_dataset(ds, request, inv)
    artifact = _publish_atomically(materialized, contract, inv, out_root)

    return {
        "cache": {
            "hits": len(fixture["entries"]),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [e["entry_id"] for e in fixture["entries"]],
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [],
    }


def _load_jsonish(value: Any) -> Any:
    if isinstance(value, (str, os.PathLike)):
        p = Path(value)
        if p.exists():
            return json.loads(p.read_text())
        return json.loads(str(value))
    return value


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    candidates = [lock]
    for key in ("contract", "dataset_contract", "selected_contract", "contract_lock"):
        if isinstance(lock, dict) and isinstance(lock.get(key), dict):
            candidates.append(lock[key])
    if isinstance(lock, dict) and isinstance(lock.get("contracts"), list):
        candidates.extend([c for c in lock["contracts"] if isinstance(c, dict)])
    for c in candidates:
        if c.get("schema_version") == "dataset_contract.v1":
            return c
    raise PipelineError("No dataset_contract.v1 object found in contract lock envelope.")


def _extract_optional_policy(lock: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("fixed_pipeline_policy", "pipeline_policy", "policy", "output_policy"):
        v = lock.get(key) if isinstance(lock, dict) else None
        if isinstance(v, dict):
            return v
    return None


def _validate_inventory(inv: dict[str, Any]) -> None:
    if inv.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("Unsupported inventory schema_version.")
    if inv.get("dataset_slug") != DATASET_SLUG or inv.get("dataset_id") != DATASET_ID:
        raise PipelineError("Inventory dataset identity does not match this adapter.")


def _validate_policy(policy: dict[str, Any] | None) -> None:
    if policy is None:
        return
    if policy.get("provider") != PROVIDER or policy.get("dataset_id") != DATASET_ID:
        raise PipelineError("Fixed output policy/provider identity mismatch.")
    if policy.get("acquisition_format") != "grib" or policy.get("publication_format") != "zarr":
        raise PipelineError("Unsupported fixed pipeline policy formats.")
    zpol = policy.get("zarr") or {}
    if zpol.get("format_version") != 3 or zpol.get("consolidated_metadata") is not True:
        raise PipelineError("This adapter publishes only consolidated Zarr v3.")


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return field["name"]
    parts = [f"{s['dimension']}={json.dumps(str(s['value']), separators=(',', ':'))}" for s in selectors]
    return f"{field['name']}[" + ",".join(parts) + "]"


def _validate_contract(c: dict[str, Any], inv: dict[str, Any]) -> dict[str, Any]:
    if c.get("dataset_slug") != DATASET_SLUG:
        raise PipelineError("Contract dataset_slug is not supported by this adapter.")
    adv = c.get("advanced_options") or {}
    if adv.get("dataset_id") != DATASET_ID:
        raise PipelineError("Contract advanced_options.dataset_id mismatch.")
    if adv.get("data_format") != "grib" or adv.get("download_format") != "unarchived":
        raise PipelineError("This fixed policy expects GRIB acquisition and unarchived downloads.")
    if adv.get("product_type") != ["reanalysis"]:
        raise PipelineError("Only product_type ['reanalysis'] is valid for this policy.")

    scope = c.get("scope") or {}
    if (scope.get("product_type") or "reanalysis") != "reanalysis":
        raise PipelineError("Only reanalysis product_type is supported.")
    geo = scope.get("geography") or {}
    area = geo.get("cds_area", inv.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise PipelineError("Invalid CDS area; expected [north, west, south, east].")
    north, west, south, east = map(float, area)
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("Invalid geography bounds.")

    times = ((scope.get("time") or {}).get("selected_times") or [])
    if not times:
        raise PipelineError("At least one selected UTC time is required.")
    for t in times:
        if t not in inv["options"]["time"]:
            raise PipelineError(f"Invalid selected time {t!r}.")
    if (scope.get("time") or {}).get("timezone", "UTC") != "UTC":
        raise PipelineError("Only UTC time selections are supported.")

    dr = scope.get("date_range") or {}
    start = date.fromisoformat(dr.get("start_date"))
    end = date.fromisoformat(dr.get("end_date"))
    if end < start:
        raise PipelineError("date_range.end_date must be on or after start_date.")
    years = set(inv["options"]["year"])
    if str(start.year) not in years or str(end.year) not in years:
        raise PipelineError("Requested year is outside the frozen inventory options.")

    fields = c.get("fields") or []
    if not fields:
        raise PipelineError("At least one field is required.")
    seen = set()
    by_var: dict[str, list[str]] = {}
    channels = []
    for f in fields:
        name = f.get("name")
        if name not in inv["options"]["variable"]:
            raise PipelineError(f"Invalid ERA5 pressure-level variable {name!r}.")
        selectors = f.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise PipelineError("ERA5 pressure-level fields require exactly one pressure_level selector.")
        level = str(selectors[0].get("value"))
        if level not in inv["options"]["pressure_level"]:
            raise PipelineError(f"Invalid pressure level {level!r}.")
        fid = _canonical_field_id(f)
        if fid in seen:
            raise PipelineError(f"Duplicate requested field-selector channel {fid}.")
        seen.add(fid)
        by_var.setdefault(name, [])
        if level not in by_var[name]:
            by_var[name].append(level)
        channels.append({"field": f, "field_id": fid, "variable": name, "level": level})
    return {"area": [north, west, south, east], "times": list(times), "start": start, "end": end, "inclusive": bool(dr.get("inclusive", True)), "by_var": by_var, "channels": channels}


def _safe_join(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if root != p and root not in p.parents:
        raise PipelineError("Fixture manifest contains a path outside cache_dir.")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_fixture(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        return {"complete": False, "entries": [], "dataset_roots": []}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("Unsupported source fixture manifest schema_version.")
    entries = manifest.get("entries") or []
    if not entries:
        raise PipelineError("source_fixture_manifest.json has no entries.")
    roots: list[Path] = []
    verified_entries = []
    for e in entries:
        rel = e.get("relative_path")
        p = _safe_join(cache_root, rel)
        if not p.is_file():
            raise PipelineError(f"Fixture entry is missing: {rel}")
        size = p.stat().st_size
        if size != int(e.get("size_bytes")):
            raise PipelineError(f"Fixture entry size mismatch: {rel}")
        digest = _sha256(p)
        if digest != e.get("sha256"):
            raise PipelineError(f"Fixture entry sha256 mismatch: {rel}")
        verified_entries.append({"entry_id": str(e.get("entry_id", digest[:16]))})
        parts = Path(rel).parts
        zidx = next((i for i, part in enumerate(parts) if part.endswith(".zarr")), None)
        if zidx is not None:
            roots.append(cache_root.joinpath(*parts[: zidx + 1]).resolve())
        else:
            roots.append(p)
    uniq = []
    for r in roots:
        if r not in uniq:
            uniq.append(r)
    return {"complete": True, "entries": verified_entries, "dataset_roots": uniq}


def _promote_scalar_pressure_coords(ds: xr.Dataset) -> xr.Dataset:
    for n in _PRESSURE_NAMES:
        if n in ds.coords and n not in ds.dims and getattr(ds[n], "ndim", 0) == 0:
            return ds.expand_dims(dim=n)
    return ds


def _open_fixture_dataset(roots: list[Path]) -> xr.Dataset:
    datasets = []
    for r in roots:
        suffix = r.suffix.lower()
        if r.is_dir() and suffix == ".zarr":
            datasets.append(_promote_scalar_pressure_coords(xr.open_zarr(r, consolidated=False, mask_and_scale=True, decode_times=True)))
        elif suffix in {".nc", ".nc4", ".cdf"}:
            datasets.append(_promote_scalar_pressure_coords(xr.open_dataset(r, mask_and_scale=True, decode_times=True)))
        elif suffix in {".grib", ".grb", ".grib2", ".grb2"}:
            try:
                import cfgrib
                opened = cfgrib.open_datasets(str(r), backend_kwargs={"indexpath": ""})
                datasets.extend(_promote_scalar_pressure_coords(d) for d in opened)
            except Exception as exc:
                raise PipelineError(f"Unable to decode GRIB fixture {r.name}: {exc}") from exc
        else:
            raise PipelineError(f"Unsupported fixture file type: {r.name}")
    if not datasets:
        raise PipelineError("No fixture datasets could be opened.")
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")


def _find_coord_name(obj: xr.Dataset | xr.DataArray, names: tuple[str, ...]) -> str:
    for n in names:
        if n in obj.coords or n in obj.dims:
            return n
    raise PipelineError(f"Could not find any coordinate/dimension named one of {names}.")


def _find_variable(ds: xr.Dataset, requested: str) -> str:
    for name in _VARIABLE_ALIASES.get(requested, [requested]):
        if name in ds.data_vars:
            return name
    for v in ds.data_vars:
        attrs = ds[v].attrs
        if attrs.get("GRIB_cfVarName") == requested or attrs.get("standard_name") == requested or attrs.get("long_name") == requested:
            return v
    raise PipelineError(f"Requested variable {requested!r} was not found in the verified fixture.")


def _expected_datetimes(req: dict[str, Any]) -> np.ndarray:
    end = req["end"] if req["inclusive"] else req["end"] - timedelta(days=1)
    out = []
    d = req["start"]
    while d <= end:
        for hhmm in req["times"]:
            hh, mm = map(int, hhmm.split(":"))
            out.append(datetime.combine(d, time(hh, mm), tzinfo=timezone.utc).replace(tzinfo=None))
        d += timedelta(days=1)
    return np.asarray(out, dtype="datetime64[ns]")


def _positions_for_exact_times(values: np.ndarray, expected: np.ndarray) -> list[int]:
    vals = values.astype("datetime64[ns]")
    positions = []
    for ts in expected:
        matches = np.where(vals == ts)[0]
        if len(matches) != 1:
            raise PipelineError(f"Fixture does not contain exactly one selected timestamp {str(ts)}.")
        positions.append(int(matches[0]))
    return positions


def _norm_level(v: Any) -> str:
    if isinstance(v, bytes):
        v = v.decode()
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(v)


def _positions_for_levels(values: np.ndarray, requested: list[str]) -> list[int]:
    normed = [_norm_level(v) for v in values]
    positions = []
    for level in requested:
        matches = [i for i, v in enumerate(normed) if v == str(level)]
        if len(matches) != 1:
            raise PipelineError(f"Fixture does not contain exactly one selected pressure level {level}.")
        positions.append(matches[0])
    return positions


def _geo_positions(lat_values: np.ndarray, lon_values: np.ndarray, area: list[float]) -> tuple[np.ndarray, np.ndarray]:
    north, west, south, east = area
    lat = np.asarray(lat_values, dtype=float)
    lon = np.asarray(lon_values, dtype=float)
    lat_idx = np.where((lat <= north + 1e-9) & (lat >= south - 1e-9))[0]
    full_global = north == 90.0 and south == -90.0 and west == -180.0 and east == 180.0
    if full_global:
        lon_idx = np.arange(lon.size)
    else:
        if np.nanmin(lon) >= 0 and west < 0:
            west = west % 360
            east = east % 360
        if west <= east:
            lon_idx = np.where((lon >= west - 1e-9) & (lon <= east + 1e-9))[0]
        else:
            lon_idx = np.where((lon >= west - 1e-9) | (lon <= east + 1e-9))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise PipelineError("Geography selection produced an empty grid.")
    return lat_idx, lon_idx


def _filtered_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for k, v in attrs.items():
        if any(secret in k.lower() for secret in ("path", "token", "key", "secret", "credential")):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
    return safe


def _filter_dataset(ds: xr.Dataset, req: dict[str, Any], inv: dict[str, Any]) -> dict[str, Any]:
    tname = _find_coord_name(ds, _TIME_NAMES)
    lat_name = _find_coord_name(ds, _LAT_NAMES)
    lon_name = _find_coord_name(ds, _LON_NAMES)
    expected_times = _expected_datetimes(req)
    tpos = _positions_for_exact_times(np.asarray(ds[tname].values), expected_times)
    lat_idx, lon_idx = _geo_positions(ds[lat_name].values, ds[lon_name].values, req["area"])

    result = {
        "time": np.asarray(ds[tname].values, dtype="datetime64[ns]")[tpos],
        "latitude": np.asarray(ds[lat_name].values)[lat_idx],
        "longitude": np.asarray(ds[lon_name].values)[lon_idx],
        "variables": {},
        "channels": req["channels"],
    }
    for var, levels in req["by_var"].items():
        source_var = _find_variable(ds, var)
        da = ds[source_var]
        p_name = _find_coord_name(da, _PRESSURE_NAMES)
        ppos = _positions_for_levels(np.asarray(da[p_name].values), levels)
        indexers = {}
        if tname in da.dims:
            indexers[tname] = tpos
        if p_name in da.dims:
            indexers[p_name] = ppos
        if lat_name in da.dims:
            indexers[lat_name] = lat_idx
        if lon_name in da.dims:
            indexers[lon_name] = lon_idx
        subset = da.isel(indexers)
        rename = {}
        if tname in subset.dims:
            rename[tname] = "time"
        if p_name in subset.dims:
            rename[p_name] = f"pressure_level__{var}"
        if lat_name in subset.dims:
            rename[lat_name] = "latitude"
        if lon_name in subset.dims:
            rename[lon_name] = "longitude"
        subset = subset.rename(rename)
        dim_order = ["time", f"pressure_level__{var}", "latitude", "longitude"]
        subset = subset.transpose(*[d for d in dim_order if d in subset.dims])
        coord_vals = np.asarray(da[p_name].values)[ppos]
        result["variables"][var] = {
            "data": np.asarray(subset.values),
            "pressure_values": coord_vals,
            "requested_levels": levels,
            "attrs": _filtered_attrs(da.attrs),
            "dtype": subset.dtype,
        }
    return result


def _blosc_codec():
    from zarr.codecs import BloscCodec, BloscShuffle
    shuffle = getattr(BloscShuffle, "bitshuffle", None) or getattr(BloscShuffle, "BITSHUFFLE", None) or "bitshuffle"
    return BloscCodec(cname="zstd", clevel=9, shuffle=shuffle)


def _create_array(group, name: str, data: np.ndarray, chunks: tuple[int, ...], dims: list[str], attrs: dict[str, Any] | None = None, codec: Any | None = None):
    kwargs = dict(name=name, shape=data.shape, chunks=chunks, dtype=data.dtype, overwrite=True, dimension_names=dims)
    if codec is not None:
        kwargs["compressors"] = [codec]
    arr = group.create_array(**kwargs)
    arr[...] = data
    arr.attrs.update(attrs or {})
    arr.attrs["_ARRAY_DIMENSIONS"] = dims
    return arr


def _publish_atomically(mat: dict[str, Any], contract: dict[str, Any], inv: dict[str, Any], out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / STORE_NAME
    tmp = out_root / f".{STORE_NAME}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    if final.exists() and not final.is_dir():
        raise PipelineError("Output store path exists and is not a directory.")

    group = zarr.open_group(str(tmp), mode="w", zarr_format=3)
    group.attrs.update({
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "provider": PROVIDER,
        "title": contract.get("title", inv.get("title", "ERA5 pressure levels")),
        "publication_format": "zarr_v3_consolidated",
    })
    _create_array(group, "time", mat["time"], (mat["time"].shape[0],), ["time"], {"standard_name": "time", "timezone": "UTC"})
    _create_array(group, "latitude", mat["latitude"], (mat["latitude"].shape[0],), ["latitude"], {"units": "degrees_north", "standard_name": "latitude"})
    _create_array(group, "longitude", mat["longitude"], (mat["longitude"].shape[0],), ["longitude"], {"units": "degrees_east", "standard_name": "longitude"})

    codec = _blosc_codec()
    channels = []
    for var, payload in mat["variables"].items():
        pname = f"pressure_level__{var}"
        pvals = np.asarray(payload["pressure_values"])
        _create_array(group, pname, pvals, (pvals.shape[0],), [pname], {"units": "hPa", "selector_dimension": "pressure_level"})
        data = np.asarray(payload["data"])
        chunks = (1, 1, mat["latitude"].shape[0], mat["longitude"].shape[0])
        attrs = dict(payload["attrs"])
        meta = inv.get("option_metadata", {}).get("variable", {}).get(var, {})
        if "units" not in attrs and meta.get("units"):
            attrs["units"] = meta["units"]
        attrs.update({"field_name": var, "selector_dimension": "pressure_level", "selector_coordinate": pname})
        _create_array(group, var, data, chunks, ["time", pname, "latitude", "longitude"], attrs, codec)

    zarr.consolidate_metadata(str(tmp))

    artifact = {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": [],
    }
    for ch in mat["channels"]:
        artifact["channels"].append({
            "field_id": ch["field_id"],
            "array_path": ch["variable"],
            "selectors": {"pressure_level": ch["level"]},
            "selector_coordinate_paths": {"pressure_level": f"pressure_level__{ch['variable']}"},
        })

    _validate_written_store(tmp, mat, artifact)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return artifact


def _array_metadata_json(store: Path, array_path: str) -> dict[str, Any]:
    return json.loads((store / array_path / "zarr.json").read_text())


def _validate_written_store(store: Path, mat: dict[str, Any], artifact: dict[str, Any]) -> None:
    root_meta = json.loads((store / "zarr.json").read_text())
    if root_meta.get("zarr_format") != 3:
        raise PipelineError("Published store is not Zarr v3.")
    if "consolidated_metadata" not in root_meta:
        raise PipelineError("Published Zarr v3 metadata is not consolidated.")
    group = zarr.open_group(str(store), mode="r")
    for var, payload in mat["variables"].items():
        arr = group[var]
        if tuple(arr.chunks)[1] != 1:
            raise PipelineError("Pressure selector chunk size must be exactly one.")
        meta = _array_metadata_json(store, var)
        text = json.dumps(meta).lower()
        if "zstd" not in text or "9" not in text or "bitshuffle" not in text:
            raise PipelineError("Data array is not encoded with lossless Blosc-Zstd bitshuffle clevel 9.")
        coord_name = f"pressure_level__{var}"
        got_levels = [_norm_level(x) for x in np.asarray(group[coord_name][...])]
        if got_levels != [str(x) for x in payload["requested_levels"]]:
            raise PipelineError("Grouped variable contains unrequested or reordered pressure levels.")
    reopened = xr.open_zarr(store, consolidated=True)
    try:
        np.testing.assert_array_equal(reopened["time"].values.astype("datetime64[ns]"), mat["time"])
        np.testing.assert_array_equal(reopened["latitude"].values, mat["latitude"])
        np.testing.assert_array_equal(reopened["longitude"].values, mat["longitude"])
        for ch in artifact["channels"]:
            var = ch["array_path"]
            coord = ch["selector_coordinate_paths"]["pressure_level"]
            level = ch["selectors"]["pressure_level"]
            idx = [i for i, v in enumerate([_norm_level(x) for x in reopened[coord].values]) if v == level][0]
            reopened_plane = reopened[var].isel({coord: idx}).values
            expected_plane = mat["variables"][var]["data"][:, idx, :, :]
            np.testing.assert_equal(reopened_plane, expected_plane)
    finally:
        reopened.close()
