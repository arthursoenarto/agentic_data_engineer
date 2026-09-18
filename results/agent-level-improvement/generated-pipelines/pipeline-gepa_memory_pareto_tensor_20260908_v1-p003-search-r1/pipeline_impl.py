from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p003-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
ZARR_STORE_NAME = "dataset.zarr"

VAR_ALIASES = {
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
TIME_NAMES = ("time", "valid_time")
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
PLEV_NAMES = ("pressure_level", "isobaricInhPa", "level", "plev")


@dataclass(frozen=True)
class FixtureEntry:
    entry_id: str
    path: Path
    size_bytes: int
    sha256: str


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize a validated ERA5 pressure-level lock from a complete local fixture to consolidated Zarr v3."""
    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Fixture verification intentionally happens before any provider/client/credential logic. This adapter has no
    # network fallback: the evaluator-owned fixture is authoritative for this pilot family.
    fixture_entries = _verify_source_fixture(cache_root)

    contract = _extract_contract(contract_lock)
    request = _validate_contract(contract, inventory)

    raw = _open_fixture_dataset([e.path for e in fixture_entries])
    filtered_source = _filter_source_dataset(raw, request)
    public, channels = _build_public_dataset(filtered_source, contract, inventory, request)

    contract_hash = _stable_hash(contract)
    tmp_store = output_root / f".{ZARR_STORE_NAME}.tmp-{contract_hash[:16]}"
    final_store = output_root / ZARR_STORE_NAME
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    try:
        _write_zarr_v3(public, tmp_store)
        _validate_publication(public, tmp_store, channels)
        if final_store.exists():
            shutil.rmtree(final_store)
        os.replace(tmp_store, final_store)
        _validate_publication(public, final_store, channels)
    finally:
        if tmp_store.exists():
            shutil.rmtree(tmp_store)

    reused = [f"fixture:{e.entry_id}:{e.sha256[:16]}" for e in fixture_entries]
    return {
        "cache": {
            "hits": len(fixture_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": reused,
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": ZARR_STORE_NAME,
            "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
            "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
            "channels": channels,
        },
        "warnings": [],
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    if not isinstance(lock, dict):
        raise ValueError("contract_lock must be a dictionary or an envelope containing a dataset_contract.v1 object")
    for key in ("contract", "dataset_contract", "selected_contract", "request_contract"):
        val = lock.get(key)
        if isinstance(val, dict) and val.get("schema_version") == "dataset_contract.v1":
            return val
    for val in lock.values():
        if isinstance(val, dict):
            try:
                return _extract_contract(val)
            except ValueError:
                pass
    raise ValueError("no dataset_contract.v1 object found in contract_lock envelope")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ValueError("unsupported inventory schema_version")
    if contract.get("dataset_slug") != DATASET_SLUG or inventory.get("dataset_slug") != DATASET_SLUG:
        raise ValueError("contract and inventory must both target reanalysis_era5_pressure_levels")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected inventory dataset_id")
    options = inventory.get("options", {})

    adv = contract.get("advanced_options") or {}
    data_format = adv.get("data_format", inventory.get("defaults", {}).get("data_format"))
    download_format = adv.get("download_format", inventory.get("defaults", {}).get("download_format"))
    product_type = adv.get("product_type", inventory.get("defaults", {}).get("product_type", ["reanalysis"]))
    if isinstance(product_type, str):
        product_type = [product_type]
    _assert_options("product_type", product_type, options)
    if product_type != ["reanalysis"]:
        raise ValueError("this fixed policy supports ERA5 pressure-level reanalysis products only")
    _assert_options("data_format", [data_format], options)
    _assert_options("download_format", [download_format], options)

    fields = contract.get("fields") or []
    if not fields:
        raise ValueError("contract must request at least one field")
    field_specs = []
    pressure_levels: list[str] = []
    variables: list[str] = []
    for field in fields:
        name = field.get("name")
        _assert_options("variable", [name], options)
        selectors = field.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ValueError("ERA5 pressure-level fields must declare exactly one pressure_level selector")
        level = str(selectors[0].get("value"))
        _assert_options("pressure_level", [level], options)
        if selectors[0].get("unit") not in (None, "hPa"):
            raise ValueError("pressure_level selector unit must be hPa when provided")
        variables.append(name)
        pressure_levels.append(level)
        field_specs.append({"name": name, "level": level, "field": field})

    scope = contract.get("scope") or {}
    if scope.get("product_type", "reanalysis") != "reanalysis":
        raise ValueError("scope.product_type must be reanalysis")
    area = ((scope.get("geography") or {}).get("cds_area") or inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise ValueError("scope.geography.cds_area must be [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError("invalid geographic area bounds")

    times = ((scope.get("time") or {}).get("selected_times") or [])
    if not times:
        raise ValueError("scope.time.selected_times must be non-empty")
    _assert_options("time", times, options)
    if (scope.get("time") or {}).get("timezone", "UTC") != "UTC":
        raise ValueError("only UTC selected_times are supported")

    dr = scope.get("date_range") or {}
    start = date.fromisoformat(dr.get("start_date"))
    end = date.fromisoformat(dr.get("end_date"))
    if end < start:
        raise ValueError("date_range end_date precedes start_date")
    selected_datetimes = _selected_datetimes(start, end, bool(dr.get("inclusive", True)), times)
    years = sorted({f"{t.year:04d}" for t in selected_datetimes})
    months = sorted({f"{t.month:02d}" for t in selected_datetimes})
    days = sorted({f"{t.day:02d}" for t in selected_datetimes})
    _assert_options("year", years, options)
    _assert_options("month", months, options)
    _assert_options("day", days, options)

    return {
        "field_specs": field_specs,
        "variables": sorted(set(variables)),
        "pressure_levels": sorted(set(pressure_levels), key=lambda x: int(float(x))),
        "area": [north, west, south, east],
        "datetimes": selected_datetimes,
        "data_format": data_format,
        "download_format": download_format,
    }


def _assert_options(name: str, values: list[str], options: dict[str, Any]) -> None:
    allowed = set(options.get(name) or [])
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"invalid {name} option(s): {bad}")


def _selected_datetimes(start: date, end: date, inclusive: bool, selected_times: list[str]) -> list[pd.Timestamp]:
    last = end if inclusive else end - timedelta(days=1)
    out: list[pd.Timestamp] = []
    cur = start
    while cur <= last:
        for hhmm in selected_times:
            hour, minute = [int(x) for x in hhmm.split(":")]
            out.append(pd.Timestamp(datetime(cur.year, cur.month, cur.day, hour, minute)))
        cur += timedelta(days=1)
    return out


def _verify_source_fixture(cache_root: Path) -> list[FixtureEntry]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("complete local source fixture required: cache_dir/source_fixture_manifest.json was not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ValueError("unsupported source fixture manifest schema_version")
    entries = manifest.get("entries") or []
    if not entries:
        raise ValueError("source fixture manifest must contain at least one entry")
    verified: list[FixtureEntry] = []
    for ent in entries:
        rel = ent.get("relative_path")
        p = (cache_root / rel).resolve()
        if cache_root not in p.parents and p != cache_root:
            raise ValueError("fixture entry escapes cache_dir")
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"fixture file not found: {rel}")
        size = p.stat().st_size
        expected_size = int(ent.get("size_bytes"))
        if size != expected_size:
            raise ValueError(f"fixture size mismatch for {rel}")
        digest = _sha256_file(p)
        expected_sha = str(ent.get("sha256", "")).lower()
        if digest != expected_sha:
            raise ValueError(f"fixture sha256 mismatch for {rel}")
        verified.append(FixtureEntry(str(ent.get("entry_id")), p, size, digest))
    return verified


def _open_fixture_dataset(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for p in sorted(paths, key=lambda x: str(x)):
        suffixes = "".join(p.suffixes).lower()
        if suffixes.endswith(".zarr"):
            ds = xr.open_zarr(p, consolidated=True)
        elif p.suffix.lower() in {".grib", ".grb", ".grb2"}:
            ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_cf=True, mask_and_scale=True)
        else:
            try:
                ds = xr.open_dataset(p, engine="netcdf4", decode_cf=True, mask_and_scale=True)
            except Exception:
                ds = xr.open_dataset(p, decode_cf=True, mask_and_scale=True)
        datasets.append(_promote_scalar_pressure_level(ds).load())
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts").load()
    except Exception:
        time_name = _find_name(datasets[0], TIME_NAMES)
        return xr.concat(datasets, dim=time_name).sortby(time_name).load()


def _promote_scalar_pressure_level(ds: xr.Dataset) -> xr.Dataset:
    try:
        plev_name = _find_name(ds, PLEV_NAMES)
    except ValueError:
        return ds
    if plev_name in ds.dims:
        return ds
    if plev_name in ds.coords and ds[plev_name].ndim == 0:
        coord = ds[plev_name]
        attrs = dict(coord.attrs)
        promoted = ds.drop_vars(plev_name).expand_dims({plev_name: np.atleast_1d(coord.values)})
        promoted[plev_name].attrs.update(attrs)
        return promoted
    return ds


def _filter_source_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    time_name = _find_name(ds, TIME_NAMES)
    lat_name = _find_name(ds, LAT_NAMES)
    lon_name = _find_name(ds, LON_NAMES)
    target_times = pd.DatetimeIndex(request["datetimes"]).tz_localize(None)
    source_times = pd.to_datetime(ds[time_name].values).tz_localize(None)
    missing = [str(t) for t in target_times if t not in set(source_times)]
    if missing:
        raise ValueError(f"source fixture is missing selected timestamps: {missing[:5]}")
    ds = ds.sel({time_name: target_times.to_numpy(dtype="datetime64[ns]")})

    north, west, south, east = request["area"]
    lats = ds[lat_name].values
    if not (north == 90.0 and south == -90.0):
        lat_slice = slice(north, south) if lats[0] > lats[-1] else slice(south, north)
        ds = ds.sel({lat_name: lat_slice})
    lons = ds[lon_name].values
    if not ((west, east) in [(-180.0, 180.0), (0.0, 360.0)]):
        if float(np.nanmin(lons)) >= 0 and west < 0:
            shifted = (((ds[lon_name] + 180) % 360) - 180)
            ds = ds.assign_coords({lon_name: shifted}).sortby(lon_name)
        lon_slice = slice(west, east)
        ds = ds.sel({lon_name: lon_slice})
    if ds.sizes.get(lat_name, 0) == 0 or ds.sizes.get(lon_name, 0) == 0:
        raise ValueError("geographic filtering selected no grid cells")
    return ds


def _build_public_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    time_name = _find_name(ds, TIME_NAMES)
    lat_name = _find_name(ds, LAT_NAMES)
    lon_name = _find_name(ds, LON_NAMES)
    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds[time_name].values,
        "latitude": ds[lat_name].values,
        "longitude": ds[lon_name].values,
    }
    channels: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for spec in request["field_specs"]:
        source_var = _pick_var(ds, spec["name"])
        da = ds[source_var]
        plev_name = _find_name(da, PLEV_NAMES)
        source_level = _match_pressure_level(da[plev_name].values, spec["level"])
        da = da.sel({plev_name: [source_level]})
        field_id = _canonical_field_id(spec["field"])
        array_name = _unique_safe_name(field_id, used_names)
        selector_dim = f"pressure_level__{array_name}"
        da = da.rename({time_name: "time", lat_name: "latitude", lon_name: "longitude", plev_name: selector_dim})
        da = da.transpose("time", selector_dim, "latitude", "longitude")
        da = da.astype(da.dtype, copy=False)
        da.name = array_name
        da.attrs = dict(ds[source_var].attrs)
        meta = (inventory.get("option_metadata", {}).get("variable", {}).get(spec["name"], {}))
        if meta.get("units") and "units" not in da.attrs:
            da.attrs["units"] = meta["units"]
        if meta.get("description") and "description" not in da.attrs:
            da.attrs["description"] = meta["description"]
        da.attrs["canonical_field_id"] = field_id
        da.attrs["source_variable"] = source_var
        coords[selector_dim] = xr.DataArray(np.array([source_level], dtype=da[selector_dim].dtype), dims=(selector_dim,), attrs={"units": "hPa", "standard_name": "air_pressure"})
        data_vars[array_name] = da
        channels.append({
            "field_id": field_id,
            "array_path": array_name,
            "selectors": {"pressure_level": spec["level"]},
            "selector_coordinate_paths": {"pressure_level": selector_dim},
        })
    public = xr.Dataset(data_vars=data_vars, coords=coords)
    public["time"].attrs.update({"standard_name": "time", "timezone": "UTC"})
    public["latitude"].attrs.update(dict(ds[lat_name].attrs))
    public["longitude"].attrs.update(dict(ds[lon_name].attrs))
    public.attrs = dict(ds.attrs)
    public.attrs.update({
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "pipeline_id": PIPELINE_ID,
        "publication_format": "zarr_v3_consolidated",
        "contract_sha256": _stable_hash(contract),
    })
    return public, channels


def _write_zarr_v3(ds: xr.Dataset, store: Path) -> None:
    import zarr
    from zarr.codecs import BloscCodec

    encoding: dict[str, dict[str, Any]] = {}
    nlat = int(ds.sizes["latitude"])
    nlon = int(ds.sizes["longitude"])
    compressor = BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim.startswith("pressure_level__"):
                chunks.append(1)
            elif dim == "latitude":
                chunks.append(nlat)
            elif dim == "longitude":
                chunks.append(nlon)
            else:
                chunks.append(int(ds.sizes[dim]))
        encoding[name] = {"chunks": tuple(chunks), "compressors": [compressor]}
    for coord in ds.coords:
        encoding.setdefault(coord, {})["chunks"] = (int(ds.sizes[coord]),) if coord in ds.sizes else None
    clean = ds.copy(deep=False)
    for v in list(clean.data_vars) + list(clean.coords):
        clean[v].encoding = {}
    try:
        clean.to_zarr(store, mode="w", consolidated=True, zarr_format=3, encoding=encoding)
    except TypeError:
        clean.to_zarr(store, mode="w", consolidated=None, zarr_format=3, encoding=encoding)
        zarr.consolidate_metadata(str(store))


def _validate_publication(expected: xr.Dataset, store: Path, channels: list[dict[str, Any]]) -> None:
    reopened = xr.open_zarr(store, consolidated=True).load()
    if set(reopened.data_vars) != set(expected.data_vars):
        raise ValueError("published Zarr data variables do not match expected channels")
    if dict(reopened.sizes) != dict(expected.sizes):
        raise ValueError("published Zarr dimensions do not match expected dimensions")
    for coord in expected.coords:
        if coord not in reopened.coords:
            raise ValueError(f"published Zarr missing coordinate {coord}")
        np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
    for name in expected.data_vars:
        if reopened[name].dims != expected[name].dims:
            raise ValueError(f"published variable {name} dimension order changed")
        np.testing.assert_array_equal(reopened[name].values, expected[name].values)
        if reopened[name].attrs.get("canonical_field_id") != expected[name].attrs.get("canonical_field_id"):
            raise ValueError(f"published variable {name} lost canonical_field_id")
    # Sanity-check declared full-field tensor read path across all channels for the first sample.
    planes = []
    for ch in channels:
        arr = reopened[ch["array_path"]].isel(time=0)
        selector_dims = [d for d in arr.dims if d.startswith("pressure_level__")]
        if len(selector_dims) != 1 or arr.sizes[selector_dims[0]] != 1:
            raise ValueError("selector dimension was not preserved as a singleton dimension")
        plane = arr.isel({selector_dims[0]: 0}).values
        if plane.shape != (expected.sizes["latitude"], expected.sizes["longitude"]):
            raise ValueError("full-field tensor sample has incorrect H,W shape")
        planes.append(plane)
    if len(planes) != len(channels):
        raise ValueError("channel read-path sanity check failed")


def _find_name(obj: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str:
    names = set(obj.dims) | set(obj.coords) | (set(obj.data_vars) if isinstance(obj, xr.Dataset) else set())
    for c in candidates:
        if c in names:
            return c
    raise ValueError(f"none of the expected names are present: {candidates}")


def _pick_var(ds: xr.Dataset, canonical: str) -> str:
    for name in VAR_ALIASES.get(canonical, (canonical,)):
        if name in ds.data_vars:
            return name
    raise ValueError(f"source fixture missing requested variable {canonical}")


def _match_pressure_level(values: np.ndarray, requested: str) -> Any:
    req = float(requested)
    for v in values.tolist():
        try:
            if float(v) == req:
                return v
        except Exception:
            if str(v) == requested:
                return v
    raise ValueError(f"source fixture missing pressure_level={requested}")


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return str(field["name"])
    parts = [f"{s['dimension']}={json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':'))}" for s in selectors]
    return f"{field['name']}[" + ",".join(parts) + "]"


def _unique_safe_name(field_id: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", field_id).strip("_") or "channel"
    if re.match(r"^[0-9]", base):
        base = "v_" + base
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
