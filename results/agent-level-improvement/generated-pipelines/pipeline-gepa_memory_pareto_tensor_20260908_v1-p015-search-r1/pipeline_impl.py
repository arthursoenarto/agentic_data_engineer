from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import zarr

STORE_NAME = "dataset.zarr"
TMP_STORE_NAME = ".tmp_dataset.zarr"
MANIFEST_NAME = "source_fixture_manifest.json"
EXPECTED_DATASET_SLUG = "reanalysis_era5_pressure_levels"
EXPECTED_DATASET_ID = "reanalysis-era5-pressure-levels"
EXPECTED_PROVIDER = "ECMWF"
FORBIDDEN_DATA_CODECS = ("blosc", "zstd", "lz4", "gzip", "zlib", "numcodecs", "shuffle", "bitshuffle")

FIELD_ALIASES = {
    "temperature": ("temperature", "t"),
    "geopotential": ("geopotential", "z"),
    "u_component_of_wind": ("u_component_of_wind", "u", "u_component"),
    "v_component_of_wind": ("v_component_of_wind", "v", "v_component"),
    "vertical_velocity": ("vertical_velocity", "w"),
    "relative_humidity": ("relative_humidity", "r"),
    "specific_humidity": ("specific_humidity", "q"),
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


def run_pipeline(contract_lock: Any, inventory: Mapping[str, Any], cache_dir: str, output_dir: str) -> Dict[str, Any]:
    """Materialize a validated ERA5 pressure-level contract from a verified local fixture."""
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = _verify_fixture_manifest(cache_root)
    contract = _extract_contract(contract_lock)
    _validate_inventory(inventory)
    request = _validate_contract(contract, inventory, contract_lock)

    source_paths = [cache_root / e["relative_path"] for e in manifest_entries]
    source = _open_fixture_dataset(source_paths)
    try:
        filtered, channels = _build_public_dataset(source, contract, inventory, request)
        final_store = out_root / STORE_NAME
        tmp_store = out_root / TMP_STORE_NAME
        _publish_zarr_atomically(filtered, channels, tmp_store, final_store)
        artifact = _artifact_declaration(channels)
    finally:
        source.close()

    return {
        "cache": {
            "hits": len(manifest_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [str(e.get("entry_id") or e["relative_path"]) for e in manifest_entries],
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [],
    }


def _extract_contract(lock: Any) -> Mapping[str, Any]:
    if not isinstance(lock, Mapping):
        raise ValueError("contract_lock must be a mapping or a lock envelope containing a mapping")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "request", "datasetContract"):
        val = lock.get(key)
        if isinstance(val, Mapping) and val.get("schema_version") == "dataset_contract.v1":
            return val
    inner = lock.get("contract_lock")
    if isinstance(inner, Mapping):
        return _extract_contract(inner)
    raise ValueError("could not locate dataset_contract.v1 inside contract_lock")


def _validate_inventory(inventory: Mapping[str, Any]) -> None:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ValueError("inventory schema_version must be dataset_inventory.v1")
    if inventory.get("dataset_slug") != EXPECTED_DATASET_SLUG:
        raise ValueError("inventory dataset_slug does not match this adapter")
    if inventory.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("inventory dataset_id does not match this adapter")


def _validate_contract(contract: Mapping[str, Any], inventory: Mapping[str, Any], lock: Mapping[str, Any]) -> Dict[str, Any]:
    if contract.get("schema_version") != "dataset_contract.v1":
        raise ValueError("contract schema_version must be dataset_contract.v1")
    if contract.get("dataset_slug") != EXPECTED_DATASET_SLUG:
        raise ValueError("contract dataset_slug does not match inventory")
    if contract.get("human_confirmed") is not True:
        raise ValueError("contract must be human_confirmed")

    options = inventory.get("options", {})
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("advanced_options.dataset_id must match frozen inventory")
    _require_option(adv.get("data_format"), options, "data_format")
    _require_option(adv.get("download_format"), options, "download_format")
    for pt in _as_list(adv.get("product_type", [contract.get("scope", {}).get("product_type")])):
        _require_option(pt, options, "product_type")

    _validate_embedded_policy(lock)

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("contract.fields must be a non-empty list")
    seen_field_ids = set()
    requested_levels: List[str] = []
    for field in fields:
        name = field.get("name")
        _require_option(name, options, "variable")
        selectors = field.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ValueError("ERA5 pressure-level fields require exactly one pressure_level selector")
        sel = selectors[0]
        value = str(sel.get("value"))
        if sel.get("unit") not in (None, "hPa"):
            raise ValueError("pressure_level selector unit must be hPa when supplied")
        _require_option(value, options, "pressure_level")
        requested_levels.append(value)
        field_id = _field_id(field)
        if field_id in seen_field_ids:
            raise ValueError(f"duplicate requested field selector: {field_id}")
        seen_field_ids.add(field_id)

    scope = contract.get("scope") or {}
    if scope.get("product_type") is not None:
        _require_option(scope.get("product_type"), options, "product_type")
    geo = scope.get("geography") or {}
    area = geo.get("cds_area") or geo.get("area")
    if geo.get("area") == "global":
        area = geo.get("cds_area", [90, -180, -90, 180])
    if not (isinstance(area, list) and len(area) == 4):
        raise ValueError("geography must include a four-value CDS area")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90):
        raise ValueError("invalid latitude bounds")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError("invalid longitude bounds")
    if geo.get("cds_area_order") not in (None, ["north", "west", "south", "east"]):
        raise ValueError("cds_area_order must be north/west/south/east")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone") not in (None, "UTC"):
        raise ValueError("only UTC time selections are supported")
    selected_times = time_scope.get("selected_times")
    if not isinstance(selected_times, list) or not selected_times:
        raise ValueError("scope.time.selected_times must be non-empty")
    for t in selected_times:
        _require_option(t, options, "time")

    date_range = scope.get("date_range") or {}
    target_times = _target_datetimes(date_range, selected_times, options)

    return {"area": [north, west, south, east], "target_times": target_times, "levels": requested_levels}


def _validate_embedded_policy(lock: Mapping[str, Any]) -> None:
    for key in ("fixed_pipeline_policy", "fixed_output_policy", "pipeline_policy", "policy"):
        pol = lock.get(key)
        if isinstance(pol, Mapping):
            if pol.get("provider") not in (None, EXPECTED_PROVIDER):
                raise ValueError("embedded policy provider mismatch")
            if pol.get("dataset_id") not in (None, EXPECTED_DATASET_ID):
                raise ValueError("embedded policy dataset_id mismatch")
            if pol.get("acquisition_format") not in (None, "grib"):
                raise ValueError("embedded policy acquisition_format mismatch")
            if pol.get("publication_format") not in (None, "zarr"):
                raise ValueError("embedded policy publication_format mismatch")
            zpol = pol.get("zarr") or {}
            if zpol and (zpol.get("format_version") != 3 or zpol.get("consolidated_metadata") is not True):
                raise ValueError("embedded Zarr policy must require consolidated Zarr v3")


def _require_option(value: Any, options: Mapping[str, Any], key: str) -> None:
    if value not in options.get(key, []):
        raise ValueError(f"invalid {key} option: {value!r}")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _target_datetimes(date_range: Mapping[str, Any], selected_times: Sequence[str], options: Mapping[str, Any]) -> pd.DatetimeIndex:
    start = date_range.get("start_date")
    end = date_range.get("end_date")
    if not start or not end:
        raise ValueError("date_range requires start_date and end_date")
    inclusive = bool(date_range.get("inclusive", True))
    start_day = pd.Timestamp(str(start)).tz_localize(None).normalize()
    end_day = pd.Timestamp(str(end)).tz_localize(None).normalize()
    if end_day < start_day:
        raise ValueError("date_range end_date precedes start_date")
    days = pd.date_range(start_day, end_day if inclusive else end_day - pd.Timedelta(days=1), freq="D")
    stamps = []
    for day in days:
        y, m, d = f"{day.year:04d}", f"{day.month:02d}", f"{day.day:02d}"
        _require_option(y, options, "year")
        _require_option(m, options, "month")
        _require_option(d, options, "day")
        for hhmm in selected_times:
            hour, minute = [int(x) for x in hhmm.split(":")]
            stamps.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(stamps)


def _verify_fixture_manifest(cache_root: Path) -> List[Mapping[str, Any]]:
    manifest_path = cache_root / MANIFEST_NAME
    if not manifest_path.exists():
        raise RuntimeError("no complete local source fixture manifest is present; remote acquisition is disabled")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ValueError("invalid source fixture manifest schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source fixture manifest must contain entries")
    verified = []
    for entry in entries:
        rel = entry.get("relative_path")
        sha = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError("manifest relative_path must stay beneath cache_dir")
        if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise ValueError("manifest sha256 must be lowercase hex")
        if not (isinstance(size, int) and size > 0):
            raise ValueError("manifest size_bytes must be a positive integer")
        path = (cache_root / rel).resolve()
        if cache_root not in path.parents and path != cache_root:
            raise ValueError("manifest path escapes cache_dir")
        if not path.is_file():
            raise FileNotFoundError("manifest-listed source file is missing")
        stat = path.stat()
        if stat.st_size != size:
            raise ValueError("manifest-listed source file size mismatch")
        if _sha256(path) != sha:
            raise ValueError("manifest-listed source file sha256 mismatch")
        verified.append(entry)
    return verified


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _open_fixture_dataset(paths: Sequence[Path]) -> xr.Dataset:
    datasets = []
    try:
        for path in paths:
            ds = _open_one_dataset(path)
            ds = _normalize_coord_names(ds)
            ds.load()
            datasets.append(ds)
        if len(datasets) == 1:
            out = datasets[0]
        else:
            out = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts", compat="override", join="outer")
            out.load()
            for ds in datasets:
                if ds is not out:
                    ds.close()
        if "time" in out.coords:
            out = out.sortby("time")
        return out
    except Exception:
        for ds in datasets:
            ds.close()
        raise


def _open_one_dataset(path: Path) -> xr.Dataset:
    suffix = path.suffix.lower()
    errors: List[str] = []
    engines = ["cfgrib"] if suffix in (".grib", ".grb", ".grib2") else ["h5netcdf", "netcdf4", None]
    if suffix in (".nc", ".nc4", ".cdf"):
        engines = ["h5netcdf", "netcdf4", None]
    for engine in engines:
        try:
            if engine == "cfgrib":
                return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}, mask_and_scale=True, decode_times=True)
            if engine is None:
                return xr.open_dataset(path, mask_and_scale=True, decode_times=True)
            return xr.open_dataset(path, engine=engine, mask_and_scale=True, decode_times=True)
        except Exception as exc:  # keep secret-safe diagnostics only
            errors.append(f"{engine or 'default'}:{exc.__class__.__name__}")
    raise RuntimeError("could not decode fixture source with supported local engines: " + ",".join(errors))


def _normalize_coord_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for cand in ("valid_time", "timestamp"):
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "time"
            break
    for cand in ("lat", "Latitude"):
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "latitude"
            break
    for cand in ("lon", "Longitude"):
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "longitude"
            break
    for cand in ("isobaricInhPa", "isobaricInPa", "level", "plev"):
        if cand in ds.coords or cand in ds.dims:
            rename[cand] = "pressure_level"
            break
    rename = {k: v for k, v in rename.items() if k != v and v not in ds}
    ds = ds.rename(rename) if rename else ds
    if "pressure_level" in ds.coords and "pressure_level" not in ds.dims and ds["pressure_level"].ndim == 0:
        level_value = ds["pressure_level"].item()
        level_attrs = dict(ds["pressure_level"].attrs)
        ds = ds.drop_vars("pressure_level").expand_dims({"pressure_level": [level_value]})
        ds["pressure_level"].attrs.update(level_attrs)
        ds["pressure_level"].attrs.setdefault("units", "hPa")
    return ds


def _build_public_dataset(source: xr.Dataset, contract: Mapping[str, Any], inventory: Mapping[str, Any], request: Mapping[str, Any]) -> Tuple[xr.Dataset, List[Dict[str, Any]]]:
    _require_coords(source, ["time", "latitude", "longitude"])
    source = _filter_time(source, request["target_times"])
    source = _filter_area(source, request["area"])

    data_vars: Dict[str, xr.DataArray] = {}
    coords: Dict[str, Any] = {
        "time": source["time"].copy(deep=True),
        "latitude": source["latitude"].copy(deep=True),
        "longitude": source["longitude"].copy(deep=True),
    }
    for c in coords.values():
        c.encoding = {}

    channels: List[Dict[str, Any]] = []
    for field in contract["fields"]:
        name = field["name"]
        selector = field["selectors"][0]
        level = str(selector["value"])
        src_name = _find_source_var(source, name)
        arr = source[src_name]
        arr = _select_pressure(arr, level)
        arr = _drop_unrequested_singletons(arr, required={"time", "latitude", "longitude"})
        if set(arr.dims) != {"time", "latitude", "longitude"}:
            raise ValueError(f"source variable {src_name!r} has unsupported dimensions after selection: {arr.dims}")
        arr = arr.transpose("time", "latitude", "longitude")
        pressure_dim = _safe_name(f"pressure_level__{_field_id(field)}")
        public_name = _safe_name(_field_id(field))
        coords[pressure_dim] = xr.DataArray(np.array([_level_value_like_source(source, level)]), dims=(pressure_dim,), attrs={"units": "hPa", "long_name": "pressure_level"})
        out_arr = arr.expand_dims({pressure_dim: coords[pressure_dim]}).transpose("time", pressure_dim, "latitude", "longitude")
        out_arr.name = public_name
        out_arr.attrs = _safe_attrs(arr.attrs)
        inv_meta = ((inventory.get("option_metadata") or {}).get("variable") or {}).get(name) or {}
        if "units" not in out_arr.attrs and inv_meta.get("units"):
            out_arr.attrs["units"] = inv_meta["units"]
        if "long_name" not in out_arr.attrs and inv_meta.get("label"):
            out_arr.attrs["long_name"] = inv_meta["label"]
        out_arr.encoding = {}
        data_vars[public_name] = out_arr
        channels.append({
            "field_id": _field_id(field),
            "array_path": public_name,
            "selectors": {"pressure_level": level},
            "selector_coordinate_paths": {"pressure_level": pressure_dim},
            "_chunks": (1, 1, int(source.sizes["latitude"]), int(source.sizes["longitude"])),
        })

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={"dataset_slug": EXPECTED_DATASET_SLUG, "dataset_id": EXPECTED_DATASET_ID})
    for var in ds.variables:
        ds[var].encoding = {}
    return ds, channels


def _require_coords(ds: xr.Dataset, names: Iterable[str]) -> None:
    missing = [n for n in names if n not in ds.coords and n not in ds.dims]
    if missing:
        raise ValueError("source fixture is missing required coordinates: " + ",".join(missing))


def _filter_time(ds: xr.Dataset, target: pd.DatetimeIndex) -> xr.Dataset:
    src_times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).tz_localize(None)
    target = pd.DatetimeIndex(target).tz_localize(None)
    missing = target.difference(src_times)
    if len(missing):
        raise ValueError("source fixture does not contain all selected UTC timestamps")
    return ds.sel(time=target.to_numpy(dtype="datetime64[ns]"))


def _filter_area(ds: xr.Dataset, area: Sequence[float]) -> xr.Dataset:
    north, west, south, east = area
    lat = ds["latitude"]
    lon = ds["longitude"]
    lat_mask = (lat >= south) & (lat <= north)
    if not bool(lat_mask.any()):
        raise ValueError("selected latitude bounds contain no source cells")
    if west == -180 and east == 180:
        lon_mask = xr.ones_like(lon, dtype=bool)
    else:
        lon_vals = lon.values.astype(float)
        west360 = west % 360
        east360 = east % 360
        lon360 = lon_vals % 360
        if west360 <= east360:
            mask_np = (lon360 >= west360) & (lon360 <= east360)
        else:
            mask_np = (lon360 >= west360) | (lon360 <= east360)
        lon_mask = xr.DataArray(mask_np, dims=lon.dims, coords=lon.coords)
    if not bool(lon_mask.any()):
        raise ValueError("selected longitude bounds contain no source cells")
    return ds.where(lat_mask & lon_mask, drop=True)


def _find_source_var(ds: xr.Dataset, requested: str) -> str:
    aliases = FIELD_ALIASES.get(requested, (requested,))
    for alias in aliases:
        if alias in ds.data_vars:
            return alias
    for var in ds.data_vars:
        attrs = ds[var].attrs
        candidates = {str(attrs.get("standard_name", "")), str(attrs.get("long_name", "")), str(attrs.get("GRIB_cfName", "")), str(attrs.get("GRIB_shortName", ""))}
        if requested in candidates or any(a in candidates for a in aliases):
            return var
    raise ValueError(f"source fixture does not contain requested variable {requested!r}")


def _select_pressure(arr: xr.DataArray, level: str) -> xr.DataArray:
    pcoord = None
    for cand in ("pressure_level", "isobaricInhPa", "level", "plev"):
        if cand in arr.coords or cand in arr.dims:
            pcoord = cand
            break
    if pcoord is None:
        raise ValueError("requested pressure-level selector but source variable has no pressure coordinate")
    coord = arr[pcoord]
    vals = coord.values
    target_num = float(level)
    if np.issubdtype(np.asarray(vals).dtype, np.number):
        matches = np.where(np.isclose(vals.astype(float), target_num, rtol=0, atol=1e-9))[0]
    else:
        matches = np.where(np.asarray([str(v) == level for v in vals]))[0]
    if len(matches) != 1:
        raise ValueError(f"source variable does not contain exactly one pressure level {level}")
    selected = arr.isel({pcoord: int(matches[0])})
    if pcoord in selected.coords and pcoord not in selected.dims:
        selected = selected.drop_vars(pcoord)
    return selected


def _level_value_like_source(ds: xr.Dataset, level: str) -> Any:
    if "pressure_level" in ds.coords and np.issubdtype(np.asarray(ds["pressure_level"].values).dtype, np.integer):
        return np.int32(int(level))
    if "pressure_level" in ds.coords and np.issubdtype(np.asarray(ds["pressure_level"].values).dtype, np.floating):
        return np.float32(float(level))
    return np.int32(int(level))


def _drop_unrequested_singletons(arr: xr.DataArray, required: set[str]) -> xr.DataArray:
    for dim in list(arr.dims):
        if dim not in required and arr.sizes[dim] == 1:
            arr = arr.isel({dim: 0})
    return arr


def _publish_zarr_atomically(ds: xr.Dataset, channels: Sequence[Mapping[str, Any]], tmp_store: Path, final_store: Path) -> None:
    if tmp_store.exists():
        shutil.rmtree(tmp_store)
    tmp_store.parent.mkdir(parents=True, exist_ok=True)
    encoding: Dict[str, Dict[str, Any]] = {}
    channel_by_array = {c["array_path"]: c for c in channels}
    for name, var in ds.variables.items():
        enc: Dict[str, Any] = {}
        if name in channel_by_array:
            enc["chunks"] = tuple(channel_by_array[name]["_chunks"])
            enc["compressors"] = None
            enc["compressor"] = None
            enc["filters"] = None
        elif ds[name].dims:
            enc["chunks"] = tuple(int(ds.sizes[d]) for d in ds[name].dims)
            enc["compressors"] = None
            enc["compressor"] = None
            enc["filters"] = None
        encoding[name] = enc
    try:
        ds.to_zarr(tmp_store, mode="w", consolidated=True, zarr_format=3, encoding=encoding)
    except TypeError:
        for enc in encoding.values():
            enc.pop("compressors", None)
        ds.to_zarr(tmp_store, mode="w", consolidated=True, zarr_format=3, encoding=encoding)

    _validate_written_store(tmp_store, ds, channels)
    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)
    _validate_written_store(final_store, ds, channels)


def _validate_written_store(store: Path, expected: xr.Dataset, channels: Sequence[Mapping[str, Any]]) -> None:
    if not (store / "zarr.json").exists():
        raise RuntimeError("published store is not Zarr v3")
    root_meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    if root_meta.get("zarr_format") != 3:
        raise RuntimeError("published store is not Zarr format 3")
    # Consolidated metadata is stored in the v3 root metadata under consolidated_metadata.
    if "consolidated_metadata" not in root_meta:
        raise RuntimeError("published Zarr v3 metadata is not consolidated")

    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        if dict(reopened.sizes) != dict(expected.sizes):
            raise RuntimeError("reopened dataset dimensions differ from expected")
        for coord in expected.coords:
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
            for k, v in expected[coord].attrs.items():
                if reopened[coord].attrs.get(k) != v:
                    raise RuntimeError(f"coordinate attribute mismatch for {coord}.{k}")
        for name in expected.data_vars:
            np.testing.assert_equal(reopened[name].values, expected[name].values)
            for k, v in expected[name].attrs.items():
                if reopened[name].attrs.get(k) != v:
                    raise RuntimeError(f"variable attribute mismatch for {name}.{k}")
    finally:
        reopened.close()

    for ch in channels:
        arr = zarr.open(str(store / ch["array_path"]), mode="r")
        chunks = tuple(int(x) for x in arr.chunks)
        if chunks != tuple(ch["_chunks"]):
            raise RuntimeError(f"data array {ch['array_path']} chunks {chunks} do not match expected {ch['_chunks']}")
        codec_text = _codec_text(arr)
        if any(bad in codec_text for bad in FORBIDDEN_DATA_CODECS):
            raise RuntimeError(f"data array {ch['array_path']} has forbidden compression/filter codec: {codec_text}")


def _codec_text(arr: Any) -> str:
    meta = getattr(arr, "metadata", None)
    codecs = getattr(meta, "codecs", None)
    if codecs is None:
        codecs = getattr(arr, "codecs", None)
    return " ".join(type(c).__name__.lower() + " " + repr(c).lower() for c in (codecs or []))


def _artifact_declaration(channels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    public_channels = []
    for ch in channels:
        public_channels.append({
            "field_id": ch["field_id"],
            "array_path": ch["array_path"],
            "selectors": dict(ch["selectors"]),
            "selector_coordinate_paths": dict(ch["selector_coordinate_paths"]),
        })
    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": STORE_NAME,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": public_channels,
    }


def _field_id(field: Mapping[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return str(field["name"])
    parts = []
    for sel in selectors:
        parts.append(f"{sel['dimension']}={json.dumps(str(sel['value']), separators=(',', ':'))}")
    return f"{field['name']}[" + ",".join(parts) + "]"


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()
    if not text or text[0].isdigit():
        text = "v_" + text
    return text


def _safe_attrs(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in attrs.items():
        if not isinstance(k, str):
            continue
        if k.lower() in {"source", "history", "references", "path", "filename", "file", "tracking_id"}:
            continue
        val = _json_scalar(v)
        if val is None:
            continue
        if isinstance(val, str) and ("://" in val or re.search(r"(^|\s)/[^\s]+", val)):
            continue
        out[k] = val
    return out


def _json_scalar(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        vals = [_json_scalar(x) for x in v]
        return vals if all(x is not None for x in vals) else None
    return None
