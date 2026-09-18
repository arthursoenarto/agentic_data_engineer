from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p006-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "dataset.zarr"

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

_TIME_NAMES = ("time", "valid_time")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_PRESSURE_NAMES = ("pressure_level", "isobaricInhPa", "level", "plev")


class PipelineError(ValueError):
    """Secret-safe pipeline validation or materialization error."""


@dataclass(frozen=True)
class SourceEntry:
    entry_id: str
    path: Path
    size_bytes: int
    sha256: str


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize an ERA5 pressure-level regular-grid fixture as consolidated Zarr v3.

    The function is intentionally offline-first. If cache_dir/source_fixture_manifest.json
    exists, all listed files are verified and used without constructing provider clients,
    checking credentials, or attempting network fallback.
    """
    contract = _extract_contract(contract_lock)
    _validate_inventory_contract(contract, inventory)
    request = _derive_request(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    entries = _load_and_verify_fixture(cache_root)
    if not entries:
        raise PipelineError("A verified source fixture manifest with at least one entry is required for offline execution")

    source_ds = _open_source_dataset([e.path for e in entries])
    try:
        source_ds = _normalize_dataset_coordinates(source_ds)
        output_ds, channels = _build_filtered_publication_dataset(source_ds, contract, inventory, request)
        _attach_publication_attrs(output_ds, contract, inventory, entries)

        final_store = out_root / STORE_NAME
        _publish_zarr_atomically(output_ds, final_store)
        _validate_published_store(final_store, output_ds, channels)
    finally:
        source_ds.close()

    cache_keys = [f"fixture:{_safe_token(e.entry_id)}:{e.sha256[:16]}" for e in entries]
    return {
        "cache": {
            "hits": len(entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": cache_keys,
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


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise PipelineError("contract_lock must be an object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract"):
        value = lock.get(key)
        if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
            return value
    nested = lock.get("lock")
    if isinstance(nested, dict):
        for key in ("contract", "dataset_contract", "selected_contract"):
            value = nested.get(key)
            if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
                return value
    contracts = lock.get("contracts")
    if isinstance(contracts, list):
        valid = [c for c in contracts if isinstance(c, dict) and c.get("schema_version") == "dataset_contract.v1"]
        if len(valid) == 1:
            return valid[0]
    raise PipelineError("Could not find a dataset_contract.v1 object in contract_lock")


def _validate_inventory_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("inventory schema_version must be dataset_inventory.v1")
    if contract.get("dataset_slug") != DATASET_SLUG or inventory.get("dataset_slug") != DATASET_SLUG:
        raise PipelineError("contract and inventory must target the ERA5 pressure-level dataset slug")
    if inventory.get("dataset_id") != DATASET_ID:
        raise PipelineError("inventory dataset_id does not match fixed policy")
    if contract.get("human_confirmed") is not True:
        raise PipelineError("contract must be human_confirmed")

    options = inventory.get("options") or {}
    advanced = contract.get("advanced_options") or {}
    dataset_id = advanced.get("dataset_id", DATASET_ID)
    if dataset_id != DATASET_ID:
        raise PipelineError("contract dataset_id does not match fixed policy")
    data_format = advanced.get("data_format", inventory.get("defaults", {}).get("data_format", "grib"))
    if data_format not in options.get("data_format", []):
        raise PipelineError("contract data_format is not offered by inventory")
    if data_format != "grib":
        raise PipelineError("fixed acquisition policy requires grib contracts")
    download_format = advanced.get("download_format", inventory.get("defaults", {}).get("download_format", "unarchived"))
    if download_format not in options.get("download_format", []):
        raise PipelineError("contract download_format is not offered by inventory")

    product_type = _as_list(advanced.get("product_type", (contract.get("scope") or {}).get("product_type", ["reanalysis"])))
    for item in product_type:
        if item not in options.get("product_type", []):
            raise PipelineError("contract product_type is not offered by inventory")
    if product_type != ["reanalysis"]:
        raise PipelineError("this fixed regular-grid policy supports ERA5 reanalysis product_type")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise PipelineError("contract must request at least one field")
    seen: set[str] = set()
    for field in fields:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise PipelineError(f"requested variable is not offered by inventory: {name}")
        for selector in field.get("selectors") or []:
            dim = selector.get("dimension")
            value = str(selector.get("value"))
            if dim not in options:
                raise PipelineError(f"selector dimension is not offered by inventory: {dim}")
            if value not in options[dim]:
                raise PipelineError(f"selector value is not offered by inventory: {dim}={value}")
        field_id = _canonical_field_id(field)
        if field_id in seen:
            raise PipelineError(f"duplicate requested field-selector channel: {field_id}")
        seen.add(field_id)

    scope = contract.get("scope") or {}
    times = ((scope.get("time") or {}).get("selected_times")) or []
    if not times:
        raise PipelineError("contract must select one or more UTC times")
    for t in times:
        if t not in options.get("time", []):
            raise PipelineError(f"selected time is not offered by inventory: {t}")
    if (scope.get("time") or {}).get("timezone", "UTC") != "UTC":
        raise PipelineError("only UTC selected_times are supported")

    area = ((scope.get("geography") or {}).get("cds_area")) or inventory.get("defaults", {}).get("area")
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)):
        raise PipelineError("scope.geography.cds_area must be a four-number north/west/south/east list")
    north, west, south, east = map(float, area)
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("scope geography area is outside supported latitude/longitude bounds")


def _derive_request(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    scope = contract.get("scope") or {}
    date_range = scope.get("date_range") or {}
    start = _parse_date(date_range.get("start_date"), "start_date")
    end = _parse_date(date_range.get("end_date"), "end_date")
    inclusive = bool(date_range.get("inclusive", True))
    if end < start:
        raise PipelineError("date_range end_date precedes start_date")

    selected_times = list((scope.get("time") or {}).get("selected_times") or [])
    requested = []
    d = start
    while d <= end if inclusive else d < end:
        for hhmm in selected_times:
            hour, minute = map(int, hhmm.split(":"))
            requested.append(np.datetime64(datetime.combine(d, time(hour, minute), tzinfo=timezone.utc).replace(tzinfo=None), "ns"))
        d += timedelta(days=1)
    if not requested:
        raise PipelineError("date_range and selected_times produce no samples")

    opts = inventory.get("options") or {}
    for ts in requested:
        dt = _npdt_to_datetime(ts)
        if f"{dt.year:04d}" not in opts.get("year", []):
            raise PipelineError(f"requested year is not available: {dt.year:04d}")
        if f"{dt.month:02d}" not in opts.get("month", []):
            raise PipelineError(f"requested month is not available: {dt.month:02d}")
        if f"{dt.day:02d}" not in opts.get("day", []):
            raise PipelineError(f"requested day is not available: {dt.day:02d}")

    area = ((scope.get("geography") or {}).get("cds_area")) or inventory.get("defaults", {}).get("area")
    return {"timestamps": np.array(requested, dtype="datetime64[ns]"), "area": [float(v) for v in area]}


def _load_and_verify_fixture(cache_root: Path) -> list[SourceEntry]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("source_fixture_manifest schema_version is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("source_fixture_manifest entries must be a non-empty list")

    verified: list[SourceEntry] = []
    for raw in entries:
        rel = raw.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
            raise PipelineError("fixture entry relative_path must stay beneath cache_dir")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineError("fixture entry path escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineError("fixture entry file is missing")
        expected_size = int(raw.get("size_bytes"))
        expected_sha = str(raw.get("sha256"))
        if expected_size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise PipelineError("fixture entry size_bytes or sha256 is invalid")
        data = path.read_bytes()
        actual_size = len(data)
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_size != expected_size or actual_sha != expected_sha:
            raise PipelineError("fixture entry failed size or sha256 verification")
        verified.append(SourceEntry(str(raw.get("entry_id", rel)), path, actual_size, actual_sha))
    return sorted(verified, key=lambda e: str(e.path))


def _open_source_dataset(paths: list[Path]) -> xr.Dataset:
    datasets = []
    errors = []
    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix in {".grib", ".grb", ".grib2", ".grb2"}:
                ds = xr.open_dataset(path, engine="cfgrib", decode_cf=True, mask_and_scale=True, backend_kwargs={"indexpath": ""})
            elif suffix == ".zarr":
                ds = xr.open_zarr(path, consolidated=None, mask_and_scale=True)
            else:
                ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
            datasets.append(ds)
        except Exception as exc:  # noqa: BLE001 - report sanitized aggregate after all attempts
            errors.append(f"{path.name}: {exc.__class__.__name__}")
    if not datasets:
        raise PipelineError("no verified fixture file could be decoded by xarray: " + "; ".join(errors))
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.merge(datasets, join="outer", compat="no_conflicts", combine_attrs="drop_conflicts")
    except Exception as exc:  # noqa: BLE001
        for ds in datasets:
            ds.close()
        raise PipelineError(f"verified fixture files could not be combined by coordinates: {exc.__class__.__name__}") from exc


def _normalize_dataset_coordinates(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for canonical, candidates in (("time", _TIME_NAMES), ("latitude", _LAT_NAMES), ("longitude", _LON_NAMES), ("pressure_level", _PRESSURE_NAMES)):
        if canonical in ds.dims or canonical in ds.coords:
            continue
        for c in candidates:
            if c in ds.dims or c in ds.coords:
                rename[c] = canonical
                break
    if rename:
        ds = ds.rename(rename)
    for required in ("time", "latitude", "longitude"):
        if required not in ds.coords and required not in ds.dims:
            raise PipelineError(f"source fixture lacks required coordinate: {required}")
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)
    return ds


def _build_filtered_publication_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    ds = _subset_time_and_area(ds, request["timestamps"], request["area"])
    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds["time"].copy(deep=True),
        "latitude": ds["latitude"].copy(deep=True),
        "longitude": ds["longitude"].copy(deep=True),
    }
    channels: list[dict[str, Any]] = []

    for field in contract["fields"]:
        field_id = _canonical_field_id(field)
        array_name = _array_name_for_field_id(field_id)
        source_name = _find_source_variable(ds, field["name"])
        da = ds[source_name]
        rename_dims: dict[str, str] = {}
        selector_paths: dict[str, str] = {}
        selectors: dict[str, str] = {}

        for selector in field.get("selectors") or []:
            dim = selector["dimension"]
            value = str(selector["value"])
            if dim in da.dims:
                coord_value = _coerce_selector_value(da[dim].values, value)
                da = da.sel({dim: [coord_value]})
            elif dim in da.coords:
                coord_value = _coerce_selector_value(da[dim].values, value)
                da = da.expand_dims({dim: [coord_value]})
            elif dim in ds.coords:
                coord_value = _coerce_selector_value(ds[dim].values, value)
                da = da.expand_dims({dim: [coord_value]})
            else:
                raise PipelineError(f"source variable {source_name} lacks selector dimension {dim}")
            specific_dim = f"{dim}__{array_name}"
            rename_dims[dim] = specific_dim
            selectors[dim] = value
            selector_paths[dim] = specific_dim
            coord = da[dim].copy(deep=True)
            coord.attrs.update({"selector_dimension": dim, "units": selector.get("unit") or inventory.get("option_units", {}).get(dim, "")})
            coords[specific_dim] = coord.rename({dim: specific_dim})

        if rename_dims:
            da = da.rename(rename_dims)
        wanted_dims = ["time"] + list(rename_dims.values()) + ["latitude", "longitude"]
        extra = [d for d in da.dims if d not in wanted_dims]
        if extra:
            raise PipelineError(f"source variable {source_name} has unsupported extra dimensions: {extra}")
        da = da.transpose(*[d for d in wanted_dims if d in da.dims])
        da = da.copy(deep=True)
        da.name = array_name
        da.attrs = _semantic_attrs(field, inventory, da.attrs, field_id)
        da.encoding = {}
        data_vars[array_name] = da
        channels.append({
            "field_id": field_id,
            "array_path": array_name,
            "selectors": selectors,
            "selector_coordinate_paths": selector_paths,
        })

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    return out, channels


def _subset_time_and_area(ds: xr.Dataset, timestamps: np.ndarray, area: list[float]) -> xr.Dataset:
    available = ds["time"].values.astype("datetime64[ns]")
    missing = [str(t) for t in timestamps if t not in available]
    if missing:
        raise PipelineError(f"source fixture is missing requested timestamps; first missing={missing[0]}")
    ds = ds.sel(time=timestamps)

    north, west, south, east = area
    lat = ds["latitude"]
    lat_mask = (lat >= south) & (lat <= north)
    if int(lat_mask.sum()) == 0:
        raise PipelineError("area filter selects no latitude values")
    ds = ds.isel(latitude=lat_mask)

    lon = ds["longitude"]
    if not _is_global_area(area):
        lon_values = lon.values
        if float(np.nanmin(lon_values)) >= 0 and west < 0:
            west_cmp = west % 360
            east_cmp = east % 360
            lon_cmp = lon
        else:
            west_cmp = west
            east_cmp = east
            lon_cmp = lon
        if west_cmp <= east_cmp:
            lon_mask = (lon_cmp >= west_cmp) & (lon_cmp <= east_cmp)
        else:
            lon_mask = (lon_cmp >= west_cmp) | (lon_cmp <= east_cmp)
        if int(lon_mask.sum()) == 0:
            raise PipelineError("area filter selects no longitude values")
        ds = ds.isel(longitude=lon_mask)
    return ds


def _publish_zarr_atomically(ds: xr.Dataset, final_store: Path) -> None:
    parent = final_store.parent
    tmp_store = Path(tempfile.mkdtemp(prefix=f".{STORE_NAME}.tmp-", dir=parent))
    try:
        encoding = _zarr_encoding(ds)
        try:
            ds.to_zarr(tmp_store, mode="w", consolidated=True, zarr_format=3, encoding=encoding)
        except TypeError:
            ds.to_zarr(tmp_store, mode="w", consolidated=True, zarr_version=3, encoding=encoding)
        # Validate staged store before exposing it.
        reopened = xr.open_zarr(tmp_store, consolidated=True)
        try:
            for name in ds.data_vars:
                if reopened[name].dims != ds[name].dims:
                    raise PipelineError(f"staged Zarr dimension mismatch for {name}")
        finally:
            reopened.close()
        backup = None
        if final_store.exists():
            backup = final_store.with_name(final_store.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(final_store, backup)
        os.replace(tmp_store, final_store)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if tmp_store.exists():
            shutil.rmtree(tmp_store, ignore_errors=True)
        raise


def _zarr_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    codec = _lossless_footprint_codec()
    enc: dict[str, dict[str, Any]] = {}
    lat_n = int(ds.sizes["latitude"])
    lon_n = int(ds.sizes["longitude"])
    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim in {"latitude", "longitude"}:
                chunks.append(lat_n if dim == "latitude" else lon_n)
            else:
                chunks.append(1)
        enc[name] = {
            "chunks": tuple(chunks),
            "compressors": [codec],
            "_FillValue": None,
        }
    for name, coord in ds.coords.items():
        enc[name] = {"chunks": tuple(int(coord.sizes[d]) for d in coord.dims), "compressors": [codec]}
    return enc


def _lossless_footprint_codec() -> Any:
    from numcodecs import Blosc

    shuffle = getattr(Blosc, "BITSHUFFLE", None)
    if shuffle is None:
        shuffle = getattr(Blosc, "SHUFFLE", 1)
    return Blosc(cname="zstd", clevel=7, shuffle=shuffle)


def _validate_published_store(store: Path, expected: xr.Dataset, channels: list[dict[str, Any]]) -> None:
    if not (store / "zarr.json").is_file():
        raise PipelineError("published store is not Zarr v3 or lacks root zarr.json")
    root_meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    if root_meta.get("zarr_format") != 3:
        raise PipelineError("published store is not Zarr format 3")
    if "consolidated_metadata" not in root_meta:
        raise PipelineError("published Zarr v3 metadata is not consolidated")

    reopened = xr.open_zarr(store, consolidated=True)
    try:
        if set(reopened.data_vars) != set(expected.data_vars):
            raise PipelineError("published data variable names differ from staged dataset")
        for coord in expected.coords:
            if coord not in reopened.coords:
                raise PipelineError(f"published coordinate missing: {coord}")
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
        for name in expected.data_vars:
            if reopened[name].dims != expected[name].dims:
                raise PipelineError(f"published dimension order mismatch for {name}")
            np.testing.assert_array_equal(reopened[name].values, expected[name].values)
            for attr_key in ("field_id", "units", "long_name"):
                if attr_key in expected[name].attrs and reopened[name].attrs.get(attr_key) != expected[name].attrs[attr_key]:
                    raise PipelineError(f"published semantic attribute mismatch for {name}.{attr_key}")
        if channels:
            arrays = [reopened[c["array_path"]].isel(time=0).values for c in channels]
            for arr in arrays:
                if arr.shape[-2:] != (expected.sizes["latitude"], expected.sizes["longitude"]):
                    raise PipelineError("full-field C,H,W-equivalent read shape is invalid")
    finally:
        reopened.close()


def _find_source_variable(ds: xr.Dataset, requested_name: str) -> str:
    candidates = [requested_name, ERA5_SHORT_NAMES.get(requested_name, "")]
    for c in candidates:
        if c and c in ds.data_vars:
            return c
    needle = requested_name.replace("_", " ").lower()
    for name, da in ds.data_vars.items():
        hay = " ".join(str(da.attrs.get(k, "")) for k in ("long_name", "standard_name", "GRIB_name", "cfVarName")).replace("_", " ").lower()
        if needle in hay:
            return name
    raise PipelineError(f"source fixture lacks requested variable: {requested_name}")


def _semantic_attrs(field: dict[str, Any], inventory: dict[str, Any], source_attrs: dict[str, Any], field_id: str) -> dict[str, Any]:
    meta = ((inventory.get("option_metadata") or {}).get("variable") or {}).get(field["name"], {})
    attrs = dict(source_attrs)
    attrs["field_id"] = field_id
    attrs["requested_name"] = field["name"]
    attrs["long_name"] = field.get("display_name") or meta.get("label") or attrs.get("long_name") or field["name"]
    units = field.get("units") or meta.get("units") or attrs.get("units")
    if units is not None:
        attrs["units"] = units
    if meta.get("description"):
        attrs.setdefault("description", meta["description"])
    return _json_safe_attrs(attrs)


def _attach_publication_attrs(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], entries: list[SourceEntry]) -> None:
    ds.attrs.update({
        "title": contract.get("title") or inventory.get("title") or DATASET_ID,
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "provider": "ECMWF",
        "publication_format": "zarr_v3_consolidated",
        "pipeline_id": PIPELINE_ID,
        "source_fixture_sha256": ",".join(e.sha256 for e in entries),
        "source_fixture_entry_count": len(entries),
    })


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get("selectors") or []
    if not selectors:
        return field["name"]
    parts = [f"{s['dimension']}={json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':'))}" for s in selectors]
    return f"{field['name']}[" + ",".join(parts) + "]"


def _array_name_for_field_id(field_id: str) -> str:
    return _safe_token(field_id).strip("_")[:180]


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    token = token.replace("..", "_")
    return token or "item"


def _coerce_selector_value(coord_values: np.ndarray, requested: str) -> Any:
    arr = np.atleast_1d(np.asarray(coord_values))
    if np.issubdtype(arr.dtype, np.integer):
        value: Any = int(requested)
    elif np.issubdtype(arr.dtype, np.floating):
        value = float(requested)
    else:
        value = requested
    if value not in set(arr.tolist()):
        raise PipelineError(f"source fixture lacks requested selector coordinate value: {requested}")
    return value


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise PipelineError(f"date_range.{field} must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _npdt_to_datetime(value: np.datetime64) -> datetime:
    ns = value.astype("datetime64[ns]").astype("int64")
    return datetime(1970, 1, 1) + timedelta(microseconds=ns / 1000)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _is_global_area(area: list[float]) -> bool:
    north, west, south, east = area
    return north >= 90 and south <= -90 and (abs(east - west) >= 360 or (west <= -180 and east >= 180))


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, np.generic):
            safe[k] = v.item()
        else:
            safe[k] = str(v)
    return safe
