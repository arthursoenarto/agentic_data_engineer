"""Deterministic ERA5 pressure-level fixture-to-Zarr adapter.

The public entry point is ``run_pipeline(contract_lock, inventory, cache_dir,
output_dir)``.  The implementation is intentionally offline-only for this pilot:
it verifies and consumes a complete ``source_fixture_manifest.json`` under
``cache_dir`` and never probes CDS or any other external service.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


FIELD_SHORT_NAMES = {
    "t": "temperature",
    "z": "geopotential",
}
COORD_ALIASES = {
    "valid_time": "time",
    "isobaricInhPa": "pressure_level",
    "level": "pressure_level",
    "plev": "pressure_level",
    "lat": "latitude",
    "lon": "longitude",
}
ALLOWED_ADVANCED = {"data_format", "dataset_id", "download_format", "product_type"}
STORE_NAME = "dataset.zarr"


class PipelineError(ValueError):
    """Secret-safe validation or publication failure."""


@dataclass(frozen=True)
class FixtureEntry:
    entry_id: str
    path: Path
    size_bytes: int
    sha256: str

    @property
    def safe_key(self) -> str:
        return f"{self.entry_id}:{self.sha256[:16]}"


@dataclass(frozen=True)
class ChannelSpec:
    field_id: str
    variable: str
    selector_dimension: str | None
    selector_value_text: str | None


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a runtime contract, filter verified fixture data, and publish Zarr v3.

    Parameters are supplied by the framework.  ``contract_lock`` may be either the
    dataset contract itself or a lock envelope containing it.  ``cache_dir`` is
    treated as an external read-only fixture root when it contains
    ``source_fixture_manifest.json``; this adapter requires that fixture and has
    no network fallback.
    """
    contract = _find_contract(contract_lock)
    _validate_contract_against_inventory(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    fixture_entries = _load_and_verify_fixture(cache_root)

    source = _open_verified_sources([entry.path for entry in fixture_entries])
    source = _standardize_dataset(source)

    channels = _channel_specs(contract)
    selected = _filter_and_group(source, contract, inventory, channels)

    out_root.mkdir(parents=True, exist_ok=True)
    store_path = out_root / STORE_NAME
    codec_mode = _publish_zarr_atomic(selected, store_path)
    _validate_publication(selected, store_path, codec_mode)

    artifact = _artifact_layout(channels, selected, STORE_NAME)
    _sanity_read_full_tensor_sample(store_path, artifact)

    return {
        "cache": {
            "hits": len(fixture_entries),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [entry.safe_key for entry in fixture_entries],
            "acquired_keys": [],
        },
        "dataset_artifact": artifact,
        "warnings": [] if codec_mode == "uncompressed" else ["Uncompressed Zarr v3 chunks were not accepted by the installed stack; used lossless Blosc-LZ4 clevel=1 byte-shuffle fallback."],
    }


def _find_contract(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict) and obj.get("schema_version") == "dataset_contract.v1":
        return obj
    if isinstance(obj, dict):
        for key in ("contract", "dataset_contract", "selected_contract", "request_contract"):
            value = obj.get(key)
            if isinstance(value, dict):
                try:
                    return _find_contract(value)
                except PipelineError:
                    pass
        for value in obj.values():
            if isinstance(value, (dict, list)):
                try:
                    return _find_contract(value)
                except PipelineError:
                    pass
    if isinstance(obj, list):
        for value in obj:
            try:
                return _find_contract(value)
            except PipelineError:
                pass
    raise PipelineError("No dataset_contract.v1 object found in contract lock envelope")


def _validate_contract_against_inventory(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if contract.get("schema_version") != "dataset_contract.v1":
        raise PipelineError("Unsupported contract schema")
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise PipelineError("Unsupported inventory schema")
    if contract.get("dataset_slug") != inventory.get("dataset_slug"):
        raise PipelineError("Contract dataset_slug does not match inventory")
    if contract.get("advanced_options", {}).get("dataset_id") not in (None, inventory.get("dataset_id")):
        raise PipelineError("advanced_options.dataset_id is not allowed by inventory")

    options = inventory.get("options", {})
    fields = contract.get("fields") or []
    if not fields:
        raise PipelineError("Contract must request at least one field")
    seen: set[str] = set()
    for field in fields:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise PipelineError(f"Requested variable is not in inventory options: {name}")
        selectors = field.get("selectors") or []
        for selector in selectors:
            dim = selector.get("dimension")
            value = str(selector.get("value"))
            if dim != "pressure_level":
                raise PipelineError(f"Unsupported selector dimension for this inventory: {dim}")
            if value not in options.get("pressure_level", []):
                raise PipelineError(f"Requested pressure level is not in inventory options: {value}")
            if selector.get("unit") not in (None, "hPa"):
                raise PipelineError("pressure_level selector unit must be hPa when supplied")
        fid = _canonical_field_id(field)
        if fid in seen:
            raise PipelineError(f"Duplicate requested field-selector channel: {fid}")
        seen.add(fid)

    scope = contract.get("scope") or {}
    dr = scope.get("date_range") or {}
    start = _parse_date(dr.get("start_date"), "start_date")
    end = _parse_date(dr.get("end_date"), "end_date")
    if end < start:
        raise PipelineError("date_range.end_date is before start_date")
    for y in range(start.year, end.year + 1):
        if f"{y:04d}" not in options.get("year", []):
            raise PipelineError(f"Requested year is not in inventory options: {y:04d}")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone") not in (None, "UTC"):
        raise PipelineError("Only UTC selected_times are supported for this inventory")
    selected_times = time_scope.get("selected_times") or []
    if not selected_times:
        raise PipelineError("scope.time.selected_times must be non-empty")
    for t in selected_times:
        if t not in options.get("time", []):
            raise PipelineError(f"Requested time is not in inventory options: {t}")
        _parse_hhmm(t)

    product_type = scope.get("product_type")
    if product_type is not None and product_type not in options.get("product_type", []):
        raise PipelineError("scope.product_type is not allowed by inventory")

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area"))
    _validate_area(area)

    advanced = contract.get("advanced_options") or {}
    unknown = set(advanced) - ALLOWED_ADVANCED
    if unknown:
        raise PipelineError(f"Unsupported advanced option(s): {sorted(unknown)}")
    for key in ("data_format", "download_format"):
        value = advanced.get(key, inventory.get("defaults", {}).get(key))
        if value not in options.get(key, []):
            raise PipelineError(f"advanced_options.{key} is not allowed by inventory")
    adv_product = advanced.get("product_type")
    if adv_product is not None:
        values = adv_product if isinstance(adv_product, list) else [adv_product]
        for value in values:
            if value not in options.get("product_type", []):
                raise PipelineError("advanced_options.product_type contains a value not allowed by inventory")


def _parse_date(value: Any, name: str) -> date:
    if not isinstance(value, str):
        raise PipelineError(f"date_range.{name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise PipelineError(f"date_range.{name} is not a valid ISO date") from exc


def _parse_hhmm(value: str) -> tuple[int, int]:
    try:
        dt = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise PipelineError(f"Invalid selected time: {value}") from exc
    return dt.hour, dt.minute


def _validate_area(area: Any) -> None:
    if not isinstance(area, list) or len(area) != 4:
        raise PipelineError("geography.cds_area must contain four values [north, west, south, east]")
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90):
        raise PipelineError("geography latitude bounds are invalid")
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError("geography longitude bounds are invalid")


def _load_and_verify_fixture(cache_root: Path) -> list[FixtureEntry]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise PipelineError("A complete source_fixture_manifest.json is required; network fallback is disabled")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError("source_fixture_manifest.json is not valid JSON") from exc
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise PipelineError("Unsupported source fixture manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("source fixture manifest must contain at least one entry")

    verified: list[FixtureEntry] = []
    for raw in entries:
        rel = raw.get("relative_path")
        if not isinstance(rel, str) or Path(rel).is_absolute():
            raise PipelineError("Fixture entry relative_path must be a relative path")
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise PipelineError("Fixture entry path escapes cache_dir") from exc
        if not path.is_file():
            raise PipelineError("Fixture entry file is missing")
        size = int(raw.get("size_bytes"))
        if size <= 0 or path.stat().st_size != size:
            raise PipelineError("Fixture entry size check failed")
        sha = str(raw.get("sha256", ""))
        if len(sha) != 64 or sha.lower() != sha:
            raise PipelineError("Fixture entry sha256 must be a lowercase SHA-256 digest")
        digest = _sha256(path)
        if digest != sha:
            raise PipelineError("Fixture entry SHA-256 check failed")
        entry_id = str(raw.get("entry_id") or rel)
        verified.append(FixtureEntry(entry_id=entry_id, path=path, size_bytes=size, sha256=sha))
    verified.sort(key=lambda e: e.entry_id)
    return verified


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_verified_sources(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    for path in paths:
        opened = None
        engines: list[str | None]
        suffix = path.suffix.lower()
        if suffix in {".grib", ".grb", ".grb2"}:
            engines = ["cfgrib", None, "h5netcdf", "netcdf4", "scipy"]
        else:
            engines = [None, "h5netcdf", "netcdf4", "scipy", "cfgrib"]
        for engine in engines:
            try:
                kwargs: dict[str, Any] = {"decode_cf": True, "mask_and_scale": True}
                if engine is not None:
                    kwargs["engine"] = engine
                if engine == "cfgrib":
                    import cfgrib
                    cf_datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
                    try:
                        loaded = [_promote_scalar_selector_coords(ds.load()) for ds in cf_datasets]
                    finally:
                        for ds in cf_datasets:
                            ds.close()
                    if len(loaded) == 1:
                        opened = loaded[0]
                    else:
                        try:
                            opened = xr.merge(loaded, compat="no_conflicts", combine_attrs="drop_conflicts")
                        except Exception:
                            opened = xr.combine_by_coords(loaded, combine_attrs="drop_conflicts")
                else:
                    ds = xr.open_dataset(path, **kwargs)
                    opened = _promote_scalar_selector_coords(ds.load())
                    ds.close()
                break
            except Exception as exc:  # keep trying engines; summarize secret-safely later
                errors.append(f"{path.name}:{engine or 'auto'}:{exc.__class__.__name__}")
        if opened is None:
            raise PipelineError(f"Unable to decode verified source fixture file(s): {errors[:8]}")
        datasets.append(opened)
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts")
    except Exception:
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")


def _standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for name in list(ds.dims) + list(ds.coords):
        if name in COORD_ALIASES and COORD_ALIASES[name] not in ds:
            rename[name] = COORD_ALIASES[name]
    for name in ds.data_vars:
        if name in FIELD_SHORT_NAMES and FIELD_SHORT_NAMES[name] not in ds:
            rename[name] = FIELD_SHORT_NAMES[name]
    if rename:
        ds = ds.rename(rename)
    for required in ("time", "latitude", "longitude"):
        if required not in ds.coords and required not in ds.dims:
            raise PipelineError(f"Source fixture lacks required coordinate: {required}")
    return ds


def _promote_scalar_selector_coords(ds: xr.Dataset) -> xr.Dataset:
    """Make single-level decoded pressure coordinates mergeable across source hypercubes."""
    rename: dict[str, str] = {}
    for name in list(ds.coords):
        target = COORD_ALIASES.get(name)
        if target == "pressure_level" and target not in ds:
            rename[name] = target
    if rename:
        ds = ds.rename(rename)
    for coord in ("pressure_level", "isobaricInhPa", "level", "plev"):
        if coord in ds.coords and coord not in ds.dims and np.ndim(ds[coord].values) == 0:
            value = ds[coord].values.item()
            return ds.expand_dims({coord: [value]})
    return ds


def _channel_specs(contract: dict[str, Any]) -> list[ChannelSpec]:
    specs: list[ChannelSpec] = []
    for field in contract.get("fields") or []:
        selectors = field.get("selectors") or []
        if selectors:
            selector = selectors[0]
            specs.append(ChannelSpec(
                field_id=_canonical_field_id(field),
                variable=str(field["name"]),
                selector_dimension=str(selector["dimension"]),
                selector_value_text=str(selector["value"]),
            ))
        else:
            specs.append(ChannelSpec(field_id=_canonical_field_id(field), variable=str(field["name"]), selector_dimension=None, selector_value_text=None))
    return specs


def _canonical_field_id(field: dict[str, Any]) -> str:
    name = str(field.get("name"))
    selectors = field.get("selectors") or []
    if not selectors:
        return name
    parts = []
    for selector in selectors:
        parts.append(f"{selector['dimension']}={json.dumps(str(selector['value']), ensure_ascii=False, sort_keys=True)}")
    return f"{name}[{','.join(parts)}]"


def _requested_timestamps(contract: dict[str, Any]) -> pd.DatetimeIndex:
    scope = contract["scope"]
    dr = scope["date_range"]
    start = _parse_date(dr["start_date"], "start_date")
    end = _parse_date(dr["end_date"], "end_date")
    inclusive = bool(dr.get("inclusive", True))
    if not inclusive:
        end = end - timedelta(days=1)
    selected_times = scope.get("time", {}).get("selected_times") or []
    values: list[pd.Timestamp] = []
    current = start
    while current <= end:
        for t in selected_times:
            hour, minute = _parse_hhmm(t)
            values.append(pd.Timestamp(datetime(current.year, current.month, current.day, hour, minute, tzinfo=timezone.utc)).tz_convert(None))
        current += timedelta(days=1)
    return pd.DatetimeIndex(values)


def _filter_and_group(source: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], channels: list[ChannelSpec]) -> xr.Dataset:
    timestamps = _requested_timestamps(contract)
    time_values = pd.to_datetime(source["time"].values)
    missing_times = [str(t) for t in timestamps if t not in set(time_values)]
    if missing_times:
        raise PipelineError(f"Source fixture does not contain requested timestamps, first missing: {missing_times[0]}")
    ds_time = source.sel(time=timestamps.to_numpy())
    ds_geo = _filter_geography(ds_time, contract)

    by_var: dict[str, list[ChannelSpec]] = {}
    for spec in channels:
        by_var.setdefault(spec.variable, []).append(spec)

    out_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": ds_geo["time"].copy(deep=True),
        "latitude": ds_geo["latitude"].copy(deep=True),
        "longitude": ds_geo["longitude"].copy(deep=True),
    }
    for coord_name in coords:
        coords[coord_name].encoding.clear()

    option_meta = inventory.get("option_metadata", {}).get("variable", {})
    for variable in sorted(by_var, key=lambda v: [c.variable for c in channels].index(v)):
        if variable not in ds_geo.data_vars:
            raise PipelineError(f"Source fixture lacks requested variable: {variable}")
        specs = by_var[variable]
        arr = ds_geo[variable]
        arr = _ensure_data_dims(arr)
        for enc_key in ("scale_factor", "add_offset", "_FillValue", "missing_value", "dtype"):
            arr.encoding.pop(enc_key, None)

        selector_specs = [s for s in specs if s.selector_dimension]
        if selector_specs:
            selector_dim = selector_specs[0].selector_dimension
            if any(s.selector_dimension != selector_dim for s in selector_specs):
                raise PipelineError("A variable cannot mix selector dimensions in one grouped array")
            requested_text = _unique_in_order([s.selector_value_text or "" for s in selector_specs])
            dim_name = f"{variable}_{selector_dim}"
            try:
                native_dim = _find_selector_dim(arr, selector_dim or "")
            except PipelineError:
                scalar_coord = next((candidate for candidate in (selector_dim or "", "isobaricInhPa", "level", "plev") if candidate in arr.coords and candidate not in arr.dims), None)
                if scalar_coord is None or len(requested_text) != 1:
                    raise
                scalar_text = _selector_text(arr[scalar_coord].values.item() if np.ndim(arr[scalar_coord].values) == 0 else arr[scalar_coord].values)
                if scalar_text != requested_text[0]:
                    raise PipelineError(f"Source fixture does not contain requested pressure level: {requested_text[0]}")
                scalar_value = arr[scalar_coord].values.item() if np.ndim(arr[scalar_coord].values) == 0 else arr[scalar_coord].values
                arr = arr.drop_vars(scalar_coord).expand_dims({dim_name: [scalar_value]})
            else:
                indices = _selector_indices(arr[native_dim].values, requested_text)
                arr = arr.isel({native_dim: indices})
                arr = arr.rename({native_dim: dim_name})
            coord_vals = arr[dim_name].values.copy()
            coords[dim_name] = xr.DataArray(coord_vals, dims=(dim_name,), attrs=dict(arr[dim_name].attrs))
            coords[dim_name].attrs.setdefault("units", "hPa")
            coords[dim_name].attrs["selector_dimension"] = selector_dim
            coords[dim_name].encoding.clear()
            arr = arr.assign_coords({dim_name: coords[dim_name]})
            desired_order = ["time", dim_name, "latitude", "longitude"]
        else:
            desired_order = ["time", "latitude", "longitude"]
        arr = arr.transpose(*desired_order)
        arr.name = variable
        arr.attrs = dict(arr.attrs)
        meta = option_meta.get(variable, {})
        if "units" not in arr.attrs and meta.get("units"):
            arr.attrs["units"] = meta["units"]
        if meta.get("description") and "description" not in arr.attrs:
            arr.attrs["description"] = meta["description"]
        arr.attrs["source_variable"] = variable
        arr.encoding.clear()
        out_vars[variable] = arr

    result = xr.Dataset(out_vars, coords=coords, attrs={
        "dataset_slug": contract.get("dataset_slug"),
        "provider": "ECMWF",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "publication_format": "zarr_v3_consolidated",
        "publication_data_chunk_codec": "pending",
        "semantic_note": "Decoded fixture values filtered exactly from the runtime contract; pressure-level channels sharing a native variable are grouped with a real selected selector dimension.",
    })
    for name in result.variables:
        result[name].encoding.clear()
    return result


def _filter_geography(ds: xr.Dataset, contract: dict[str, Any]) -> xr.Dataset:
    area = contract.get("scope", {}).get("geography", {}).get("cds_area", [90, -180, -90, 180])
    north, west, south, east = [float(x) for x in area]
    lat = ds["latitude"]
    lon = ds["longitude"]
    if float(lat[0]) <= float(lat[-1]):
        lat_mask = (lat >= south) & (lat <= north)
    else:
        lat_mask = (lat <= north) & (lat >= south)
    selected = ds.sel(latitude=lat[lat_mask])
    lon_values = selected["longitude"]
    if west <= east:
        lon_mask = (lon_values >= west) & (lon_values <= east)
    else:
        lon_mask = (lon_values >= west) | (lon_values <= east)
    # Global requests against 0..360 fixtures should retain native coordinates.
    if west <= -180 and east >= 180:
        lon_mask = xr.ones_like(lon_values, dtype=bool)
    selected = selected.sel(longitude=lon_values[lon_mask])
    if selected.sizes.get("latitude", 0) == 0 or selected.sizes.get("longitude", 0) == 0:
        raise PipelineError("Geographic filter selected no source grid cells")
    return selected


def _ensure_data_dims(arr: xr.DataArray) -> xr.DataArray:
    rename = {d: COORD_ALIASES[d] for d in arr.dims if d in COORD_ALIASES and COORD_ALIASES[d] not in arr.dims}
    if rename:
        arr = arr.rename(rename)
    for d in ("time", "latitude", "longitude"):
        if d not in arr.dims:
            raise PipelineError(f"Requested variable lacks required dimension: {d}")
    return arr


def _find_selector_dim(arr: xr.DataArray, selector_dimension: str) -> str:
    candidates = [selector_dimension, "isobaricInhPa", "level", "plev"]
    for candidate in candidates:
        if candidate in arr.dims:
            return candidate
    raise PipelineError(f"Requested variable lacks selector dimension: {selector_dimension}")


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _selector_indices(native_values: np.ndarray, requested_text: list[str]) -> list[int]:
    native_strings = [_selector_text(v) for v in native_values]
    indices: list[int] = []
    for req in requested_text:
        if req not in native_strings:
            raise PipelineError(f"Source fixture does not contain requested pressure level: {req}")
        indices.append(native_strings.index(req))
    return indices


def _selector_text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _encoding_for(ds: xr.Dataset, codec_mode: str) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    lat_n = int(ds.sizes["latitude"])
    lon_n = int(ds.sizes["longitude"])
    for name, var in ds.variables.items():
        dims = tuple(var.dims)
        if name in ds.data_vars:
            chunks = []
            for dim in dims:
                if dim == "time":
                    chunks.append(1)
                elif dim.endswith("pressure_level") or dim.endswith("_pressure_level"):
                    chunks.append(1)
                elif dim == "latitude":
                    chunks.append(lat_n)
                elif dim == "longitude":
                    chunks.append(lon_n)
                else:
                    chunks.append(int(ds.sizes[dim]))
            encoding[name] = {"chunks": tuple(chunks)}
        else:
            encoding[name] = {"chunks": tuple(max(1, int(ds.sizes[d])) for d in dims)}
        if codec_mode == "uncompressed_list":
            encoding[name]["compressors"] = []
        elif codec_mode == "uncompressed_none":
            encoding[name]["compressors"] = None
        elif codec_mode == "uncompressed_legacy":
            encoding[name]["compressor"] = None
        elif codec_mode == "lz4":
            from numcodecs import Blosc
            codec = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
            encoding[name]["compressors"] = [codec]
    return encoding


def _to_zarr(ds: xr.Dataset, path: Path, encoding: dict[str, dict[str, Any]]) -> None:
    kwargs = dict(store=str(path), mode="w", consolidated=True, encoding=encoding)
    try:
        ds.to_zarr(**kwargs, zarr_format=3)
    except TypeError:
        ds.to_zarr(**kwargs, zarr_version=3)


def _publish_zarr_atomic(ds: xr.Dataset, final_store: Path) -> str:
    parent = final_store.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{final_store.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    attempts = [
        ("uncompressed", "uncompressed_list"),
        ("uncompressed", "uncompressed_none"),
        ("uncompressed", "uncompressed_legacy"),
        ("lz4", "lz4"),
    ]
    last_error: Exception | None = None
    for public_mode, encoding_mode in attempts:
        if tmp.exists():
            shutil.rmtree(tmp)
        trial = ds.copy(deep=False)
        trial.attrs = dict(ds.attrs)
        trial.attrs["publication_data_chunk_codec"] = public_mode
        for name in trial.variables:
            trial[name].encoding.clear()
        try:
            _to_zarr(trial, tmp, _encoding_for(trial, encoding_mode))
            _validate_store_basic(tmp, public_mode)
            if final_store.exists():
                shutil.rmtree(final_store)
            os.replace(tmp, final_store)
            return public_mode
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                shutil.rmtree(tmp)
    raise PipelineError(f"Unable to publish deterministic Zarr v3 store: {last_error.__class__.__name__ if last_error else 'unknown'}")


def _validate_store_basic(store: Path, codec_mode: str) -> None:
    root_meta = store / "zarr.json"
    if not root_meta.exists():
        raise PipelineError("Published store is missing root zarr.json")
    meta = json.loads(root_meta.read_text(encoding="utf-8"))
    if meta.get("zarr_format") != 3:
        raise PipelineError("Published store is not Zarr format 3")
    if "consolidated_metadata" not in meta:
        raise PipelineError("Published Zarr v3 metadata is not consolidated")
    import zarr
    group = zarr.open_group(str(store), mode="r")
    for name in group.array_keys():
        arr = group[name]
        codec_text = json.dumps(_jsonable(getattr(arr.metadata, "codecs", [])), sort_keys=True).lower()
        if codec_mode == "uncompressed" and any(token in codec_text for token in ("blosc", "zstd", "gzip", "lz4")):
            raise PipelineError("Expected uncompressed data chunks but compression codec metadata is present")
        if codec_mode == "lz4" and "lz4" not in codec_text:
            raise PipelineError("Expected lossless Blosc-LZ4 fallback codec metadata")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _expected_chunks(ds: xr.Dataset, var_name: str) -> tuple[int, ...]:
    lat_n = int(ds.sizes["latitude"])
    lon_n = int(ds.sizes["longitude"])
    chunks: list[int] = []
    for dim in ds[var_name].dims:
        if dim == "time":
            chunks.append(1)
        elif dim.endswith("pressure_level") or dim.endswith("_pressure_level"):
            chunks.append(1)
        elif dim == "latitude":
            chunks.append(lat_n)
        elif dim == "longitude":
            chunks.append(lon_n)
        else:
            chunks.append(int(ds.sizes[dim]))
    return tuple(chunks)


def _validate_publication(expected: xr.Dataset, store: Path, codec_mode: str) -> None:
    _validate_store_basic(store, codec_mode)
    reopened = xr.open_zarr(str(store), consolidated=True)
    try:
        if set(reopened.data_vars) != set(expected.data_vars):
            raise PipelineError("Published data variables do not match selected source variables")
        for name in expected.data_vars:
            if tuple(reopened[name].dims) != tuple(expected[name].dims):
                raise PipelineError(f"Published dimensions differ for {name}")
            np.testing.assert_array_equal(reopened[name].values, expected[name].values)
            for attr_key, attr_value in expected[name].attrs.items():
                if reopened[name].attrs.get(attr_key) != attr_value:
                    raise PipelineError(f"Published attributes differ for {name}.{attr_key}")
        for coord in expected.coords:
            if coord not in reopened.coords:
                raise PipelineError(f"Published coordinate missing: {coord}")
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
        import zarr
        group = zarr.open_group(str(store), mode="r")
        for name in expected.data_vars:
            if tuple(group[name].chunks) != _expected_chunks(expected, name):
                raise PipelineError(f"Unexpected chunk topology for {name}: {group[name].chunks}")
    finally:
        reopened.close()


def _artifact_layout(channels: list[ChannelSpec], ds: xr.Dataset, store_name: str) -> dict[str, Any]:
    artifact_channels: list[dict[str, Any]] = []
    for spec in channels:
        selectors: dict[str, str] = {}
        selector_paths: dict[str, str] = {}
        if spec.selector_dimension is not None:
            selectors[spec.selector_dimension] = spec.selector_value_text or ""
            coord_name = f"{spec.variable}_{spec.selector_dimension}"
            if coord_name not in ds.coords:
                raise PipelineError("Internal error: grouped selector coordinate missing")
            selector_paths[spec.selector_dimension] = coord_name
        artifact_channels.append({
            "field_id": spec.field_id,
            "array_path": spec.variable,
            "selectors": selectors,
            "selector_coordinate_paths": selector_paths,
        })
    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": store_name,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": artifact_channels,
    }


def _sanity_read_full_tensor_sample(store: Path, artifact: dict[str, Any]) -> None:
    ds = xr.open_zarr(str(store), consolidated=True)
    try:
        planes: list[np.ndarray] = []
        for channel in artifact["channels"]:
            arr = ds[channel["array_path"]]
            indexers: dict[str, int] = {"time": 0}
            for selector_dim, selector_value in channel.get("selectors", {}).items():
                coord_path = channel["selector_coordinate_paths"][selector_dim]
                coord_values = [_selector_text(v) for v in ds[coord_path].values]
                if selector_value not in coord_values:
                    raise PipelineError("Artifact selector does not resolve in published store")
                indexers[coord_path] = coord_values.index(selector_value)
            plane = arr.isel(indexers).values
            if plane.shape != (ds.sizes["latitude"], ds.sizes["longitude"]):
                raise PipelineError("Full-field channel plane has unexpected H,W shape")
            planes.append(plane)
        tensor = np.stack(planes, axis=0)
        expected_shape = (len(artifact["channels"]), ds.sizes["latitude"], ds.sizes["longitude"])
        if tensor.shape != expected_shape:
            raise PipelineError("C,H,W tensor sanity read failed")
    finally:
        ds.close()
