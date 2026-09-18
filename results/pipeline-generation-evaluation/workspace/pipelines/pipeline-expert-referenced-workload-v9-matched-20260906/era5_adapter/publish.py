from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .contract import canonical_field_id, requested_timestamps


SHORT_NAMES = {
    "temperature": "t",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "divergence": "d",
    "ozone_mass_mixing_ratio": "o3",
    "potential_vorticity": "pv",
    "fraction_of_cloud_cover": "cc",
}


@dataclass(frozen=True)
class ArtifactChannel:
    field_id: str
    array_path: str
    selectors: dict[str, str]
    selector_coordinate_paths: dict[str, str]


class PublicationError(RuntimeError):
    """Raised when public Zarr construction or validation fails."""


def build_public_dataset(contract: dict[str, Any], inventory: dict[str, Any], sources: list[xr.Dataset]) -> tuple[xr.Dataset, list[ArtifactChannel]]:
    times = np.array(requested_timestamps(contract, inventory), dtype="datetime64[ns]")
    if len(times) == 0:
        raise PublicationError("contract selected zero timestamps")
    merged = _merge_sources(sources)
    merged = _select_times(merged, times)
    merged = _select_area(merged, contract)

    coords: dict[str, Any] = {
        "time": merged["time"].values.astype("datetime64[ns]"),
        "latitude": merged["latitude"].values,
        "longitude": merged["longitude"].values,
    }
    data_vars: dict[str, xr.DataArray] = {}
    channels: list[ArtifactChannel] = []

    for field in contract["fields"]:
        fid = canonical_field_id(field)
        sel = field["selectors"][0]
        level_text = str(sel["value"])
        level_value = _typed_level(level_text)
        var_name = _find_var(merged, field["name"])
        arr = merged[var_name]
        arr = _select_or_reconstruct_pressure(arr, level_value, level_text)
        arr = _ensure_spatial_dims(arr, merged)
        arr = arr.transpose("time", "pressure_level", "latitude", "longitude")
        if "pressure_level" in arr.coords:
            arr = arr.drop_vars("pressure_level")
        arr = arr.astype("float32", copy=False).load()
        arr.encoding = {}

        public_name = _safe_array_name(field["name"], {"pressure_level": level_text})
        selector_coord_name = f"pressure_level__{public_name}"
        coords[selector_coord_name] = ("pressure_level", np.array([level_value]))
        arr = arr.assign_coords({selector_coord_name: ("pressure_level", np.array([level_value]))})
        arr.attrs = dict(arr.attrs)
        meta = ((inventory.get("option_metadata") or {}).get("variable") or {}).get(field["name"], {})
        if meta.get("units") and not arr.attrs.get("units"):
            arr.attrs["units"] = meta["units"]
        arr.attrs["field_id"] = fid
        arr.attrs["selector_pressure_level"] = level_text
        arr.attrs["selector_pressure_level_units"] = sel.get("unit") or "hPa"
        data_vars[public_name] = arr
        channels.append(
            ArtifactChannel(
                field_id=fid,
                array_path=public_name,
                selectors={"pressure_level": level_text},
                selector_coordinate_paths={"pressure_level": selector_coord_name},
            )
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(
        {
            "title": contract.get("title", "ERA5 pressure-level materialization"),
            "dataset_slug": contract.get("dataset_slug"),
            "provider": contract.get("provider") or inventory.get("provider"),
            "source_url": contract.get("source_url") or inventory.get("source_url"),
            "publication_policy": "regular_grid_zarr_output_policy.v1",
        }
    )
    for name in ds.variables:
        ds[name].encoding = {}
    return ds, channels


def publish_dataset(ds: xr.Dataset, output_dir: Path, store_rel: str, channels: list[ArtifactChannel]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / store_rel
    tmp = output_dir / f".{store_rel}.tmp-{uuid.uuid4().hex}"
    backup = output_dir / f".{store_rel}.bak-{uuid.uuid4().hex}"
    if tmp.exists():
        shutil.rmtree(tmp)

    encoding = _encoding_for(ds)
    try:
        ds.to_zarr(tmp, mode="w", encoding=encoding, consolidated=True, zarr_format=3)
        _validate_store(tmp, ds, channels)
        if final.exists():
            os.replace(final, backup)
        os.replace(tmp, final)
        if backup.exists():
            shutil.rmtree(backup)
        _validate_store(final, ds, channels)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise

    return {
        "schema_version": "dataset_artifact_layout.v1",
        "storage_format": "zarr",
        "store_path": store_rel,
        "dimensions": {"sample": "time", "y": "latitude", "x": "longitude"},
        "coordinates": {"sample": "time", "y": "latitude", "x": "longitude"},
        "channels": [
            {
                "field_id": c.field_id,
                "array_path": c.array_path,
                "selectors": c.selectors,
                "selector_coordinate_paths": c.selector_coordinate_paths,
            }
            for c in channels
        ],
    }


def _merge_sources(sources: list[xr.Dataset]) -> xr.Dataset:
    by_var: dict[str, list[xr.DataArray]] = {}
    for ds in sources:
        for name, da in ds.data_vars.items():
            if "pressure_level" not in da.dims and "pressure_level" in da.coords:
                scalar = da["pressure_level"].values
                if np.ndim(scalar) == 0:
                    da = da.expand_dims(pressure_level=[scalar.item()])
            by_var.setdefault(name, []).append(da)
    if not by_var:
        raise PublicationError("fixture contains no data variables")

    merged_vars: list[xr.Dataset] = []
    for name, arrays in by_var.items():
        if len(arrays) == 1:
            merged_vars.append(arrays[0].to_dataset(name=name))
            continue
        try:
            merged = xr.combine_by_coords(
                [arr.to_dataset(name=name) for arr in arrays],
                compat="override",
                join="outer",
                combine_attrs="override",
            )
        except Exception as exc:
            raise PublicationError(f"could not merge fixture hypercubes for {name}: {exc}") from exc
        merged_vars.append(merged[[name]])

    try:
        return xr.merge(merged_vars, compat="override", join="outer")
    except Exception as exc:
        raise PublicationError(f"could not merge fixture hypercubes: {exc}") from exc


def _find_var(ds: xr.Dataset, long_name: str) -> str:
    candidates = [long_name, SHORT_NAMES.get(long_name, "")]
    for cand in candidates:
        if cand and cand in ds.data_vars:
            return cand
    for name, da in ds.data_vars.items():
        attrs = {str(k).lower(): str(v).lower() for k, v in da.attrs.items()}
        if attrs.get("long_name") == long_name.lower() or attrs.get("standard_name") == long_name.lower():
            return name
    raise PublicationError(f"fixture does not contain requested variable {long_name!r}")


def _select_times(ds: xr.Dataset, times: np.ndarray) -> xr.Dataset:
    if "time" not in ds.coords:
        raise PublicationError("fixture is missing time coordinate")
    source_times = ds["time"].values.astype("datetime64[ns]")
    ds = ds.assign_coords(time=source_times)
    missing = sorted(set(times.tolist()) - set(source_times.tolist()))
    if missing:
        raise PublicationError(f"fixture is missing requested timestamps, first missing: {missing[0]}")
    return ds.sel(time=times)


def _select_area(ds: xr.Dataset, contract: dict[str, Any]) -> xr.Dataset:
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise PublicationError("fixture is missing latitude/longitude coordinates")
    geo = (contract.get("scope") or {}).get("geography") or {}
    area = geo.get("cds_area")
    if not area or geo.get("area") == "global" or list(area) == [90, -180, -90, 180]:
        return ds
    north, west, south, east = [float(v) for v in area]
    lat = ds["latitude"].values
    if lat[0] > lat[-1]:
        ds = ds.sel(latitude=slice(north, south))
    else:
        ds = ds.sel(latitude=slice(south, north))
    lon = ds["longitude"].values
    if lon.min() >= 0 and west < 0:
        west = west % 360
        east = east % 360
    if west <= east:
        ds = ds.sel(longitude=slice(west, east))
    else:
        raise PublicationError("wrapped longitude subregions are not supported by this compact adapter")
    return ds


def _typed_level(text: str) -> Any:
    try:
        val = int(text)
        return val
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _select_or_reconstruct_pressure(arr: xr.DataArray, level_value: Any, level_text: str) -> xr.DataArray:
    if "pressure_level" in arr.dims:
        coord_vals = arr["pressure_level"].values
        key = None
        for candidate in coord_vals.tolist():
            if str(candidate) == level_text or str(candidate) == str(level_value):
                key = candidate
                break
            try:
                if float(candidate) == float(level_value):
                    key = candidate
                    break
            except (TypeError, ValueError):
                pass
        if key is None:
            raise PublicationError(f"pressure_level {level_text} not present for {arr.name}")
        return arr.sel(pressure_level=[key])
    if "pressure_level" in arr.coords:
        scalar = arr["pressure_level"].values
        if np.ndim(scalar) == 0:
            if str(scalar.item()) != level_text and scalar.item() != level_value:
                raise PublicationError(f"scalar pressure_level mismatch for {arr.name}")
            return arr.expand_dims(pressure_level=[level_value])
    # Some GRIB decoders expose the selected isobaric level only as metadata.
    for attr in ("GRIB_level", "level"):
        if attr in arr.attrs and str(arr.attrs[attr]) not in {level_text, str(level_value)}:
            raise PublicationError(f"metadata pressure level mismatch for {arr.name}")
    return arr.expand_dims(pressure_level=[level_value])


def _ensure_spatial_dims(arr: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    for dim in ("time", "latitude", "longitude"):
        if dim not in arr.dims:
            if dim in ds.coords:
                arr = arr.expand_dims({dim: ds[dim].values})
            else:
                raise PublicationError(f"array {arr.name} is missing dimension {dim}")
    return arr


def _safe_array_name(field: str, selectors: dict[str, str]) -> str:
    base = field + "__" + "__".join(f"{k}_{v}" for k, v in selectors.items())
    return re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")


def _encoding_for(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    y = int(ds.sizes.get("latitude", 1))
    x = int(ds.sizes.get("longitude", 1))
    y_chunk = _y_chunk_for_float32(x, y)
    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(1)
            elif dim == "pressure_level":
                chunks.append(1)
            elif dim == "latitude":
                chunks.append(y_chunk)
            elif dim == "longitude":
                chunks.append(x)
            else:
                chunks.append(int(ds.sizes[dim]))
        enc[name] = {"chunks": tuple(chunks), "dtype": "float32"}
    for name in ds.coords:
        if name in ds.dims:
            enc[name] = {"chunks": (int(ds.sizes[name]),)}
        elif ds[name].dims:
            enc[name] = {"chunks": tuple(int(ds.sizes[d]) for d in ds[name].dims)}
    return enc


def _y_chunk_for_float32(x_size: int, y_size: int) -> int:
    if x_size <= 0:
        return max(1, y_size)
    target = int((1536 * 1024) / (4 * x_size))
    target = max(16, target)
    return max(1, min(y_size, target))


def _validate_store(path: Path, expected: xr.Dataset, channels: list[ArtifactChannel]) -> None:
    reopened = xr.open_zarr(path, consolidated=True)
    try:
        for coord in ("time", "latitude", "longitude"):
            if coord not in reopened.coords:
                raise PublicationError(f"published store missing coordinate {coord}")
        for c in channels:
            if c.array_path not in reopened.data_vars:
                raise PublicationError(f"published store missing channel array {c.array_path}")
            arr = reopened[c.array_path]
            if tuple(arr.dims) != ("time", "pressure_level", "latitude", "longitude"):
                raise PublicationError(f"unexpected dimension order for {c.array_path}: {arr.dims}")
            for selector_path in c.selector_coordinate_paths.values():
                if selector_path not in reopened.coords:
                    raise PublicationError(f"missing selector coordinate {selector_path}")
        # Force a small read-back from every array path returned to the framework.
        for c in channels:
            _ = reopened[c.array_path].isel(time=0, pressure_level=0).values
    finally:
        reopened.close()
