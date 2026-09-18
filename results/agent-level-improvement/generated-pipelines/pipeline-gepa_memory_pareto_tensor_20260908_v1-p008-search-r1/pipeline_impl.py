"""Standalone ERA5 pressure-level family adapter.

Entry point required by the framework:
    run_pipeline(contract_lock, inventory, cache_dir, output_dir)

The adapter is deliberately offline-first.  A complete source fixture manifest at
``cache_dir/source_fixture_manifest.json`` is verified and consumed before any
provider/network/credential logic.  This implementation performs no network
access; without a verified fixture it fails closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p008-search-r1"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "dataset.zarr"
SOURCE_ALIASES = {
    "temperature": ("temperature", "t", "air_temperature"),
    "geopotential": ("geopotential", "z"),
}
COORD_ALIASES = {
    "time": ("time", "valid_time", "datetime"),
    "pressure_level": ("pressure_level", "isobaricInhPa", "isobaric", "level", "plev"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon"),
}


class ContractError(ValueError):
    """Raised when the runtime lock is not supported by the frozen inventory."""


class FixtureError(RuntimeError):
    """Raised when a required local source fixture is absent or invalid."""


@dataclass(frozen=True)
class RequestedChannel:
    field_name: str
    field_id: str
    source_var: str
    selector_dim: str
    selector_value: str
    array_name: str
    selector_coord_name: str


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    """Materialize a validated runtime contract from verified local source files.

    Parameters are supplied by the framework.  ``contract_lock`` may be either a
    raw DatasetContract or a lock envelope containing one.  ``inventory`` is the
    frozen DatasetInventory.  ``cache_dir`` is treated as an external read-only
    source/cache root when it contains ``source_fixture_manifest.json``.
    """

    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fixture_entries, source_paths = _verify_source_fixture(cache_root)
    # Only after successful fixture verification do we inspect/validate runtime
    # request semantics.  This adapter never constructs a provider or checks
    # credentials, so complete fixtures are credential-free and offline.
    contract = _extract_contract(contract_lock)
    _validate_contract_against_inventory(contract, inventory)

    expected_times = _contract_timestamps(contract)
    channels = _requested_channels(contract)
    source = _open_verified_sources(source_paths)
    source = _normalize_dataset(source)
    _validate_source_covers_request(source, channels, expected_times, contract)
    filtered, artifact_channels = _build_publication_dataset(source, contract, channels, expected_times)

    store_path = out_root / STORE_NAME
    codec, codec_record, codec_warning = _preferred_lossless_codec()
    warnings: list[str] = []
    if codec_warning:
        warnings.append(codec_warning)

    encoding = _zarr_encoding(filtered, codec)
    _publish_zarr_atomically(filtered, store_path, encoding)
    reopened = _open_published(store_path)
    _validate_published_dataset(filtered, reopened, artifact_channels, codec_record)

    cache_keys = [f"fixture:{e['entry_id']}:{e['sha256'][:16]}" for e in fixture_entries]
    result = {
        "cache": {
            "hits": len(fixture_entries),
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
            "channels": artifact_channels,
        },
        "warnings": warnings,
    }
    return result


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get("schema_version") == "dataset_contract.v1":
        return lock
    if isinstance(lock, dict):
        for key in ("contract", "dataset_contract", "selected_contract", "request", "payload"):
            value = lock.get(key)
            if isinstance(value, dict) and value.get("schema_version") == "dataset_contract.v1":
                return value
        for value in lock.values():
            if isinstance(value, dict):
                try:
                    return _extract_contract(value)
                except ContractError:
                    pass
    raise ContractError("No dataset_contract.v1 object found in contract lock envelope")


def _verify_source_fixture(cache_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise FixtureError("A complete local source_fixture_manifest.json is required; network fallback is disabled")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise FixtureError("Unsupported source fixture manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise FixtureError("Source fixture manifest must contain at least one entry")

    verified: list[Path] = []
    root_s = str(cache_root)
    for entry in entries:
        rel = entry.get("relative_path")
        sha = entry.get("sha256")
        size = entry.get("size_bytes")
        entry_id = entry.get("entry_id")
        if not isinstance(rel, str) or not isinstance(sha, str) or not isinstance(size, int) or not isinstance(entry_id, str):
            raise FixtureError("Malformed source fixture entry")
        path = (cache_root / rel).resolve()
        if os.path.commonpath([root_s, str(path)]) != root_s:
            raise FixtureError("Source fixture path escapes cache_dir")
        if not path.is_file():
            raise FixtureError(f"Missing source fixture file for entry {entry_id}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise FixtureError(f"Size mismatch for source fixture entry {entry_id}")
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        if actual_sha != sha:
            raise FixtureError(f"SHA-256 mismatch for source fixture entry {entry_id}")
        verified.append(path)
    return entries, verified


def _validate_contract_against_inventory(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if contract.get("dataset_slug") != DATASET_SLUG or inventory.get("dataset_slug") != DATASET_SLUG:
        raise ContractError("Dataset slug is not supported by this adapter/inventory")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ContractError("Inventory dataset_id mismatch")
    options = inventory.get("options", {})
    adv = contract.get("advanced_options", {}) or {}
    if adv.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ContractError("Unsupported dataset_id in contract advanced_options")
    if adv.get("data_format", "grib") not in options.get("data_format", []):
        raise ContractError("Unsupported data_format")
    if adv.get("download_format", "unarchived") not in options.get("download_format", []):
        raise ContractError("Unsupported download_format")

    product_type = contract.get("scope", {}).get("product_type", "reanalysis")
    if product_type not in options.get("product_type", []):
        raise ContractError("Unsupported product_type")
    for pt in adv.get("product_type", [product_type]):
        if pt not in options.get("product_type", []):
            raise ContractError("Unsupported advanced product_type")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("Contract must request at least one field")
    for f in fields:
        name = f.get("name")
        if name not in options.get("variable", []):
            raise ContractError(f"Unsupported variable {name!r}")
        selectors = f.get("selectors") or []
        if len(selectors) != 1 or selectors[0].get("dimension") != "pressure_level":
            raise ContractError("ERA5 pressure-level fields must specify exactly one pressure_level selector")
        value = str(selectors[0].get("value"))
        if value not in options.get("pressure_level", []):
            raise ContractError(f"Unsupported pressure_level {value!r}")

    for ts in _contract_timestamps(contract):
        ts_dt = ts.astype("datetime64[s]").astype(datetime)
        y, m, d, hhmm = f"{ts_dt.year:04d}", f"{ts_dt.month:02d}", f"{ts_dt.day:02d}", f"{ts_dt.hour:02d}:00"
        if y not in options.get("year", []) or m not in options.get("month", []) or d not in options.get("day", []):
            raise ContractError(f"Requested date {ts_dt.date()} is outside inventory options")
        if hhmm not in options.get("time", []):
            raise ContractError(f"Requested time {hhmm} is outside inventory options")

    geo = contract.get("scope", {}).get("geography", {})
    area = geo.get("cds_area", inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)):
        raise ContractError("cds_area must be a four-number north/west/south/east list")
    north, west, south, east = map(float, area)
    if north < south or not (-90 <= south <= 90) or not (-90 <= north <= 90) or not (-360 <= west <= 360) or not (-360 <= east <= 360):
        raise ContractError("Invalid geographic area bounds")


def _contract_timestamps(contract: dict[str, Any]) -> np.ndarray:
    scope = contract.get("scope", {})
    dr = scope.get("date_range", {})
    start = date.fromisoformat(dr["start_date"])
    end = date.fromisoformat(dr["end_date"])
    inclusive = bool(dr.get("inclusive", True))
    if end < start:
        raise ContractError("date_range end_date precedes start_date")
    selected_times = scope.get("time", {}).get("selected_times", [])
    if not selected_times:
        raise ContractError("At least one selected time is required")
    parsed_times: list[time] = []
    for s in selected_times:
        if not re.fullmatch(r"\d{2}:\d{2}", str(s)):
            raise ContractError(f"Invalid selected time {s!r}")
        hh, mm = map(int, str(s).split(":"))
        parsed_times.append(time(hh, mm, tzinfo=timezone.utc))
    last = end if inclusive else end - timedelta(days=1)
    if last < start:
        return np.array([], dtype="datetime64[ns]")
    values: list[np.datetime64] = []
    cur = start
    while cur <= last:
        for t in parsed_times:
            dt = datetime(cur.year, cur.month, cur.day, t.hour, t.minute, tzinfo=timezone.utc)
            values.append(np.datetime64(dt.replace(tzinfo=None), "ns"))
        cur += timedelta(days=1)
    return np.array(values, dtype="datetime64[ns]")


def _canonical_field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    inner = ",".join(f"{s['dimension']}={json.dumps(str(s['value']))}" for s in selectors)
    return f"{name}[{inner}]"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def _requested_channels(contract: dict[str, Any]) -> list[RequestedChannel]:
    out: list[RequestedChannel] = []
    seen: set[str] = set()
    for f in contract["fields"]:
        selectors = f.get("selectors") or []
        field_id = _canonical_field_id(f["name"], selectors)
        if field_id in seen:
            raise ContractError(f"Duplicate requested channel {field_id}")
        seen.add(field_id)
        sel = selectors[0]
        value = str(sel["value"])
        arr = _safe_name(f"{f['name']}__pressure_level_{value}")
        coord = _safe_name(f"pressure_level__{f['name']}__{value}")
        out.append(RequestedChannel(f["name"], field_id, "", "pressure_level", value, arr, coord))
    return out


def _open_verified_sources(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in paths:
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith((".grib", ".grb", ".grib2", ".grb2")):
            try:
                import cfgrib  # noqa: F401
                parts = xr.backends.cfgrib_.CfGribBackendEntrypoint().open_dataset(str(path), indexpath="")
                datasets.append(_normalize_dataset(parts.load()))
            except Exception:
                import cfgrib
                for ds in cfgrib.open_datasets(str(path), indexpath=""):
                    datasets.append(_normalize_dataset(ds.load()))
        else:
            with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as ds:
                datasets.append(_normalize_dataset(ds.load()))
    if not datasets:
        raise FixtureError("No readable source datasets in fixture")
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs="drop_conflicts").load()
    except Exception:
        return xr.merge(datasets, compat="no_conflicts", combine_attrs="drop_conflicts").load()


def _find_name(ds: xr.Dataset, canonical: str, aliases: dict[str, tuple[str, ...]], *, variables: bool = False) -> str:
    candidates = aliases.get(canonical, (canonical,))
    names = list(ds.data_vars if variables else set(ds.coords) | set(ds.dims))
    for c in candidates:
        if c in names:
            return c
    raise ContractError(f"Could not find required {'variable' if variables else 'coordinate'} {canonical!r} in source fixture")


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    renames: dict[str, str] = {}
    for canonical in ("time", "pressure_level", "latitude", "longitude"):
        try:
            src = _find_name(ds, canonical, COORD_ALIASES)
        except ContractError:
            continue
        if src != canonical:
            renames[src] = canonical
    ds = ds.rename(renames)
    # Normalize source variable names to contract names where known aliases are present.
    vrenames: dict[str, str] = {}
    for canonical, aliases in SOURCE_ALIASES.items():
        for alias in aliases:
            if alias in ds.data_vars and alias != canonical and canonical not in ds.data_vars:
                vrenames[alias] = canonical
                break
    if vrenames:
        ds = ds.rename(vrenames)
    if "pressure_level" in ds.coords and "pressure_level" not in ds.dims and ds["pressure_level"].ndim == 0:
        level_value = ds["pressure_level"].values.item()
        ds = ds.drop_vars("pressure_level").expand_dims(pressure_level=[level_value])
    for required in ("time", "pressure_level", "latitude", "longitude"):
        if required not in ds.coords and required not in ds.dims:
            raise ContractError(f"Source fixture lacks required coordinate/dimension {required}")
    return ds


def _validate_source_covers_request(ds: xr.Dataset, channels: list[RequestedChannel], times: np.ndarray, contract: dict[str, Any]) -> None:
    source_times = ds["time"].values.astype("datetime64[ns]")
    missing_times = [str(t) for t in times if t not in source_times]
    if missing_times:
        raise ContractError(f"Source fixture does not cover requested timestamps; first missing {missing_times[0]}")
    pvals = {_selector_token(v) for v in ds["pressure_level"].values}
    for ch in channels:
        if ch.field_name not in ds.data_vars:
            raise ContractError(f"Source fixture lacks requested variable {ch.field_name}")
        if _selector_token(ch.selector_value) not in pvals:
            raise ContractError(f"Source fixture lacks pressure_level={ch.selector_value}")
    north, west, south, east = _contract_area(contract)
    lat = ds["latitude"].values
    if not np.any((lat <= north) & (lat >= south)):
        raise ContractError("Source fixture lacks requested latitude coverage")
    lon = ds["longitude"].values
    if not _is_global_area(north, west, south, east) and not np.any(_longitude_mask(lon, west, east)):
        raise ContractError("Source fixture lacks requested longitude coverage")


def _selector_token(v: Any) -> str:
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return repr(f)
    except Exception:
        return str(v)


def _contract_area(contract: dict[str, Any]) -> tuple[float, float, float, float]:
    area = contract.get("scope", {}).get("geography", {}).get("cds_area", [90, -180, -90, 180])
    north, west, south, east = map(float, area)
    return north, west, south, east


def _is_global_area(north: float, west: float, south: float, east: float) -> bool:
    return north >= 90 and south <= -90 and abs(east - west) >= 360


def _longitude_mask(lon: np.ndarray, west: float, east: float) -> np.ndarray:
    lon = np.asarray(lon)
    if np.nanmin(lon) >= 0 and (west < 0 or east < 0):
        west_m = west % 360
        east_m = east % 360
        if west_m <= east_m:
            return (lon >= west_m) & (lon <= east_m)
        return (lon >= west_m) | (lon <= east_m)
    if west <= east:
        return (lon >= west) & (lon <= east)
    return (lon >= west) | (lon <= east)


def _build_publication_dataset(
    ds: xr.Dataset,
    contract: dict[str, Any],
    channels: list[RequestedChannel],
    times: np.ndarray,
) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    north, west, south, east = _contract_area(contract)
    time_indexer = np.nonzero(np.isin(ds["time"].values.astype("datetime64[ns]"), times))[0]
    base = ds.isel(time=time_indexer)
    # Reorder exactly as the runtime lock requested, not as the source happened to store.
    base = base.sel(time=times)
    lat_mask = (base["latitude"].values <= north) & (base["latitude"].values >= south)
    base = base.isel(latitude=np.nonzero(lat_mask)[0])
    if not _is_global_area(north, west, south, east):
        lon_mask = _longitude_mask(base["longitude"].values, west, east)
        base = base.isel(longitude=np.nonzero(lon_mask)[0])

    out_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        "time": base["time"],
        "latitude": base["latitude"],
        "longitude": base["longitude"],
    }
    artifact_channels: list[dict[str, Any]] = []
    for ch in channels:
        p_idx = [i for i, v in enumerate(base["pressure_level"].values) if _selector_token(v) == _selector_token(ch.selector_value)]
        if len(p_idx) != 1:
            raise ContractError(f"Expected exactly one source level for {ch.field_id}")
        src = base[ch.field_name].isel(pressure_level=p_idx)
        src = src.rename({"pressure_level": ch.selector_coord_name})
        if ch.selector_coord_name not in src.coords:
            src = src.assign_coords({ch.selector_coord_name: [base["pressure_level"].values[p_idx[0]]]})
        src = src.transpose("time", ch.selector_coord_name, "latitude", "longitude")
        attrs = dict(src.attrs)
        attrs.setdefault("units", _contract_or_known_units(contract, ch.field_name))
        attrs["field_id"] = ch.field_id
        attrs["source_variable"] = ch.field_name
        attrs["selector_dimension"] = "pressure_level"
        attrs["selector_value"] = ch.selector_value
        out_vars[ch.array_name] = src.astype(src.dtype, copy=False).assign_attrs(_json_safe_attrs(attrs))
        coords[ch.selector_coord_name] = src[ch.selector_coord_name].assign_attrs({"units": "hPa", "selector_dimension": "pressure_level"})
        artifact_channels.append({
            "field_id": ch.field_id,
            "array_path": ch.array_name,
            "selectors": {"pressure_level": ch.selector_value},
            "selector_coordinate_paths": {"pressure_level": ch.selector_coord_name},
        })

    out = xr.Dataset(out_vars, coords=coords, attrs=_json_safe_attrs({
        "dataset_slug": DATASET_SLUG,
        "dataset_id": DATASET_ID,
        "publication_policy": "regular_grid_zarr_output_policy.v1",
        "zarr_format": 3,
        "consolidated_metadata": True,
        "chunk_policy": "time=1, selector=1, latitude=full, longitude=full",
    }))
    for c in ("time", "latitude", "longitude"):
        if c in base[c].attrs:
            out[c].attrs.update(_json_safe_attrs(base[c].attrs))
    return out, artifact_channels


def _contract_or_known_units(contract: dict[str, Any], field_name: str) -> str | None:
    for f in contract.get("fields", []):
        if f.get("name") == field_name and f.get("units"):
            return f["units"]
    return {"temperature": "K", "geopotential": "m**2 s**-2"}.get(field_name)


def _json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in {"scale_factor", "add_offset", "_FillValue", "missing_value"}:
            # These are source encoding conventions already decoded/masked by xarray.
            continue
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[str(k)] = v
        elif isinstance(v, (list, tuple)):
            safe[str(k)] = [x.item() if isinstance(x, np.generic) else x for x in v]
        else:
            safe[str(k)] = str(v)
    return safe


def _preferred_lossless_codec() -> tuple[Any, dict[str, Any], str | None]:
    try:
        from zarr.codecs import BloscCodec, BloscShuffle
        shuffle = getattr(BloscShuffle, "bitshuffle", None) or getattr(BloscShuffle, "BITSHUFFLE", None) or "bitshuffle"
        codec = BloscCodec(cname="lz4", clevel=1, shuffle=shuffle)
        return codec, {"name": "blosc", "cname": "lz4", "clevel": 1, "shuffle": "bitshuffle"}, None
    except Exception as exc:
        try:
            from zarr.codecs import ZstdCodec
            codec = ZstdCodec(level=1)
            return codec, {"name": "zstd", "level": 1, "fallback_for": "blosc_lz4_bitshuffle"}, (
                "Host zarr library could not express Blosc-LZ4 bitshuffle through maintained Zarr v3 APIs; "
                f"used deterministic low-level Zstd fallback. Reason: {type(exc).__name__}"
            )
        except Exception as exc2:
            raise RuntimeError("No maintained Zarr v3 lossless compressor is available") from exc2


def _zarr_encoding(ds: xr.Dataset, codec: Any) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    nlat = int(ds.sizes["latitude"])
    nlon = int(ds.sizes["longitude"])
    for name, da in ds.data_vars.items():
        selector_dims = [d for d in da.dims if d not in {"time", "latitude", "longitude"}]
        if len(selector_dims) != 1:
            raise RuntimeError(f"Data variable {name} does not retain exactly one selector dimension")
        chunks = (1, 1, nlat, nlon)
        enc[name] = {"chunks": chunks, "compressors": (codec,)}
    for cname, coord in ds.coords.items():
        if coord.ndim == 1:
            enc[cname] = {"chunks": (int(coord.sizes[coord.dims[0]]),), "compressors": ()}
    return enc


def _publish_zarr_atomically(ds: xr.Dataset, final_store: Path, encoding: dict[str, dict[str, Any]]) -> None:
    tmp = final_store.with_name(final_store.name + ".tmp")
    backup = final_store.with_name(final_store.name + ".bak")
    for p in (tmp, backup):
        if p.exists():
            shutil.rmtree(p)
    # Ensure decoded values are not silently repacked from source encodings.
    clean = ds.copy(deep=False)
    for v in clean.variables:
        clean[v].encoding = {}
    try:
        clean.to_zarr(tmp, mode="w", consolidated=True, zarr_format=3, encoding=encoding)
    except TypeError:
        # Older xarray pre-release builds used zarr_version instead of zarr_format.
        clean.to_zarr(tmp, mode="w", consolidated=True, zarr_version=3, encoding=encoding)
    _assert_consolidated_metadata(tmp)
    if final_store.exists():
        final_store.rename(backup)
    try:
        tmp.rename(final_store)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if final_store.exists():
            shutil.rmtree(final_store)
        if backup.exists():
            backup.rename(final_store)
        raise


def _assert_consolidated_metadata(store: Path) -> None:
    root_meta = json.loads((store / "zarr.json").read_text())
    if root_meta.get("zarr_format") != 3:
        raise RuntimeError("Published store is not Zarr v3")
    if "consolidated_metadata" not in root_meta:
        raise RuntimeError("Published Zarr v3 metadata is not consolidated")


def _open_published(store: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    except TypeError:
        return xr.open_zarr(store, consolidated=True, zarr_version=3).load()


def _validate_published_dataset(expected: xr.Dataset, actual: xr.Dataset, artifact_channels: list[dict[str, Any]], codec_record: dict[str, Any]) -> None:
    if set(expected.data_vars) != set(actual.data_vars):
        raise RuntimeError("Published data variables do not match expected channels")
    for cname in expected.coords:
        if cname not in actual.coords:
            raise RuntimeError(f"Missing published coordinate {cname}")
        if not np.array_equal(expected[cname].values, actual[cname].values, equal_nan=True):
            raise RuntimeError(f"Published coordinate {cname} differs from decoded filtered source")
    for v in expected.data_vars:
        if expected[v].dims != actual[v].dims:
            raise RuntimeError(f"Published dimensions for {v} changed")
        if not np.array_equal(expected[v].values, actual[v].values, equal_nan=True):
            raise RuntimeError(f"Published values for {v} differ from decoded filtered source")
        for attr in ("units", "field_id", "selector_dimension", "selector_value"):
            if attr in expected[v].attrs and expected[v].attrs.get(attr) != actual[v].attrs.get(attr):
                raise RuntimeError(f"Published attribute {attr} for {v} changed")
    declared = {c["array_path"] for c in artifact_channels}
    if declared != set(expected.data_vars):
        raise RuntimeError("Artifact channel declarations do not match data arrays")
    if codec_record.get("name") == "blosc" and codec_record.get("cname") != "lz4":
        raise RuntimeError("Primary codec record is not LZ4")
