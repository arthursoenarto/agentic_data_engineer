"""Standalone ERA5 pressure-level fixture adapter.

Implements run_pipeline(contract_lock, inventory, cache_dir, output_dir) for the
framework interface.  The adapter is deliberately offline-only: a verified
source_fixture_manifest.json is required and no provider/client/network fallback
is attempted.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

PIPELINE_ID = "pipeline-gepa_memory_pareto_tensor_20260908_v1-p016-confirmation-r3"
DATASET_SLUG = "reanalysis_era5_pressure_levels"
DATASET_ID = "reanalysis-era5-pressure-levels"
STORE_NAME = "era5_pressure_levels.zarr"
SELECTOR_DIM = "pressure_level"
Y_DIM = "latitude"
X_DIM = "longitude"
SAMPLE_DIM = "time"

SHORT_TO_LONG = {"t": "temperature", "z": "geopotential"}
COORD_ALIASES = {
    "valid_time": SAMPLE_DIM,
    "time": SAMPLE_DIM,
    "latitude": Y_DIM,
    "lat": Y_DIM,
    "longitude": X_DIM,
    "lon": X_DIM,
    "isobaricInhPa": SELECTOR_DIM,
    "isobaricInPa": SELECTOR_DIM,
    "level": SELECTOR_DIM,
    "plev": SELECTOR_DIM,
}


class ContractError(ValueError):
    """Raised when the runtime lock is not valid for this inventory/policy."""


@dataclass(frozen=True)
class VerifiedSource:
    entry_id: str
    path: Path
    size_bytes: int
    sha256: str


def run_pipeline(contract_lock: Any, inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Materialize a contract-selected ERA5 pressure-level regular-grid Zarr store.

    Parameters match the framework contract exactly.  The function performs all
    acquisition from a complete, verified local fixture manifest under cache_dir.
    """
    contract = _extract_contract(contract_lock)
    request = _validate_contract(contract, inventory)
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    sources = _verify_fixture(cache_root)

    raw = _open_sources([s.path for s in sources])
    raw = _normalise_source_dataset(raw)
    groups = _filter_into_variable_groups(raw, request, inventory)

    out_root.mkdir(parents=True, exist_ok=True)
    final_store = out_root / STORE_NAME
    tmp_store = out_root / f".{STORE_NAME}.tmp"
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    codec_mode = "uncompressed"
    try:
        _write_grouped_zarr(groups, tmp_store, codec_mode)
    except Exception as first_exc:
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        codec_mode = "blosc_lz4_clevel1_shuffle"
        try:
            _write_grouped_zarr(groups, tmp_store, codec_mode)
        except Exception as second_exc:  # pragma: no cover - preserves root cause context
            raise RuntimeError(f"Zarr v3 publication failed with uncompressed chunks ({first_exc!r}) and LZ4 fallback ({second_exc!r})") from second_exc

    _consolidate(tmp_store)
    _validate_publication(tmp_store, groups, codec_mode)

    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)

    channels = _artifact_channels(request)
    axis_group = sorted(groups)[0]
    warnings: list[str] = []
    if codec_mode != "uncompressed":
        warnings.append("Uncompressed Zarr v3 chunks were not accepted by the host stack; used lossless read-speed-biased Blosc LZ4 clevel=1 with byte shuffle.")

    return {
        "cache": {
            "hits": len(sources),
            "misses": 0,
            "acquired": 0,
            "reused_keys": [f"fixture:{s.entry_id}:{s.sha256[:16]}" for s in sources],
            "acquired_keys": [],
        },
        "dataset_artifact": {
            "schema_version": "dataset_artifact_layout.v1",
            "storage_format": "zarr",
            "store_path": STORE_NAME,
            "dimensions": {"sample": f"{axis_group}/{SAMPLE_DIM}", "y": f"{axis_group}/{Y_DIM}", "x": f"{axis_group}/{X_DIM}"},
            "coordinates": {"sample": f"{axis_group}/{SAMPLE_DIM}", "y": f"{axis_group}/{Y_DIM}", "x": f"{axis_group}/{X_DIM}"},
            "channels": channels,
        },
        "warnings": warnings,
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise ContractError("contract_lock must be a JSON object")
    if lock.get("schema_version") == "dataset_contract.v1":
        return lock
    for key in ("contract", "dataset_contract", "selected_contract", "runtime_contract"):
        val = lock.get(key)
        if isinstance(val, dict) and val.get("schema_version") == "dataset_contract.v1":
            return val
    contracts = lock.get("contracts")
    if isinstance(contracts, list):
        candidates = [c for c in contracts if isinstance(c, dict) and c.get("schema_version") == "dataset_contract.v1"]
        confirmed = [c for c in candidates if c.get("human_confirmed") is True]
        if len(confirmed) == 1:
            return confirmed[0]
        if len(candidates) == 1:
            return candidates[0]
    raise ContractError("could not locate a dataset_contract.v1 object in contract_lock")


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != "dataset_inventory.v1":
        raise ContractError("inventory schema_version must be dataset_inventory.v1")
    if contract.get("dataset_slug") != inventory.get("dataset_slug") or contract.get("dataset_slug") != DATASET_SLUG:
        raise ContractError("contract dataset_slug does not match this inventory adapter")
    if contract.get("human_confirmed") is not True:
        raise ContractError("contract must be human_confirmed")
    if inventory.get("dataset_id") != DATASET_ID:
        raise ContractError("unexpected inventory dataset_id")

    options = inventory.get("options", {})
    adv = contract.get("advanced_options") or {}
    if adv.get("dataset_id", DATASET_ID) != DATASET_ID:
        raise ContractError("advanced_options.dataset_id is not supported by this policy")
    for name in ("data_format", "download_format"):
        if name in adv and adv[name] not in options.get(name, []):
            raise ContractError(f"advanced option {name}={adv[name]!r} is not in inventory options")
    product_type = adv.get("product_type", contract.get("scope", {}).get("product_type", ["reanalysis"]))
    if isinstance(product_type, str):
        product_type = [product_type]
    if product_type != ["reanalysis"]:
        raise ContractError("this fixed regular-grid tensor policy supports product_type=['reanalysis'] only")

    scope = contract.get("scope") or {}
    date_range = scope.get("date_range") or {}
    start = _parse_date(date_range.get("start_date"), "start_date")
    end = _parse_date(date_range.get("end_date"), "end_date")
    if end < start:
        raise ContractError("end_date precedes start_date")
    if date_range.get("inclusive") is not True:
        raise ContractError("date_range.inclusive must be true for this adapter")
    for yr in sorted({f"{d.year:04d}" for d in _iter_dates(start, end)}):
        if yr not in options.get("year", []):
            raise ContractError(f"year {yr} is not available in inventory")

    time_scope = scope.get("time") or {}
    if time_scope.get("timezone", "UTC") != "UTC":
        raise ContractError("only UTC selected_times are supported")
    selected_times = time_scope.get("selected_times")
    if not isinstance(selected_times, list) or not selected_times:
        raise ContractError("scope.time.selected_times must be a non-empty list")
    inv_times = set(options.get("time", []))
    for t in selected_times:
        _parse_hhmm(t)
        if t not in inv_times:
            raise ContractError(f"selected time {t!r} is not available in inventory")

    geography = scope.get("geography") or {}
    area = geography.get("cds_area", inventory.get("defaults", {}).get("area"))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(v, (int, float)) for v in area)):
        raise ContractError("scope.geography.cds_area must be [north, west, south, east]")
    north, west, south, east = [float(v) for v in area]
    if not (-90.0 <= south <= north <= 90.0 and -360.0 <= west <= 360.0 and -360.0 <= east <= 360.0):
        raise ContractError("cds_area bounds are outside plausible latitude/longitude limits")

    fields = contract.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ContractError("contract.fields must be non-empty")
    requested: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in fields:
        name = field.get("name")
        if name not in options.get("variable", []):
            raise ContractError(f"field {name!r} is not an inventory variable")
        selectors = field.get("selectors") or []
        selector_map: dict[str, str] = {}
        for sel in selectors:
            dim = sel.get("dimension")
            val = str(sel.get("value"))
            if dim != SELECTOR_DIM:
                raise ContractError(f"unsupported selector dimension {dim!r}")
            if val not in options.get(SELECTOR_DIM, []):
                raise ContractError(f"pressure_level {val!r} is not in inventory options")
            if sel.get("unit") not in (None, "hPa"):
                raise ContractError("pressure_level selector unit must be hPa when supplied")
            selector_map[dim] = val
        if SELECTOR_DIM not in selector_map:
            raise ContractError("ERA5 pressure-level fields must include a pressure_level selector")
        fid = _field_id(name, selectors)
        if fid in seen:
            raise ContractError(f"duplicate requested channel {fid}")
        seen.add(fid)
        requested.append({"name": name, "selectors": selectors, "pressure_level": selector_map[SELECTOR_DIM], "field_id": fid})

    timestamps = _selected_timestamps(start, end, selected_times)
    return {"fields": requested, "timestamps": timestamps, "area": [north, west, south, east], "selected_times": selected_times}


def _verify_fixture(cache_root: Path) -> list[VerifiedSource]:
    manifest_path = cache_root / "source_fixture_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("complete local source fixture is required: cache_dir/source_fixture_manifest.json not found")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != "source_fixture_manifest.v1":
        raise ContractError("source fixture manifest schema_version mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ContractError("source fixture manifest must contain at least one entry")
    verified: list[VerifiedSource] = []
    for entry in entries:
        rel = entry.get("relative_path")
        if not isinstance(rel, str) or rel.startswith("/"):
            raise ContractError("fixture relative_path must be a relative string")
        path = (cache_root / rel).resolve()
        if not _is_relative_to(path, cache_root):
            raise ContractError("fixture path escapes cache_dir")
        expected_size = int(entry.get("size_bytes"))
        expected_sha = str(entry.get("sha256"))
        if expected_size <= 0 or len(expected_sha) != 64 or expected_sha.lower() != expected_sha:
            raise ContractError("fixture entry size_bytes/sha256 is invalid")
        stat = path.stat()
        if stat.st_size != expected_size:
            raise ContractError(f"fixture size mismatch for entry {entry.get('entry_id')!r}")
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ContractError(f"fixture sha256 mismatch for entry {entry.get('entry_id')!r}")
        verified.append(VerifiedSource(str(entry.get("entry_id", rel)), path, expected_size, actual_sha))
    return verified


def _open_sources(paths: list[Path]) -> xr.Dataset:
    datasets = [_normalise_source_dataset(_open_one_source(p)) for p in paths]
    try:
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.combine_by_coords(datasets, combine_attrs="override")
        return ds.load()
    finally:
        for ds0 in datasets:
            try:
                ds0.close()
            except Exception:
                pass


def _open_one_source(path: Path) -> xr.Dataset:
    suffixes = "".join(path.suffixes).lower()
    netcdf_like = suffixes.endswith((".nc", ".nc4", ".cdf", ".netcdf"))
    errors: list[str] = []
    engines = [None, "h5netcdf", "netcdf4"] if netcdf_like else ["cfgrib", None, "h5netcdf", "netcdf4"]
    for engine in engines:
        try:
            kwargs: dict[str, Any] = {"decode_cf": True, "mask_and_scale": True, "chunks": None}
            if engine is not None:
                kwargs["engine"] = engine
            if engine == "cfgrib":
                kwargs["backend_kwargs"] = {"indexpath": ""}
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}:{type(exc).__name__}")
    raise RuntimeError(f"could not open verified source fixture {path.name}; tried engines {errors}")


def _normalise_source_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for name in list(ds.dims) + list(ds.coords):
        if name in COORD_ALIASES and name != COORD_ALIASES[name] and COORD_ALIASES[name] not in ds:
            rename[name] = COORD_ALIASES[name]
    for short, long in SHORT_TO_LONG.items():
        if short in ds.data_vars and long not in ds.data_vars:
            rename[short] = long
    if rename:
        ds = ds.rename(rename)
    if SELECTOR_DIM in ds.coords and SELECTOR_DIM not in ds.dims:
        ds = ds.expand_dims(SELECTOR_DIM)
    required_coords = [SAMPLE_DIM, Y_DIM, X_DIM, SELECTOR_DIM]
    missing = [c for c in required_coords if c not in ds.coords and c not in ds.dims]
    if missing:
        raise ContractError(f"source fixture is missing required coordinates/dimensions: {missing}")
    # Avoid silent source packing propagation into public Zarr.
    for var in ds.variables.values():
        var.encoding = {}
    return ds


def _filter_into_variable_groups(ds: xr.Dataset, request: dict[str, Any], inventory: dict[str, Any]) -> dict[str, xr.Dataset]:
    timestamps = np.array(request["timestamps"], dtype="datetime64[ns]")
    source_times = ds[SAMPLE_DIM].values.astype("datetime64[ns]")
    missing_times = sorted(set(timestamps.tolist()) - set(source_times.tolist()))
    if missing_times:
        raise ContractError(f"source fixture does not contain all selected timestamps; missing {len(missing_times)}")
    ds = ds.sel({SAMPLE_DIM: timestamps})
    ds = _select_area(ds, request["area"])

    by_var: dict[str, list[str]] = {}
    for f in request["fields"]:
        by_var.setdefault(f["name"], [])
        if f["pressure_level"] not in by_var[f["name"]]:
            by_var[f["name"]].append(f["pressure_level"])

    groups: dict[str, xr.Dataset] = {}
    for var_name in sorted(by_var):
        if var_name not in ds.data_vars:
            raise ContractError(f"source fixture is missing requested variable {var_name!r}")
        levels_as_strings = by_var[var_name]
        level_values = _coerce_levels_for_source(ds[SELECTOR_DIM].values, levels_as_strings)
        # List-based sel preserves pressure_level as a real dimension even at cardinality one.
        sub = ds[[var_name]].sel({SELECTOR_DIM: level_values})
        da = sub[var_name]
        expected_dims = (SAMPLE_DIM, SELECTOR_DIM, Y_DIM, X_DIM)
        if set(da.dims) != set(expected_dims):
            raise ContractError(f"variable {var_name!r} dimensions {da.dims!r} do not match expected ERA5 pressure-level grid")
        da = da.transpose(*expected_dims)
        out = xr.Dataset({var_name: da}, coords={
            SAMPLE_DIM: sub[SAMPLE_DIM],
            SELECTOR_DIM: sub[SELECTOR_DIM],
            Y_DIM: sub[Y_DIM],
            X_DIM: sub[X_DIM],
        }, attrs={k: v for k, v in ds.attrs.items() if _jsonable(v)})
        meta = inventory.get("option_metadata", {}).get("variable", {}).get(var_name, {})
        out[var_name].attrs = {k: v for k, v in da.attrs.items() if _jsonable(v)}
        if "units" not in out[var_name].attrs and meta.get("units"):
            out[var_name].attrs["units"] = meta["units"]
        if "long_name" not in out[var_name].attrs and meta.get("label"):
            out[var_name].attrs["long_name"] = meta["label"]
        out[SELECTOR_DIM].attrs.setdefault("units", "hPa")
        out[SAMPLE_DIM].attrs.setdefault("standard_name", "time")
        out[Y_DIM].attrs.setdefault("units", "degrees_north")
        out[X_DIM].attrs.setdefault("units", "degrees_east")
        for v in out.variables.values():
            v.encoding = {}
        groups[var_name] = out.load()
    return groups


def _select_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    if [north, west, south, east] == [90.0, -180.0, -90.0, 180.0]:
        return ds
    lat = ds[Y_DIM]
    ds = ds.where((lat <= north) & (lat >= south), drop=True)
    lon = ds[X_DIM]
    lon_vals = lon.values
    if float(np.nanmin(lon_vals)) >= 0.0 and west < 0:
        west = west % 360.0
        east = east % 360.0
    if west <= east:
        mask = (lon >= west) & (lon <= east)
    else:
        mask = (lon >= west) | (lon <= east)
    ds = ds.where(mask, drop=True)
    if ds.sizes.get(Y_DIM, 0) == 0 or ds.sizes.get(X_DIM, 0) == 0:
        raise ContractError("geography selection produced an empty grid")
    return ds


def _write_grouped_zarr(groups: dict[str, xr.Dataset], store: Path, codec_mode: str) -> None:
    import zarr  # noqa: F401  # imported here to keep module import independent of store creation

    if store.exists():
        shutil.rmtree(store)
    first = True
    for group_name in sorted(groups):
        ds = groups[group_name]
        chunks = (1, 1, int(ds.sizes[Y_DIM]), int(ds.sizes[X_DIM]))
        encoding = _zarr_encoding(ds, group_name, chunks, codec_mode)
        ds.to_zarr(
            str(store),
            group=group_name,
            mode="w" if first else "a",
            zarr_format=3,
            consolidated=False,
            encoding=encoding,
            compute=True,
            safe_chunks=False,
        )
        first = False


def _zarr_encoding(ds: xr.Dataset, data_var: str, data_chunks: tuple[int, ...], codec_mode: str) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    for name, arr in ds.variables.items():
        e: dict[str, Any] = {}
        if not (np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64)):
            e["dtype"] = arr.dtype
        if name == data_var:
            e["chunks"] = data_chunks
        elif arr.ndim > 0:
            e["chunks"] = tuple(int(n) for n in arr.shape)
        if codec_mode == "uncompressed":
            # Zarr v3: None is an explicit request for no compression in maintained xarray/zarr stacks.
            e["compressors"] = None
        else:
            e["compressors"] = [_lz4_codec()]
        enc[name] = e
    return enc


def _lz4_codec() -> Any:
    try:
        from zarr.codecs import BloscCodec
        return BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")
    except Exception:
        from numcodecs import Blosc
        return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)


def _consolidate(store: Path) -> None:
    import zarr
    zarr.consolidate_metadata(str(store))
    meta = store / "zarr.json"
    if not meta.exists():
        raise RuntimeError("Zarr v3 root zarr.json was not created")
    text = meta.read_text(encoding="utf-8")
    if "consolidated_metadata" not in text:
        raise RuntimeError("Zarr consolidated metadata is missing from root metadata")


def _validate_publication(store: Path, groups: dict[str, xr.Dataset], codec_mode: str) -> None:
    import zarr
    root = zarr.open_group(str(store), mode="r")
    for group_name, expected in groups.items():
        reopened = xr.open_zarr(str(store), group=group_name, consolidated=True, zarr_format=3)
        try:
            xr.testing.assert_identical(reopened[group_name], expected[group_name])
            for coord in (SAMPLE_DIM, SELECTOR_DIM, Y_DIM, X_DIM):
                xr.testing.assert_identical(reopened[coord], expected[coord])
            arr = root[group_name][group_name]
            chunks = tuple(arr.chunks)
            wanted = (1, 1, int(expected.sizes[Y_DIM]), int(expected.sizes[X_DIM]))
            if chunks != wanted:
                raise RuntimeError(f"unexpected chunk layout for {group_name}: {chunks} != {wanted}")
            codecs = json.dumps(_codec_metadata(arr), sort_keys=True).lower()
            if codec_mode == "uncompressed":
                if "blosc" in codecs or "zstd" in codecs or "gzip" in codecs or "lz4" in codecs:
                    raise RuntimeError(f"data array {group_name} is compressed despite uncompressed publication mode")
            else:
                if "lz4" not in codecs or "clevel" not in codecs:
                    raise RuntimeError(f"data array {group_name} lacks declared LZ4 fallback codec")
        finally:
            reopened.close()


def _codec_metadata(arr: Any) -> Any:
    meta = getattr(arr, "metadata", None)
    if meta is not None and hasattr(meta, "codecs"):
        return [repr(c) for c in meta.codecs]
    if hasattr(arr, "compressor"):
        return repr(arr.compressor)
    return repr(arr)


def _artifact_channels(request: dict[str, Any]) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for f in request["fields"]:
        channels.append({
            "field_id": f["field_id"],
            "array_path": f"{f['name']}/{f['name']}",
            "selectors": {SELECTOR_DIM: f["pressure_level"]},
            "selector_coordinate_paths": {SELECTOR_DIM: f"{f['name']}/{SELECTOR_DIM}"},
        })
    return channels


def _field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    parts = [f"{sel['dimension']}={json.dumps(str(sel['value']), separators=(',', ':'))}" for sel in selectors]
    return f"{name}[{','.join(parts)}]"


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ContractError(f"{label} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError(f"{label} must be YYYY-MM-DD") from exc


def _parse_hhmm(value: Any) -> time:
    if not isinstance(value, str):
        raise ContractError("selected time must be HH:MM")
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ContractError(f"selected time {value!r} must be HH:MM") from exc


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _selected_timestamps(start: date, end: date, selected_times: list[str]) -> list[np.datetime64]:
    # Inclusive date-only end means the complete final UTC calendar day; then selected times are applied exactly.
    out: list[np.datetime64] = []
    parsed_times = [_parse_hhmm(t) for t in selected_times]
    for d in _iter_dates(start, end):
        for t in parsed_times:
            dt = datetime.combine(d, t, tzinfo=timezone.utc)
            out.append(np.datetime64(dt.replace(tzinfo=None), "ns"))
    return out


def _coerce_levels_for_source(source_values: np.ndarray, requested: list[str]) -> list[Any]:
    values = list(source_values.tolist())
    out: list[Any] = []
    for r in requested:
        candidates: list[Any] = [r]
        try:
            candidates.extend([int(r), float(r)])
        except ValueError:
            pass
        found = None
        for c in candidates:
            if c in values:
                found = c
                break
        if found is None:
            # numpy scalar equality can be finicky across dtypes; compare numerically as a final exact-value fallback.
            try:
                rv = float(r)
                for v in values:
                    if float(v) == rv:
                        found = v
                        break
            except Exception:
                pass
        if found is None:
            raise ContractError(f"source fixture is missing pressure_level {r}")
        out.append(found)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
