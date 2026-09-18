from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr

DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'dataset.zarr'
_TMP_STORE_NAME = '.tmp_dataset.zarr'
_VAR_ALIASES = {
    'temperature': ['temperature', 't'],
    'geopotential': ['geopotential', 'z'],
}
_DIM_ALIASES = {
    'time': ['time', 'valid_time'],
    'latitude': ['latitude', 'lat'],
    'longitude': ['longitude', 'lon'],
    'pressure_level': ['pressure_level', 'isobaricInhPa', 'level', 'plev'],
}
_BAD_CODEC_TOKENS = ('blosc', 'zstd', 'zstandard', 'lz4', 'gzip', 'zlib', 'deflate', 'bz2', 'compressor')


class PipelineError(ValueError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract = _extract_contract(contract_lock)
    lock_policy = _extract_policy(contract_lock)
    inv = _validate_inventory(inventory)
    if lock_policy is not None:
        _validate_policy(lock_policy)
    request = _validate_contract(contract, inv)

    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fixture_entries = _verify_fixture_manifest(cache_root)
    if not fixture_entries:
        raise PipelineError('A complete local source fixture is required; network acquisition is disabled by this adapter.')

    source = _open_source_dataset(cache_root, fixture_entries)
    source = _normalize_dataset(source)
    filtered, channels = _build_public_dataset(source, request, inv)

    tmp_store = output_root / _TMP_STORE_NAME
    final_store = output_root / STORE_NAME
    _safe_remove(tmp_store)
    _write_zarr_v3(filtered, tmp_store, channels)
    _validate_published_store(tmp_store, filtered, channels)
    if final_store.exists():
        _safe_remove(final_store)
    os.replace(tmp_store, final_store)
    _validate_published_store(final_store, filtered, channels)

    reused = [entry['entry_id'] for entry in fixture_entries]
    return {
        'cache': {
            'hits': len(fixture_entries),
            'misses': 0,
            'acquired': 0,
            'reused_keys': reused,
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': STORE_NAME,
            'dimensions': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'coordinates': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'channels': channels,
        },
        'warnings': [],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    preferred = ['contract', 'dataset_contract', 'selected_contract', 'runtime_contract']
    for key in preferred:
        val = lock.get(key) if isinstance(lock, dict) else None
        if isinstance(val, dict) and val.get('schema_version') == 'dataset_contract.v1':
            return val
    found: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get('schema_version') == 'dataset_contract.v1':
                found.append(obj)
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)

    walk(lock)
    if len(found) == 1:
        return found[0]
    raise PipelineError('contract_lock must contain exactly one dataset_contract.v1 object')


def _extract_policy(lock: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(lock, dict):
        return None
    for key in ['fixed_pipeline_policy', 'pipeline_policy', 'policy']:
        val = lock.get(key)
        if isinstance(val, dict) and val.get('publication_format') == 'zarr':
            return val
    return None


def _validate_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise PipelineError('inventory schema_version must be dataset_inventory.v1')
    if inventory.get('dataset_slug') != DATASET_SLUG:
        raise PipelineError('inventory dataset_slug mismatch')
    if inventory.get('dataset_id') != DATASET_ID:
        raise PipelineError('inventory dataset_id mismatch')
    return inventory


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get('provider') not in (None, 'ECMWF'):
        raise PipelineError('policy provider mismatch')
    if policy.get('dataset_id') != DATASET_ID:
        raise PipelineError('policy dataset_id mismatch')
    if policy.get('acquisition_format') != 'grib':
        raise PipelineError('policy acquisition_format must be grib')
    if policy.get('publication_format') != 'zarr':
        raise PipelineError('policy publication_format must be zarr')
    zpol = policy.get('zarr') or {}
    if zpol.get('format_version') != 3 or zpol.get('consolidated_metadata') is not True:
        raise PipelineError('policy must require consolidated Zarr v3')


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if contract.get('dataset_slug') != DATASET_SLUG:
        raise PipelineError('contract dataset_slug mismatch')
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id') != DATASET_ID:
        raise PipelineError('advanced_options.dataset_id mismatch')
    if adv.get('data_format') != 'grib':
        raise PipelineError('advanced_options.data_format must remain grib')
    if adv.get('download_format') != 'unarchived':
        raise PipelineError('advanced_options.download_format must be unarchived')
    if adv.get('product_type') != ['reanalysis']:
        raise PipelineError('advanced_options.product_type must be [reanalysis]')

    options = inventory.get('options') or {}
    fields = contract.get('fields') or []
    if not fields:
        raise PipelineError('contract must request at least one field')
    field_requests = []
    seen = set()
    for field in fields:
        name = field.get('name')
        if name not in options.get('variable', []):
            raise PipelineError(f'unsupported variable: {name}')
        selectors = field.get('selectors') or []
        if len(selectors) != 1 or selectors[0].get('dimension') != 'pressure_level':
            raise PipelineError('ERA5 pressure-level fields require exactly one pressure_level selector')
        val = str(selectors[0].get('value'))
        if val not in options.get('pressure_level', []):
            raise PipelineError(f'unsupported pressure_level: {val}')
        if selectors[0].get('unit') not in (None, 'hPa'):
            raise PipelineError('pressure_level selector unit must be hPa when provided')
        fid = _canonical_field_id(name, selectors)
        if fid in seen:
            raise PipelineError(f'duplicate requested channel: {fid}')
        seen.add(fid)
        field_requests.append({'name': name, 'pressure_level': val, 'selectors': selectors, 'field_id': fid})

    scope = contract.get('scope') or {}
    if scope.get('product_type') != 'reanalysis':
        raise PipelineError('scope.product_type must be reanalysis')
    time_scope = scope.get('time') or {}
    if time_scope.get('timezone') != 'UTC':
        raise PipelineError('scope.time.timezone must be UTC')
    selected_times = time_scope.get('selected_times') or []
    if not selected_times:
        raise PipelineError('at least one selected time is required')
    for t in selected_times:
        if t not in options.get('time', []):
            raise PipelineError(f'unsupported selected time: {t}')
    timestamps = _timestamps_from_scope(scope, options)

    geography = scope.get('geography') or {}
    area = geography.get('cds_area', inventory.get('defaults', {}).get('area'))
    if not isinstance(area, list) or len(area) != 4:
        raise PipelineError('geography.cds_area must have four numbers')
    area = [float(v) for v in area]
    north, west, south, east = area
    if not (-90 <= south <= north <= 90):
        raise PipelineError('invalid latitude bounds')
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise PipelineError('invalid longitude bounds')
    if geography.get('cds_area_order', ['north', 'west', 'south', 'east']) != ['north', 'west', 'south', 'east']:
        raise PipelineError('unsupported cds_area_order')

    return {'fields': field_requests, 'timestamps': timestamps, 'area': area}


def _timestamps_from_scope(scope: dict[str, Any], options: dict[str, Any]) -> pd.DatetimeIndex:
    dr = scope.get('date_range') or {}
    start = _parse_date(dr.get('start_date'), 'start_date')
    end = _parse_date(dr.get('end_date'), 'end_date')
    if end < start:
        raise PipelineError('end_date must not precede start_date')
    inclusive = bool(dr.get('inclusive', True))
    final = end if inclusive else end - timedelta(days=1)
    if final < start:
        raise PipelineError('exclusive date range selects no days')
    selected_times = scope.get('time', {}).get('selected_times') or []
    out = []
    cur = start
    while cur <= final:
        y, m, d = f'{cur.year:04d}', f'{cur.month:02d}', f'{cur.day:02d}'
        if y not in options.get('year', []) or m not in options.get('month', []) or d not in options.get('day', []):
            raise PipelineError(f'date outside inventory options: {cur.isoformat()}')
        for ts in selected_times:
            hh, mm = [int(p) for p in ts.split(':')]
            out.append(pd.Timestamp(datetime.combine(cur, time(hh, mm), tzinfo=timezone.utc)).tz_convert(None))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(out).sort_values()


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise PipelineError(f'{label} must be an ISO date string')
    return date.fromisoformat(value[:10])


def _canonical_field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    parts = []
    for sel in selectors:
        parts.append(sel['dimension'] + '=' + json.dumps(str(sel['value']), ensure_ascii=False, separators=(',', ':')))
    return name + '[' + ','.join(parts) + ']'


def _verify_fixture_manifest(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise PipelineError('invalid source fixture manifest schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise PipelineError('source fixture manifest has no entries')
    verified = []
    for entry in entries:
        rel = entry.get('relative_path')
        eid = entry.get('entry_id')
        if not isinstance(rel, str) or not rel or rel.startswith('/'):
            raise PipelineError('fixture entry relative_path must be relative')
        path = (cache_root / rel).resolve()
        if cache_root not in path.parents and path != cache_root:
            raise PipelineError('fixture entry escapes cache_dir')
        if not path.is_file():
            raise PipelineError('fixture raw file is missing')
        size = path.stat().st_size
        if size != int(entry.get('size_bytes')):
            raise PipelineError('fixture raw file size mismatch')
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry.get('sha256'):
            raise PipelineError('fixture raw file sha256 mismatch')
        if not isinstance(eid, str) or not eid:
            eid = 'sha256:' + sha[:16]
        verified.append({'entry_id': eid, 'relative_path': rel, 'sha256': sha})
    verified.sort(key=lambda e: e['entry_id'])
    return verified


def _open_source_dataset(cache_root: Path, entries: list[dict[str, Any]]) -> xr.Dataset:
    datasets = []
    for entry in entries:
        path = cache_root / entry['relative_path']
        suffix = path.suffix.lower()
        if suffix in ('.zarr',):
            ds = xr.open_zarr(path, consolidated=False, mask_and_scale=True)
        elif suffix in ('.grib', '.grb', '.grb2'):
            ds = xr.open_dataset(path, engine='cfgrib', decode_cf=True, mask_and_scale=True)
        else:
            try:
                ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
            except Exception:
                ds = xr.open_dataset(path, engine='h5netcdf', decode_cf=True, mask_and_scale=True)
        datasets.append(ds.load())
        ds.close()
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat='override', join='outer')


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    for canonical, aliases in _DIM_ALIASES.items():
        if canonical in ds.dims or canonical in ds.coords:
            continue
        for alias in aliases:
            if alias in ds.dims or alias in ds.coords:
                renames[alias] = canonical
                break
    if renames:
        ds = ds.rename(renames)
    required = ['time', 'latitude', 'longitude', 'pressure_level']
    missing = [d for d in required if d not in ds.coords and d not in ds.dims]
    if missing:
        raise PipelineError('source fixture is missing required coordinates: ' + ', '.join(missing))
    if 'time' not in ds.indexes:
        ds = ds.assign_coords(time=pd.to_datetime(ds['time'].values))
    return ds


def _build_public_dataset(source: xr.Dataset, request: dict[str, Any], inventory: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    src = _select_time_and_area(source, request['timestamps'], request['area'])
    data_vars = {}
    coords: dict[str, Any] = {
        'time': src['time'],
        'latitude': src['latitude'],
        'longitude': src['longitude'],
    }
    channels = []
    for req in request['fields']:
        source_name = _find_source_var(src, req['name'], req['pressure_level'])
        arr = src[source_name]
        arr = _select_pressure(arr, req['pressure_level'])
        selector_dim = _selector_dim_name(req['field_id'])
        native_level = arr['pressure_level'].values
        if np.ndim(native_level) == 0:
            native_level = np.asarray([native_level.item()])
        arr = arr.expand_dims({selector_dim: native_level[:1]})
        arr = arr.transpose('time', selector_dim, 'latitude', 'longitude')
        public_name = _array_name(req['field_id'])
        attrs = dict(arr.attrs)
        meta = (inventory.get('option_metadata') or {}).get('variable', {}).get(req['name'], {})
        if meta.get('units') and 'units' not in attrs:
            attrs['units'] = meta['units']
        if meta.get('label') and 'long_name' not in attrs:
            attrs['long_name'] = meta['label']
        attrs['source_variable'] = req['name']
        attrs['pressure_level'] = req['pressure_level']
        arr.attrs = attrs
        arr.encoding = {}
        coords[selector_dim] = xr.DataArray(native_level[:1], dims=(selector_dim,), attrs={'units': 'hPa', 'selector_dimension': 'pressure_level'})
        data_vars[public_name] = arr.rename(public_name)
        channels.append({
            'field_id': req['field_id'],
            'array_path': public_name,
            'selectors': {'pressure_level': req['pressure_level']},
            'selector_coordinate_paths': {'pressure_level': selector_dim},
        })
    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=_safe_global_attrs(source.attrs))
    for name in out.variables:
        out[name].encoding = {}
    return out, channels


def _select_time_and_area(ds: xr.Dataset, timestamps: pd.DatetimeIndex, area: list[float]) -> xr.Dataset:
    available = pd.DatetimeIndex(pd.to_datetime(ds['time'].values)).tz_localize(None)
    wanted = pd.DatetimeIndex(timestamps).tz_localize(None)
    missing = wanted.difference(available)
    if len(missing):
        raise PipelineError('source fixture is missing requested timestamps')
    ds = ds.sel(time=wanted)
    north, west, south, east = area
    lat = ds['latitude']
    ds = ds.where((lat >= south) & (lat <= north), drop=True)
    lon = ds['longitude']
    if not (abs(west + 180) < 1e-9 and abs(east - 180) < 1e-9):
        vals = lon.values.astype(float)
        if np.nanmin(vals) >= 0:
            w = west % 360
            e = east % 360
            mask = (lon >= w) & (lon <= e) if w <= e else ((lon >= w) | (lon <= e))
        else:
            mask = (lon >= west) & (lon <= east) if west <= east else ((lon >= west) | (lon <= east))
        ds = ds.where(mask, drop=True)
    if ds.sizes.get('latitude', 0) == 0 or ds.sizes.get('longitude', 0) == 0:
        raise PipelineError('geography selection produced an empty grid')
    return ds


def _find_source_var(ds: xr.Dataset, requested: str, pressure_level: str | None = None) -> str:
    aliases = _VAR_ALIASES.get(requested, [requested])
    matches: list[str] = []
    for name in aliases:
        if name in ds.data_vars and name not in matches:
            matches.append(name)
    lower_aliases = {a.lower() for a in aliases}
    for name, var in ds.data_vars.items():
        candidates = {name.lower()}
        for attr in ['standard_name', 'long_name', 'GRIB_shortName', 'GRIB_cfVarName']:
            val = var.attrs.get(attr)
            if isinstance(val, str):
                candidates.add(val.lower())
        if candidates & lower_aliases and name not in matches:
            matches.append(name)
    if pressure_level is not None:
        for name in matches:
            if _var_pressure_matches(ds[name], pressure_level):
                return name
    if matches:
        return matches[0]
    raise PipelineError(f'source fixture lacks requested variable: {requested}')


def _select_pressure(arr: xr.DataArray, level: str) -> xr.DataArray:
    if 'pressure_level' not in arr.coords and 'pressure_level' not in arr.dims:
        for alias in _DIM_ALIASES['pressure_level']:
            if alias != 'pressure_level' and (alias in arr.coords or alias in arr.dims):
                arr = arr.rename({alias: 'pressure_level'})
                break
    if 'pressure_level' not in arr.coords and 'pressure_level' not in arr.dims:
        attr_level = _pressure_level_from_attrs(arr)
        if attr_level is not None and _pressure_value_to_string(attr_level) == level:
            return arr.assign_coords(pressure_level=_pressure_coord_value(attr_level))
        raise PipelineError('requested variable lacks pressure_level coordinate')
    vals = arr['pressure_level'].values
    if 'pressure_level' not in arr.dims:
        if _pressure_values_match(vals, level):
            return arr
        raise PipelineError(f'source fixture lacks pressure_level {level}')
    matches = [i for i, v in enumerate(np.ravel(vals)) if _pressure_value_to_string(v) == level]
    if not matches:
        raise PipelineError(f'source fixture lacks pressure_level {level}')
    return arr.isel(pressure_level=matches[0])


def _var_pressure_matches(arr: xr.DataArray, level: str) -> bool:
    if 'pressure_level' in arr.coords or 'pressure_level' in arr.dims:
        return _pressure_values_match(arr['pressure_level'].values, level)
    for alias in _DIM_ALIASES['pressure_level']:
        if alias != 'pressure_level' and (alias in arr.coords or alias in arr.dims):
            return _pressure_values_match(arr[alias].values, level)
    attr_level = _pressure_level_from_attrs(arr)
    return attr_level is not None and _pressure_value_to_string(attr_level) == level


def _pressure_values_match(values: Any, level: str) -> bool:
    return any(_pressure_value_to_string(v) == level for v in np.ravel(np.asarray(values)))


def _pressure_level_from_attrs(arr: xr.DataArray) -> Any:
    for key in ['pressure_level', 'isobaricInhPa', 'level', 'plev', 'GRIB_level']:
        if key in arr.attrs:
            return arr.attrs[key]
    return None


def _pressure_value_to_string(value: Any) -> str:
    try:
        if np.issubdtype(np.asarray(value).dtype, np.number):
            numeric = float(value)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
            return str(numeric)
    except (TypeError, ValueError):
        pass
    return str(value)


def _pressure_coord_value(value: Any) -> Any:
    text = _pressure_value_to_string(value)
    try:
        return np.int32(int(text))
    except ValueError:
        return text


def _selector_dim_name(field_id: str) -> str:
    return 'pressure_level__' + _slug(field_id)


def _array_name(field_id: str) -> str:
    return 'channel__' + _slug(field_id)


def _slug(text: str) -> str:
    return re.sub('[^A-Za-z0-9_]+', '_', text).strip('_')[:180]


def _safe_global_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, val in attrs.items():
        s = str(val)
        if '://' in s or 'token' in key.lower() or 'secret' in key.lower() or 'key' == key.lower():
            continue
        if isinstance(val, (str, int, float, np.integer, np.floating)):
            out[key] = val.item() if hasattr(val, 'item') else val
    out['dataset_id'] = DATASET_ID
    out['publication_format'] = 'zarr_v3_consolidated_uncompressed_data_chunks'
    return out


def _write_zarr_v3(ds: xr.Dataset, store: Path, channels: list[dict[str, Any]]) -> None:
    y = int(ds.sizes['latitude'])
    x = int(ds.sizes['longitude'])
    encoding: dict[str, dict[str, Any]] = {}
    channel_arrays = {c['array_path'] for c in channels}
    for name in ds.variables:
        if name in channel_arrays:
            sel_dim = [d for d in ds[name].dims if d.startswith('pressure_level__')][0]
            encoding[name] = {'chunks': (1, 1, y, x), 'compressors': []}
        else:
            shape = tuple(max(1, int(ds.sizes.get(d, ds[name].sizes.get(d, 1)))) for d in ds[name].dims)
            encoding[name] = {'chunks': shape, 'compressors': []}
    try:
        ds.to_zarr(store, mode='w', zarr_format=3, consolidated=True, encoding=encoding, compute=True)
    except TypeError:
        for enc in encoding.values():
            enc.pop('compressors', None)
            enc['compressor'] = None
        ds.to_zarr(store, mode='w', zarr_format=3, consolidated=True, encoding=encoding, compute=True)
    try:
        zarr.consolidate_metadata(store)
    except Exception:
        pass


def _validate_published_store(store: Path, expected: xr.Dataset, channels: list[dict[str, Any]]) -> None:
    if not (store / 'zarr.json').exists():
        raise PipelineError('published store is not Zarr v3')
    root_meta = json.loads((store / 'zarr.json').read_text(encoding='utf-8'))
    if root_meta.get('zarr_format') != 3:
        raise PipelineError('published store zarr_format is not 3')
    if 'consolidated_metadata' not in root_meta:
        raise PipelineError('published store lacks consolidated metadata')
    group = zarr.open_group(store, mode='r')
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3, mask_and_scale=True).load()
    try:
        if set(reopened.data_vars) != set(expected.data_vars):
            raise PipelineError('reopened data variables differ from expected output')
        for coord in ['time', 'latitude', 'longitude']:
            if coord not in reopened.coords or not np.array_equal(reopened[coord].values, expected[coord].values):
                raise PipelineError(f'reopened coordinate mismatch: {coord}')
        for channel in channels:
            name = channel['array_path']
            arr = group[name]
            exp_chunks = tuple(expected[name].encoding.get('chunks') or (1, 1, expected.sizes['latitude'], expected.sizes['longitude']))
            if tuple(arr.chunks) != exp_chunks:
                raise PipelineError(f'chunk shape mismatch for {name}: {arr.chunks} != {exp_chunks}')
            codecs = _codec_names(arr)
            bad = [c for c in codecs if any(tok in c.lower() for tok in _BAD_CODEC_TOKENS)]
            if bad:
                raise PipelineError(f'compression codec found for {name}: {bad}')
            if not np.array_equal(reopened[name].values, expected[name].values, equal_nan=True):
                raise PipelineError(f'reopened data mismatch for {name}')
            sel_path = channel['selector_coordinate_paths']['pressure_level']
            if sel_path not in reopened.coords:
                raise PipelineError('selector coordinate missing after reopen')
            if reopened[name].dims[1] != sel_path or reopened.sizes[sel_path] != 1:
                raise PipelineError('pressure-level selector is not a real one-element dimension')
    finally:
        reopened.close()


def _codec_names(arr: Any) -> list[str]:
    meta = getattr(arr, 'metadata', None)
    codecs = getattr(meta, 'codecs', []) if meta is not None else []
    names = []
    for codec in codecs:
        if isinstance(codec, dict):
            names.append(str(codec.get('name') or codec.get('id') or codec))
        else:
            names.append(codec.__class__.__name__)
            name = getattr(codec, 'name', None)
            if name:
                names.append(str(name))
    return names


def _safe_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
