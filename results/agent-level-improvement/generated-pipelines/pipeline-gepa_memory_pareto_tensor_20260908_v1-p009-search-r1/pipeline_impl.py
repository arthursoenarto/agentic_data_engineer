from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p009-search-r1'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'dataset.zarr'

VAR_ALIASES = {
    'temperature': ('temperature', 't'),
    'geopotential': ('geopotential', 'z'),
}
DIM_ALIASES = {
    'time': ('time', 'valid_time'),
    'pressure_level': ('pressure_level', 'isobaricInhPa', 'level', 'plev'),
    'latitude': ('latitude', 'lat'),
    'longitude': ('longitude', 'lon'),
}


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract = _extract_contract(contract_lock)
    _validate_inventory(inventory)
    requested = _validate_contract(contract, inventory)

    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fixture = _verify_fixture_manifest(cache_root)
    if fixture is None:
        raise RuntimeError('No complete local source fixture was found; this adapter does not perform network access or credential probing.')

    source = _open_sources(fixture['paths'])
    source = _canonicalize_dataset(source)
    filtered = _filter_dataset(source, requested, inventory)
    published = _publish_zarr(filtered, requested, out_root)
    _validate_published(published, filtered, requested)

    return {
        'cache': {
            'hits': len(fixture['entries']),
            'misses': 0,
            'acquired': 0,
            'reused_keys': fixture['keys'],
            'acquired_keys': [],
        },
        'dataset_artifact': _artifact(requested),
        'warnings': [],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise ValueError('contract_lock must be a JSON object')
    if lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    for key in ('contract', 'dataset_contract', 'runtime_contract'):
        val = lock.get(key)
        if isinstance(val, dict) and val.get('schema_version') == 'dataset_contract.v1':
            return val
    for key in ('lock', 'runtime', 'payload'):
        val = lock.get(key)
        if isinstance(val, dict):
            try:
                return _extract_contract(val)
            except ValueError:
                pass
    raise ValueError('dataset_contract.v1 payload not found in contract_lock')


def _validate_inventory(inv: dict[str, Any]) -> None:
    if inv.get('schema_version') != 'dataset_inventory.v1':
        raise ValueError('unsupported inventory schema_version')
    if inv.get('dataset_slug') != DATASET_SLUG or inv.get('dataset_id') != DATASET_ID:
        raise ValueError('inventory is not the frozen ERA5 pressure-level inventory')


def _option(inv: dict[str, Any], name: str) -> set[str]:
    return {str(x) for x in inv.get('options', {}).get(name, [])}


def _validate_contract(contract: dict[str, Any], inv: dict[str, Any]) -> dict[str, Any]:
    if contract.get('schema_version') != 'dataset_contract.v1':
        raise ValueError('unsupported contract schema_version')
    if contract.get('dataset_slug') != DATASET_SLUG:
        raise ValueError('contract dataset_slug does not match inventory')
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ValueError('advanced_options.dataset_id is invalid')
    if adv.get('data_format', 'grib') != 'grib':
        raise ValueError('fixed policy requires GRIB acquisition')
    if adv.get('download_format', 'unarchived') not in _option(inv, 'download_format'):
        raise ValueError('invalid download_format')
    product_type = contract.get('scope', {}).get('product_type', 'reanalysis')
    if product_type not in _option(inv, 'product_type'):
        raise ValueError('invalid product_type')
    if product_type != 'reanalysis':
        raise ValueError('fixed policy for this adapter supports product_type=reanalysis only')

    selected_times = contract.get('scope', {}).get('time', {}).get('selected_times') or []
    if not selected_times:
        raise ValueError('scope.time.selected_times is required')
    valid_times = _option(inv, 'time')
    for t in selected_times:
        if t not in valid_times or not re.fullmatch(r'\d{2}:\d{2}', t):
            raise ValueError(f'invalid selected time: {t}')

    dr = contract.get('scope', {}).get('date_range') or {}
    start = _parse_date(dr.get('start_date'), 'start_date')
    end = _parse_date(dr.get('end_date'), 'end_date')
    if end < start:
        raise ValueError('end_date precedes start_date')
    years = _option(inv, 'year')
    months = _option(inv, 'month')
    days = _option(inv, 'day')
    cur = start
    while cur <= end:
        if f'{cur.year:04d}' not in years or f'{cur.month:02d}' not in months or f'{cur.day:02d}' not in days:
            raise ValueError(f'date outside inventory options: {cur.isoformat()}')
        cur = date.fromordinal(cur.toordinal() + 1)

    geography = contract.get('scope', {}).get('geography') or {}
    area = geography.get('cds_area', inv.get('defaults', {}).get('area'))
    if not isinstance(area, list) or len(area) != 4:
        raise ValueError('geography.cds_area must be [north, west, south, east]')
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError('geography.cds_area is outside valid latitude/longitude bounds')
    if geography.get('cds_area_order', ['north', 'west', 'south', 'east']) != ['north', 'west', 'south', 'east']:
        raise ValueError('unsupported cds_area_order')

    fields = contract.get('fields') or []
    if not fields:
        raise ValueError('at least one field is required')
    channels = []
    by_variable: dict[str, list[str]] = {}
    for fld in fields:
        name = fld.get('name')
        if name not in _option(inv, 'variable'):
            raise ValueError(f'invalid variable: {name}')
        selectors = fld.get('selectors') or []
        if len(selectors) != 1 or selectors[0].get('dimension') != 'pressure_level':
            raise ValueError('ERA5 pressure-level fields require exactly one pressure_level selector')
        level = str(selectors[0].get('value'))
        if level not in _option(inv, 'pressure_level'):
            raise ValueError(f'invalid pressure_level: {level}')
        if selectors[0].get('unit') not in (None, 'hPa'):
            raise ValueError('pressure_level selector unit must be hPa when supplied')
        fid = _field_id(name, selectors)
        channels.append({'field_id': fid, 'name': name, 'pressure_level': level, 'selectors': selectors})
        by_variable.setdefault(name, [])
        if level not in by_variable[name]:
            by_variable[name].append(level)

    return {
        'start_date': start,
        'end_date': end,
        'inclusive': bool(dr.get('inclusive', True)),
        'selected_times': list(selected_times),
        'area': [north, west, south, east],
        'channels': channels,
        'by_variable': by_variable,
    }


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f'{label} is required')
    return date.fromisoformat(value)


def _field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    parts = []
    for sel in selectors:
        parts.append(f"{sel['dimension']}={json.dumps(str(sel['value']), ensure_ascii=False, separators=(',', ':'))}")
    return f"{name}[{','.join(parts)}]"


def _safe_array_path(field_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', field_id).strip('_')


def _verify_fixture_manifest(cache_root: Path) -> dict[str, Any] | None:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ValueError('invalid source fixture manifest schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise ValueError('source fixture manifest has no entries')
    paths = []
    keys = []
    for ent in entries:
        rel = ent.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/') or '..' in Path(rel).parts:
            raise ValueError('unsafe fixture relative_path')
        path = (cache_root / rel).resolve()
        if cache_root not in path.parents and path != cache_root:
            raise ValueError('fixture path escapes cache_dir')
        data = path.read_bytes()
        size = int(ent.get('size_bytes'))
        sha = str(ent.get('sha256'))
        got = hashlib.sha256(data).hexdigest()
        if len(data) != size or got != sha:
            raise ValueError(f'fixture verification failed for entry_id={ent.get("entry_id", "<unknown>")}')
        paths.append(path)
        keys.append(f"fixture:{ent.get('entry_id','source')}:{got[:16]}")
    return {'entries': entries, 'paths': paths, 'keys': keys}


def _open_sources(paths: list[Path]) -> xr.Dataset:
    datasets = [_open_one(p) for p in sorted(paths, key=lambda p: str(p))]
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')


def _open_one(path: Path) -> xr.Dataset:
    suffix = path.suffix.lower()
    if suffix in ('.grib', '.grb', '.grib2', '.grb2'):
        try:
            return xr.open_dataset(path, engine='cfgrib', decode_cf=True, mask_and_scale=True).load()
        except Exception as exc:
            raise RuntimeError('failed to decode local GRIB fixture with cfgrib') from exc
    if suffix in ('.nc', '.nc4', '.cdf'):
        return xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
    if suffix == '.zarr':
        return xr.open_zarr(path, consolidated=False).load()
    try:
        return xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
    except Exception as exc:
        raise RuntimeError(f'unsupported local fixture format for {path.name}') from exc


def _canonicalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for canon, aliases in DIM_ALIASES.items():
        for a in aliases:
            if a in ds.dims or a in ds.coords:
                rename[a] = canon
                break
    for canon, aliases in VAR_ALIASES.items():
        if canon in ds.data_vars:
            continue
        for a in aliases:
            if a in ds.data_vars:
                rename[a] = canon
                break
    ds = ds.rename({k: v for k, v in rename.items() if k != v})
    required = ('time', 'latitude', 'longitude')
    for dim in required:
        if dim not in ds.coords and dim not in ds.dims:
            raise ValueError(f'source fixture is missing {dim}')
    return ds


def _wanted_datetimes(req: dict[str, Any]) -> np.ndarray:
    days = []
    cur = req['start_date']
    end = req['end_date']
    last = end if req['inclusive'] else date.fromordinal(end.toordinal() - 1)
    while cur <= last:
        for hhmm in req['selected_times']:
            hh, mm = [int(x) for x in hhmm.split(':')]
            days.append(np.datetime64(datetime.combine(cur, time(hh, mm), tzinfo=timezone.utc).replace(tzinfo=None), 'ns'))
        cur = date.fromordinal(cur.toordinal() + 1)
    return np.array(days, dtype='datetime64[ns]')


def _filter_dataset(ds: xr.Dataset, req: dict[str, Any], inv: dict[str, Any]) -> xr.Dataset:
    wanted_time = _wanted_datetimes(req)
    source_time = ds['time'].values.astype('datetime64[ns]')
    missing = [str(t) for t in wanted_time if t not in set(source_time)]
    if missing:
        raise ValueError('source fixture is missing requested timestamps: ' + ', '.join(missing[:5]))
    ds = ds.sel(time=wanted_time)

    north, west, south, east = req['area']
    lat = ds['latitude']
    lat_mask = (lat >= south) & (lat <= north)
    ds = ds.sel(latitude=lat[lat_mask])
    lon = ds['longitude']
    if not (west <= -180 and east >= 180):
        vals = lon.values.astype(float)
        if vals.min() >= 0 and west < 0:
            w = west % 360
            e = east % 360
            mask = (vals >= w) | (vals <= e) if w > e else (vals >= w) & (vals <= e)
        else:
            mask = (vals >= west) & (vals <= east) if west <= east else (vals >= west) | (vals <= east)
        ds = ds.sel(longitude=lon[mask])

    out_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        'time': ds['time'],
        'latitude': ds['latitude'],
        'longitude': ds['longitude'],
    }
    requested_levels_all: list[str] = []
    for levels in req['by_variable'].values():
        for lev in levels:
            if lev not in requested_levels_all:
                requested_levels_all.append(lev)

    for var_name, levels in req['by_variable'].items():
        if var_name not in ds.data_vars:
            raise ValueError(f'source fixture is missing requested variable {var_name}')
        da = ds[var_name]
        plevel = ds['pressure_level']
        selected_coord_values = []
        for lev in levels:
            selected_coord_values.append(_match_level_value(plevel.values, lev))
        if 'pressure_level' not in da.dims:
            if len(selected_coord_values) != 1:
                raise ValueError(f'source variable {var_name} is missing pressure_level dimension')
            da = da.expand_dims(pressure_level=selected_coord_values)
        da = da.sel(pressure_level=selected_coord_values)
        da = da.transpose('time', 'pressure_level', 'latitude', 'longitude')
        da.encoding = {}
        if 'units' not in da.attrs:
            meta = inv.get('option_metadata', {}).get('variable', {}).get(var_name, {})
            if meta.get('units'):
                da.attrs['units'] = meta['units']
        out_vars[var_name] = da
    coords['pressure_level'] = sorted({v.item() if hasattr(v, 'item') else v for da in out_vars.values() for v in da['pressure_level'].values}, key=lambda x: float(x))
    out = xr.Dataset(out_vars, attrs=dict(ds.attrs), coords={k: v for k, v in coords.items() if k != 'pressure_level'})
    out.attrs.update({'dataset_id': DATASET_ID, 'publication_format': 'zarr_v3', 'pipeline_id': PIPELINE_ID})
    return out


def _match_level_value(values: np.ndarray, requested: str) -> Any:
    for val in values:
        if str(val) == requested:
            return val
        try:
            if float(val) == float(requested):
                return val
        except Exception:
            pass
    raise ValueError(f'source fixture is missing pressure_level={requested}')


def _encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    lat_n = int(ds.sizes['latitude'])
    lon_n = int(ds.sizes['longitude'])
    for name, da in ds.data_vars.items():
        if 'pressure_level' in da.dims:
            chunks = tuple(1 if d in ('time', 'pressure_level') else lat_n if d == 'latitude' else lon_n for d in da.dims)
        else:
            chunks = tuple(1 if d == 'time' else lat_n if d == 'latitude' else lon_n for d in da.dims)
        enc[name] = {'chunks': chunks, 'compressors': None}
    for name in ds.coords:
        if name in ds.sizes:
            enc[name] = {'chunks': (int(ds.sizes[name]),), 'compressors': None}
    return enc


def _publish_zarr(ds: xr.Dataset, req: dict[str, Any], out_root: Path) -> Path:
    tmp = out_root / ('.tmp_' + STORE_NAME + '_' + hashlib.sha256(json.dumps(req, default=str, sort_keys=True).encode()).hexdigest()[:12])
    dest = out_root / STORE_NAME
    backup = out_root / ('.old_' + STORE_NAME)
    for p in (tmp, backup):
        if p.exists():
            shutil.rmtree(p)
    enc = _encoding(ds)
    try:
        ds.to_zarr(tmp, mode='w', zarr_format=3, consolidated=True, encoding=enc)
    except TypeError:
        if tmp.exists():
            shutil.rmtree(tmp)
        enc2 = {k: {('compressor' if kk == 'compressors' else kk): vv for kk, vv in v.items()} for k, v in enc.items()}
        ds.to_zarr(tmp, mode='w', zarr_format=3, consolidated=True, encoding=enc2)
    if dest.exists():
        dest.rename(backup)
    tmp.rename(dest)
    if backup.exists():
        shutil.rmtree(backup)
    return dest


def _validate_published(store: Path, expected: xr.Dataset, req: dict[str, Any]) -> None:
    meta = json.loads((store / 'zarr.json').read_text())
    if meta.get('zarr_format') != 3:
        raise AssertionError('published store is not Zarr v3')
    if 'consolidated_metadata' not in meta:
        raise AssertionError('published store is missing consolidated metadata')
    reopened = xr.open_zarr(store, consolidated=True).load()
    if set(reopened.data_vars) != set(expected.data_vars):
        raise AssertionError('data variable names changed during publication')
    for coord in ('time', 'pressure_level', 'latitude', 'longitude'):
        if coord in expected.coords:
            np.testing.assert_array_equal(reopened[coord].values, expected[coord].values)
    for name in expected.data_vars:
        if reopened[name].dims != expected[name].dims:
            raise AssertionError(f'dimensions changed for {name}')
        np.testing.assert_array_equal(reopened[name].values, expected[name].values)
        _assert_no_compressor_and_chunks(store, name, expected[name])


def _assert_no_compressor_and_chunks(store: Path, name: str, da: xr.DataArray) -> None:
    arr = zarr.open_array(str(store / name), mode='r')
    lat_n = int(da.sizes['latitude'])
    lon_n = int(da.sizes['longitude'])
    wanted = (1, 1, lat_n, lon_n) if 'pressure_level' in da.dims else (1, lat_n, lon_n)
    if tuple(arr.chunks) != wanted:
        raise AssertionError(f'unexpected chunks for {name}: {arr.chunks} != {wanted}')
    codecs = getattr(arr.metadata, 'codecs', [])
    text = json.dumps([repr(c).lower() for c in codecs])
    forbidden = ('blosc', 'zstd', 'gzip', 'lz4', 'zlib', 'bz2', 'compressor')
    if any(x in text for x in forbidden):
        raise AssertionError(f'data array {name} has a data chunk compressor: {text}')


def _artifact(req: dict[str, Any]) -> dict[str, Any]:
    channels = []
    for ch in req['channels']:
        channels.append({
            'field_id': ch['field_id'],
            'array_path': ch['name'],
            'selectors': {'pressure_level': ch['pressure_level']},
            'selector_coordinate_paths': {'pressure_level': 'pressure_level'},
        })
    return {
        'schema_version': 'dataset_artifact_layout.v1',
        'storage_format': 'zarr',
        'store_path': STORE_NAME,
        'dimensions': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
        'coordinates': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
        'channels': channels,
    }
