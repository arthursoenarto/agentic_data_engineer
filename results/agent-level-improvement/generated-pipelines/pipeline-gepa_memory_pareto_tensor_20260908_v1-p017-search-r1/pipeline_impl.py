from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p017-search-r1'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'era5_pressure_levels.zarr'

VAR_ALIASES = {
    'temperature': ['temperature', 't'],
    'geopotential': ['geopotential', 'z'],
    'u_component_of_wind': ['u_component_of_wind', 'u'],
    'v_component_of_wind': ['v_component_of_wind', 'v'],
    'relative_humidity': ['relative_humidity', 'r'],
    'specific_humidity': ['specific_humidity', 'q'],
    'vertical_velocity': ['vertical_velocity', 'w'],
    'vorticity': ['vorticity', 'vo'],
    'divergence': ['divergence', 'd'],
    'ozone_mass_mixing_ratio': ['ozone_mass_mixing_ratio', 'o3'],
    'potential_vorticity': ['potential_vorticity', 'pv'],
    'fraction_of_cloud_cover': ['fraction_of_cloud_cover', 'cc'],
    'specific_cloud_ice_water_content': ['specific_cloud_ice_water_content', 'ciwc'],
    'specific_cloud_liquid_water_content': ['specific_cloud_liquid_water_content', 'clwc'],
    'specific_rain_water_content': ['specific_rain_water_content', 'crwc'],
    'specific_snow_water_content': ['specific_snow_water_content', 'cswc'],
}

COORD_ALIASES = {
    'time': ['time', 'valid_time'],
    'pressure_level': ['pressure_level', 'isobaricInhPa', 'level', 'plev'],
    'latitude': ['latitude', 'lat'],
    'longitude': ['longitude', 'lon'],
}

@dataclass(frozen=True)
class Channel:
    field_name: str
    selectors: tuple[tuple[str, str, str | None], ...]
    field_id: str
    array_path: str
    selector_dim: str | None


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract = _select_contract(contract_lock)
    _validate_inventory_and_contract(contract, inventory)
    channels = _channels_from_contract(contract, inventory)
    wanted_times = _selected_datetimes(contract)
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fixture_entries, source_paths = _verify_source_fixture(cache_root)
    source = _open_source_dataset(source_paths)
    source = _normalise_source_dataset(source)
    filtered = _filter_source_dataset(source, contract, wanted_times)
    published = _build_public_dataset(filtered, channels, inventory)

    final_store = out_root / STORE_NAME
    _publish_zarr_atomic(published, final_store, channels)
    reopened = xr.open_zarr(final_store, consolidated=True)
    try:
        _validate_reopened(reopened, published, channels)
    finally:
        reopened.close()

    reused_keys = [_safe_fixture_key(e) for e in fixture_entries]
    return {
        'cache': {
            'hits': len(fixture_entries),
            'misses': 0,
            'acquired': 0,
            'reused_keys': reused_keys,
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': STORE_NAME,
            'dimensions': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'coordinates': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'channels': [
                {
                    'field_id': c.field_id,
                    'array_path': c.array_path,
                    'selectors': {d: v for d, v, _u in c.selectors},
                    'selector_coordinate_paths': ({'pressure_level': c.selector_dim} if c.selector_dim else {}),
                }
                for c in channels
            ],
        },
        'warnings': [],
    }


def _select_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lock, dict):
        raise ValueError('contract_lock must be a JSON object')
    if lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    for key in ('contract', 'dataset_contract', 'selected_contract'):
        obj = lock.get(key)
        if isinstance(obj, dict) and obj.get('schema_version') == 'dataset_contract.v1':
            return obj
    nested = lock.get('lock')
    if isinstance(nested, dict):
        return _select_contract(nested)
    contracts = lock.get('contracts')
    if isinstance(contracts, list):
        matches = [c for c in contracts if isinstance(c, dict) and c.get('schema_version') == 'dataset_contract.v1']
        if len(matches) == 1:
            return matches[0]
    raise ValueError('could not locate a dataset_contract.v1 object in the lock envelope')


def _validate_inventory_and_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise ValueError('inventory schema_version must be dataset_inventory.v1')
    if inventory.get('dataset_slug') != DATASET_SLUG or contract.get('dataset_slug') != DATASET_SLUG:
        raise ValueError('contract and inventory must both describe reanalysis_era5_pressure_levels')
    if inventory.get('dataset_id') != DATASET_ID:
        raise ValueError('unexpected inventory dataset_id')
    options = inventory.get('options', {})
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ValueError('advanced_options.dataset_id is not supported by this inventory')
    if adv.get('data_format', inventory.get('defaults', {}).get('data_format')) not in options.get('data_format', []):
        raise ValueError('unsupported data_format')
    if adv.get('download_format', inventory.get('defaults', {}).get('download_format')) not in options.get('download_format', []):
        raise ValueError('unsupported download_format')
    product_types = adv.get('product_type', contract.get('scope', {}).get('product_type', ['reanalysis']))
    if isinstance(product_types, str):
        product_types = [product_types]
    for product_type in product_types:
        if product_type not in options.get('product_type', []):
            raise ValueError(f'unsupported product_type {product_type!r}')
    scope = contract.get('scope') or {}
    geography = scope.get('geography') or {}
    area = geography.get('cds_area', inventory.get('defaults', {}).get('area'))
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise ValueError('scope.geography.cds_area must contain four numeric values')
    north, west, south, east = [float(x) for x in area]
    if not (-90 <= south <= north <= 90 and -360 <= west <= 360 and -360 <= east <= 360):
        raise ValueError('scope.geography.cds_area is outside supported latitude/longitude bounds')
    selected_times = ((scope.get('time') or {}).get('selected_times'))
    if not selected_times:
        raise ValueError('scope.time.selected_times is required')
    for t in selected_times:
        if t not in options.get('time', []):
            raise ValueError(f'unsupported selected time {t!r}')
    date_range = scope.get('date_range') or {}
    if 'start_date' not in date_range or 'end_date' not in date_range:
        raise ValueError('scope.date_range.start_date and end_date are required')
    for field in contract.get('fields') or []:
        name = field.get('name')
        if name not in options.get('variable', []):
            raise ValueError(f'unsupported variable {name!r}')
        selectors = field.get('selectors') or []
        if len(selectors) != 1 or selectors[0].get('dimension') != 'pressure_level':
            raise ValueError('ERA5 pressure-level fields require exactly one pressure_level selector')
        level = str(selectors[0].get('value'))
        if level not in options.get('pressure_level', []):
            raise ValueError(f'unsupported pressure_level {level!r}')


def _channels_from_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> list[Channel]:
    channels: list[Channel] = []
    seen: set[str] = set()
    for field in contract.get('fields') or []:
        selectors = tuple((s['dimension'], str(s['value']), s.get('unit')) for s in (field.get('selectors') or []))
        field_id = _canonical_field_id(field['name'], selectors)
        if field_id in seen:
            raise ValueError(f'duplicate requested field-selector channel {field_id}')
        seen.add(field_id)
        suffix = '__'.join(f'{_slug(d)}_{_slug(v)}' for d, v, _u in selectors)
        array_path = _slug(field['name']) + (('__' + suffix) if suffix else '')
        selector_dim = (array_path + '__pressure_level') if selectors else None
        channels.append(Channel(field['name'], selectors, field_id, array_path, selector_dim))
    if not channels:
        raise ValueError('contract must request at least one field')
    return channels


def _canonical_field_id(name: str, selectors: tuple[tuple[str, str, str | None], ...]) -> str:
    if not selectors:
        return name
    body = ','.join(f'{d}={json.dumps(v, ensure_ascii=False, separators=(",", ":"))}' for d, v, _u in selectors)
    return f'{name}[{body}]'


def _slug(value: str) -> str:
    out = re.sub(r'[^A-Za-z0-9_]+', '_', str(value)).strip('_').lower()
    return out or 'value'


def _selected_datetimes(contract: dict[str, Any]) -> pd.DatetimeIndex:
    scope = contract['scope']
    dr = scope['date_range']
    start_day = date.fromisoformat(str(dr['start_date'])[:10])
    end_day = date.fromisoformat(str(dr['end_date'])[:10])
    if end_day < start_day:
        raise ValueError('end_date is before start_date')
    inclusive = bool(dr.get('inclusive', True))
    selected_times = list(scope['time']['selected_times'])
    stamps: list[pd.Timestamp] = []
    day = start_day
    last_day = end_day if inclusive else end_day - timedelta(days=1)
    while day <= last_day:
        for hhmm in selected_times:
            h, m = [int(x) for x in hhmm.split(':')]
            stamps.append(pd.Timestamp(datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc)).tz_convert(None))
        day += timedelta(days=1)
    return pd.DatetimeIndex(stamps).sort_values()


def _verify_source_fixture(cache_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('source_fixture_manifest.json is required for offline credential-free execution; network fallback is intentionally disabled')
    with manifest_path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ValueError('source fixture manifest schema_version must be source_fixture_manifest.v1')
    entries = manifest.get('entries') or []
    if not entries:
        raise ValueError('source fixture manifest contains no entries')
    paths: list[Path] = []
    for entry in entries:
        rel = Path(entry['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('fixture relative_path must stay beneath cache_dir')
        path = (cache_root / rel).resolve()
        if not str(path).startswith(str(cache_root) + os.sep) and path != cache_root:
            raise ValueError('fixture path escapes cache_dir')
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'fixture file is missing: {entry["relative_path"]}')
        size = path.stat().st_size
        if size != int(entry['size_bytes']):
            raise ValueError(f'fixture size mismatch for {entry["relative_path"]}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(entry['sha256']).lower():
            raise ValueError(f'fixture sha256 mismatch for {entry["relative_path"]}')
        paths.append(path)
    return entries, paths


def _safe_fixture_key(entry: dict[str, Any]) -> str:
    entry_id = str(entry.get('entry_id') or '')
    digest = str(entry.get('sha256') or '')[:16]
    if re.fullmatch(r'[A-Za-z0-9_.:-]{1,80}', entry_id):
        return f'fixture:{entry_id}:{digest}'
    return f'fixture:sha256:{digest}'


def _open_source_dataset(paths: list[Path]) -> xr.Dataset:
    datasets = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {'.grib', '.grb', '.grb2'}:
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True)
        else:
            ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True)
        ds = _normalise_source_dataset(ds)
        if 'pressure_level' in ds.coords and 'pressure_level' not in ds.dims:
            ds = ds.expand_dims('pressure_level')
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')
    except Exception:
        return xr.merge(datasets, compat='no_conflicts', combine_attrs='drop_conflicts')


def _normalise_source_dataset(ds: xr.Dataset) -> xr.Dataset:
    renames = {}
    all_names = set(ds.dims) | set(ds.coords) | set(ds.variables)
    for canonical, aliases in COORD_ALIASES.items():
        if canonical in all_names:
            continue
        for alias in aliases:
            if alias in all_names:
                renames[alias] = canonical
                break
    if renames:
        ds = ds.rename(renames)
    required = {'time', 'latitude', 'longitude'}
    missing = required - (set(ds.dims) | set(ds.coords))
    if missing:
        raise ValueError(f'source fixture is missing required coordinates/dimensions: {sorted(missing)}')
    return ds


def _filter_source_dataset(ds: xr.Dataset, contract: dict[str, Any], wanted_times: pd.DatetimeIndex) -> xr.Dataset:
    ds = _select_times(ds, wanted_times)
    ds = _select_area(ds, contract)
    return ds


def _select_times(ds: xr.Dataset, wanted_times: pd.DatetimeIndex) -> xr.Dataset:
    source_times = pd.DatetimeIndex(pd.to_datetime(ds['time'].values)).tz_localize(None)
    index_by_time = {ts: i for i, ts in enumerate(source_times)}
    missing = [str(ts) for ts in wanted_times if ts not in index_by_time]
    if missing:
        raise ValueError('source fixture does not contain all selected timestamps; first missing: ' + missing[0])
    positions = [index_by_time[ts] for ts in wanted_times]
    return ds.isel(time=positions)


def _select_area(ds: xr.Dataset, contract: dict[str, Any]) -> xr.Dataset:
    area = (contract.get('scope', {}).get('geography', {}) or {}).get('cds_area', [90, -180, -90, 180])
    north, west, south, east = [float(x) for x in area]
    if [north, west, south, east] == [90.0, -180.0, -90.0, 180.0]:
        return ds
    lat = ds['latitude'].values
    lat_mask = (lat >= south) & (lat <= north)
    if not lat_mask.any():
        raise ValueError('area selection removed all latitude coordinates')
    lon = ds['longitude'].values
    if west <= east:
        lon_mask = (lon >= west) & (lon <= east)
        if not lon_mask.any() and np.nanmin(lon) >= 0 and west < 0:
            w2, e2 = west % 360, east % 360
            lon_mask = (lon >= w2) | (lon <= e2) if w2 > e2 else ((lon >= w2) & (lon <= e2))
    else:
        lon_mask = (lon >= west) | (lon <= east)
    if not lon_mask.any():
        raise ValueError('area selection removed all longitude coordinates')
    return ds.isel(latitude=np.flatnonzero(lat_mask), longitude=np.flatnonzero(lon_mask))


def _build_public_dataset(source: xr.Dataset, channels: list[Channel], inventory: dict[str, Any]) -> xr.Dataset:
    coords: dict[str, Any] = {
        'time': source['time'].copy(deep=True),
        'latitude': source['latitude'].copy(deep=True),
        'longitude': source['longitude'].copy(deep=True),
    }
    data_vars: dict[str, xr.DataArray] = {}
    for c in channels:
        da = _extract_channel(source, c)
        da = _ensure_exact_float32(da, c.field_id)
        meta = ((inventory.get('option_metadata') or {}).get('variable') or {}).get(c.field_name, {})
        attrs = dict(da.attrs)
        if meta.get('units') and not attrs.get('units'):
            attrs['units'] = meta['units']
        if meta.get('description') and not attrs.get('description'):
            attrs['description'] = meta['description']
        attrs['canonical_field_id'] = c.field_id
        attrs['source_variable'] = c.field_name
        attrs['publication_note'] = 'decoded float32 values; uncompressed sample-aligned Zarr v3 chunks'
        da.attrs = attrs
        data_vars[c.array_path] = da
        if c.selector_dim:
            coords[c.selector_dim] = da.coords[c.selector_dim].copy(deep=True)
    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        'dataset_slug': DATASET_SLUG,
        'dataset_id': DATASET_ID,
        'provider': 'ECMWF',
        'publication_format': 'zarr_v3_consolidated',
        'pipeline_id': PIPELINE_ID,
    })
    for v in out.variables:
        out[v].encoding = {}
    return out


def _extract_channel(ds: xr.Dataset, c: Channel) -> xr.DataArray:
    var_name = _find_source_variable(ds, c.field_name)
    da = ds[var_name]
    for dim, value, unit in c.selectors:
        if dim != 'pressure_level':
            raise ValueError(f'unsupported selector dimension {dim}')
        da = _select_pressure_level(da, value, c.selector_dim or 'pressure_level', unit)
    required_dims = ['time', c.selector_dim, 'latitude', 'longitude'] if c.selector_dim else ['time', 'latitude', 'longitude']
    missing = [d for d in required_dims if d not in da.dims]
    if missing:
        raise ValueError(f'channel {c.field_id} is missing required dimensions after filtering: {missing}')
    da = da.transpose(*required_dims)
    return da.rename(c.array_path)


def _find_source_variable(ds: xr.Dataset, field_name: str) -> str:
    for candidate in VAR_ALIASES.get(field_name, [field_name]):
        if candidate in ds.data_vars:
            return candidate
    for name, da in ds.data_vars.items():
        attrs = {k.lower(): str(v).lower() for k, v in da.attrs.items()}
        if attrs.get('long_name') == field_name.lower() or attrs.get('standard_name') == field_name.lower():
            return name
    raise ValueError(f'source fixture does not contain requested variable {field_name!r}')


def _select_pressure_level(da: xr.DataArray, value: str, target_dim: str, unit: str | None) -> xr.DataArray:
    coord_name = next((name for name in COORD_ALIASES['pressure_level'] if name in da.dims or name in da.coords), None)
    if coord_name is not None and coord_name in da.dims and coord_name in da.coords:
        vals = da[coord_name].values
        matches = [i for i, v in enumerate(vals) if _coord_value_equal(v, value)]
        if not matches:
            raise ValueError(f'source variable lacks requested pressure_level {value}')
        da = da.isel({coord_name: [matches[0]]}).rename({coord_name: target_dim})
    elif coord_name is not None and coord_name in da.coords:
        values = da[coord_name].values
        scalar = values.item() if np.ndim(values) == 0 else np.ravel(values)[0]
        if not _coord_value_equal(scalar, value):
            raise ValueError(f'source scalar pressure_level {scalar!r} does not match requested {value!r}')
        da = da.drop_vars(coord_name).expand_dims({target_dim: [scalar]})
    else:
        grib_level = da.attrs.get('GRIB_level')
        grib_type = str(da.attrs.get('GRIB_typeOfLevel', '')).lower()
        if grib_level is not None and ('isobaric' in grib_type or not grib_type) and _coord_value_equal(grib_level, value):
            da = da.expand_dims({target_dim: [grib_level]})
        else:
            raise ValueError('source variable lacks pressure_level coordinate')
    da[target_dim].attrs.update({'units': unit or 'hPa', 'long_name': 'pressure level'})
    return da


def _coord_value_equal(actual: Any, requested: str) -> bool:
    try:
        return float(actual) == float(requested)
    except Exception:
        return str(actual) == str(requested)


def _ensure_exact_float32(da: xr.DataArray, field_id: str) -> xr.DataArray:
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype('float32')
        return da
    vals = da.values
    vals32 = vals.astype('float32')
    if vals.dtype == np.float32:
        return da
    roundtrip = vals32.astype(vals.dtype)
    if not np.array_equal(vals, roundtrip, equal_nan=True):
        raise ValueError(f'decoded values for {field_id} are not exactly representable as float32; refusing precision-changing publication')
    return da.astype('float32')


def _publish_zarr_atomic(ds: xr.Dataset, final_store: Path, channels: list[Channel]) -> None:
    tmp = final_store.parent / ('.tmp-' + final_store.name)
    backup = final_store.parent / ('.old-' + final_store.name)
    for p in (tmp, backup):
        if p.exists():
            shutil.rmtree(p)
    encoding = _zarr_encoding(ds, channels)
    try:
        ds.to_zarr(tmp, mode='w', consolidated=True, zarr_format=3, encoding=encoding)
    except TypeError:
        legacy = {k: {kk: vv for kk, vv in enc.items() if kk not in {'compressors'}} for k, enc in encoding.items()}
        for enc in legacy.values():
            enc['compressor'] = None
        if tmp.exists():
            shutil.rmtree(tmp)
        ds.to_zarr(tmp, mode='w', consolidated=True, zarr_format=3, encoding=legacy)
    _assert_zarr_v3_uncompressed(tmp, [c.array_path for c in channels])
    reopened = xr.open_zarr(tmp, consolidated=True)
    try:
        _validate_reopened(reopened, ds, channels)
    finally:
        reopened.close()
    if backup.exists():
        shutil.rmtree(backup)
    if final_store.exists():
        os.replace(final_store, backup)
    os.replace(tmp, final_store)
    if backup.exists():
        shutil.rmtree(backup)


def _zarr_encoding(ds: xr.Dataset, channels: list[Channel]) -> dict[str, dict[str, Any]]:
    channel_names = {c.array_path for c in channels}
    enc: dict[str, dict[str, Any]] = {}
    for name in ds.variables:
        var = ds[name]
        chunks = tuple(int(ds.sizes[d]) for d in var.dims)
        if name in channel_names:
            chunks = tuple(1 if d == 'time' or d.endswith('__pressure_level') else int(ds.sizes[d]) for d in var.dims)
            enc[name] = {'chunks': chunks, 'dtype': 'float32', 'compressors': None, 'filters': None, '_FillValue': None}
        else:
            enc[name] = {'chunks': chunks, 'compressors': None, 'filters': None}
    return enc


def _assert_zarr_v3_uncompressed(store: Path, data_array_names: list[str]) -> None:
    root_meta = json.loads((store / 'zarr.json').read_text(encoding='utf-8'))
    if root_meta.get('zarr_format') != 3:
        raise ValueError('published store is not Zarr format 3')
    if 'consolidated_metadata' not in root_meta:
        raise ValueError('published Zarr v3 store lacks consolidated metadata')
    forbidden = ('blosc', 'zstd', 'zlib', 'gzip', 'lz4', 'quant', 'delta', 'scale', 'offset', 'shuffle')
    for name in data_array_names:
        meta_path = store / name / 'zarr.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        text = json.dumps(meta, sort_keys=True).lower()
        if any(token in text for token in forbidden):
            raise ValueError(f'data array {name} metadata contains a forbidden compression or value-changing transform')
        codecs = meta.get('codecs', [])
        non_bytes = [c for c in codecs if str(c.get('name', '')).lower() not in {'bytes', 'endian'}]
        if non_bytes:
            raise ValueError(f'data array {name} is not explicitly uncompressed: {non_bytes}')


def _validate_reopened(reopened: xr.Dataset, expected: xr.Dataset, channels: list[Channel]) -> None:
    if set(reopened.data_vars) != set(expected.data_vars):
        raise ValueError('reopened Zarr data variable names do not match expected publication')
    for coord in expected.coords:
        if coord not in reopened.coords:
            raise ValueError(f'reopened Zarr is missing coordinate {coord}')
        if not np.array_equal(reopened[coord].values, expected[coord].values, equal_nan=True):
            raise ValueError(f'reopened coordinate {coord} differs from expected values')
    for c in channels:
        actual = reopened[c.array_path]
        exp = expected[c.array_path]
        if actual.dims != exp.dims or actual.shape != exp.shape:
            raise ValueError(f'reopened channel {c.field_id} shape/dims mismatch')
        if c.selector_dim and c.selector_dim not in actual.dims:
            raise ValueError(f'reopened channel {c.field_id} lost its selector dimension')
        if not np.array_equal(actual.values, exp.values, equal_nan=True):
            raise ValueError(f'reopened channel {c.field_id} values changed')
        for attr in ('units', 'canonical_field_id'):
            if attr in exp.attrs and actual.attrs.get(attr) != exp.attrs.get(attr):
                raise ValueError(f'reopened channel {c.field_id} lost attribute {attr}')
