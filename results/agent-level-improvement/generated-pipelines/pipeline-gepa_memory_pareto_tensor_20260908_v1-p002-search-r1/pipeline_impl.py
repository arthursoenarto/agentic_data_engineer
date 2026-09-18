from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p002-search-r1'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'dataset.zarr'

VAR_ALIASES = {
    'temperature': ['temperature', 't'],
    'geopotential': ['geopotential', 'z'],
    'u_component_of_wind': ['u_component_of_wind', 'u', 'u_component'],
    'v_component_of_wind': ['v_component_of_wind', 'v', 'v_component'],
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


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract = _find_contract(contract_lock)
    _validate_contract(contract, inventory)
    cache_root = Path(cache_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fixture_entries = _verify_fixture(cache_root)
    source = _open_fixture(cache_root, fixture_entries)
    try:
        selected = _build_output_dataset(source, contract, inventory)
        tmp_store = _publish_zarr_atomic(selected, out_root, contract)
        _validate_publication(tmp_store, selected)
    finally:
        source.close()

    final_store = out_root / STORE_NAME
    channels = []
    for item in selected.attrs['_artifact_channels']:
        channels.append(item)
    selected.attrs.pop('_artifact_channels', None)

    reused = [f"fixture:{e['entry_id']}:{e['sha256'][:16]}" for e in fixture_entries]
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
            'store_path': final_store.relative_to(out_root).as_posix(),
            'dimensions': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'coordinates': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'channels': channels,
        },
        'warnings': [],
    }


def _find_contract(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        if obj.get('schema_version') == 'dataset_contract.v1':
            return obj
        for key in ('contract', 'dataset_contract', 'selected_contract', 'lock', 'payload'):
            if key in obj:
                try:
                    return _find_contract(obj[key])
                except ValueError:
                    pass
        for value in obj.values():
            try:
                return _find_contract(value)
            except ValueError:
                pass
    elif isinstance(obj, list):
        for value in obj:
            try:
                return _find_contract(value)
            except ValueError:
                pass
    raise ValueError('dataset_contract.v1 not found in lock envelope')


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> None:
    if contract.get('dataset_slug') != DATASET_SLUG:
        raise ValueError('contract dataset_slug is not supported by this adapter')
    if inventory.get('dataset_slug') != DATASET_SLUG or inventory.get('dataset_id') != DATASET_ID:
        raise ValueError('inventory does not match frozen ERA5 pressure-level inventory')
    opts = inventory.get('options', {})
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ValueError('advanced_options.dataset_id is outside this inventory')
    if adv.get('data_format', 'grib') != 'grib':
        raise ValueError('fixed policy requires grib acquisition_format')
    if adv.get('download_format', 'unarchived') not in opts.get('download_format', []):
        raise ValueError('invalid download_format')
    product_type = contract.get('scope', {}).get('product_type', 'reanalysis')
    if product_type not in opts.get('product_type', []):
        raise ValueError('invalid product_type')
    times = contract.get('scope', {}).get('time', {}).get('selected_times', [])
    if not times:
        raise ValueError('scope.time.selected_times is required')
    for t in times:
        if t not in opts.get('time', []):
            raise ValueError(f'invalid selected time {t}')
    for field in contract.get('fields', []):
        name = field.get('name')
        if name not in opts.get('variable', []):
            raise ValueError(f'invalid variable {name}')
        seen = set()
        for selector in field.get('selectors') or []:
            dim = selector.get('dimension')
            val = str(selector.get('value'))
            if dim in seen:
                raise ValueError(f'duplicate selector dimension {dim}')
            seen.add(dim)
            if dim != 'pressure_level':
                raise ValueError(f'unsupported selector dimension {dim}')
            if val not in opts.get('pressure_level', []):
                raise ValueError(f'invalid pressure_level {val}')
    dr = contract.get('scope', {}).get('date_range') or {}
    if not dr.get('start_date') or not dr.get('end_date'):
        raise ValueError('date_range start_date and end_date are required')
    start = pd.Timestamp(dr['start_date'], tz='UTC')
    end = pd.Timestamp(dr['end_date'], tz='UTC')
    if end < start:
        raise ValueError('date_range end_date precedes start_date')
    years = {str(y) for y in range(start.year, end.year + 1)}
    bad_years = sorted(years - set(opts.get('year', [])))
    if bad_years:
        raise ValueError(f'year outside inventory options: {bad_years[0]}')
    area = contract.get('scope', {}).get('geography', {}).get('cds_area', inventory.get('defaults', {}).get('area'))
    if area is None or len(area) != 4:
        raise ValueError('geography.cds_area must contain north, west, south, east')
    n, w, s, e = [float(x) for x in area]
    if not (-90 <= s <= 90 and -90 <= n <= 90 and -360 <= w <= 360 and -360 <= e <= 360 and n >= s):
        raise ValueError('geography.cds_area is outside valid latitude/longitude bounds')


def _verify_fixture(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('source_fixture_manifest.json is required; network acquisition is intentionally unsupported')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ValueError('unsupported source fixture manifest schema')
    entries = manifest.get('entries') or []
    if not entries:
        raise ValueError('source fixture manifest contains no entries')
    checked = []
    root_resolved = cache_root.resolve()
    for entry in entries:
        rel = entry.get('relative_path')
        if not rel or Path(rel).is_absolute():
            raise ValueError('fixture relative_path must be relative')
        path = (cache_root / rel).resolve()
        if root_resolved not in path.parents and path != root_resolved:
            raise ValueError('fixture path escapes cache_dir')
        if not path.exists():
            raise FileNotFoundError(f'fixture file missing for entry {entry.get("entry_id")}')
        if path.is_dir():
            size = sum(p.stat().st_size for p in sorted(path.rglob('*')) if p.is_file())
            digest = _hash_dir(path)
        else:
            size = path.stat().st_size
            digest = _hash_file(path)
        if int(entry.get('size_bytes')) != size:
            raise ValueError(f'fixture size mismatch for entry {entry.get("entry_id")}')
        if entry.get('sha256') != digest:
            raise ValueError(f'fixture sha256 mismatch for entry {entry.get("entry_id")}')
        checked.append(dict(entry))
    return checked


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob('*') if q.is_file()):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(b'\0')
        h.update(_hash_file(p).encode())
        h.update(b'\0')
    return h.hexdigest()


def _open_fixture(cache_root: Path, entries: list[dict[str, Any]]) -> xr.Dataset:
    datasets = []
    for entry in entries:
        path = cache_root / entry['relative_path']
        suffixes = ''.join(path.suffixes).lower()
        if path.is_dir() or suffixes.endswith('.zarr'):
            ds = xr.open_zarr(path, consolidated=None).load()
        elif suffixes.endswith(('.nc', '.nc4', '.netcdf')):
            ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
        elif suffixes.endswith(('.grib', '.grb', '.grib2', '.grb2')):
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}).load()
        else:
            try:
                ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
            except Exception as exc:
                raise ValueError(f'unsupported fixture file format for {entry.get("entry_id")}') from exc
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.combine_by_coords(datasets, combine_attrs='override').load()
    except Exception:
        return xr.merge(datasets, compat='override', combine_attrs='override').load()


def _build_output_dataset(source: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any]) -> xr.Dataset:
    time_name = _find_name(source, ['time', 'valid_time', 'timestamp'])
    lat_name = _find_name(source, ['latitude', 'lat'])
    lon_name = _find_name(source, ['longitude', 'lon'])
    pressure_name = _find_name(source, ['pressure_level', 'level', 'isobaricInhPa', 'plev'], required=False)
    timestamps = _selected_timestamps(contract)
    ds = source
    ds = _subset_time(ds, time_name, timestamps)
    ds = _subset_area(ds, lat_name, lon_name, contract)

    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, Any] = {
        'time': ds[time_name].values,
        'latitude': ds[lat_name].values,
        'longitude': ds[lon_name].values,
    }
    coord_attrs = {
        'time': dict(ds[time_name].attrs),
        'latitude': dict(ds[lat_name].attrs),
        'longitude': dict(ds[lon_name].attrs),
    }
    coord_attrs['latitude'].setdefault('units', 'degrees_north')
    coord_attrs['longitude'].setdefault('units', 'degrees_east')
    channels = []

    for ordinal, field in enumerate(contract.get('fields', [])):
        src_var = _find_variable(ds, field['name'])
        selectors = field.get('selectors') or []
        field_id = _field_id(field)
        array_name = _array_name(field_id, ordinal)
        da = ds[src_var]
        rename = {}
        if time_name in da.dims:
            rename[time_name] = 'time'
        if lat_name in da.dims:
            rename[lat_name] = 'latitude'
        if lon_name in da.dims:
            rename[lon_name] = 'longitude'
        selector_paths = {}
        selector_values = {}
        for sel in selectors:
            if sel['dimension'] == 'pressure_level':
                requested_pressure = str(sel['value'])
                da_pressure_name = pressure_name
                has_pressure_coord = da_pressure_name is not None and (da_pressure_name in da.dims or da_pressure_name in da.coords)
                if not has_pressure_coord:
                    if pressure_name is None:
                        raise ValueError(f'pressure_level selector requested for {field["name"]}, but source variable has no pressure coordinate')
                    native = _select_pressure_value(np.asarray(ds[pressure_name].values).reshape(-1), requested_pressure)
                    da = da.expand_dims({pressure_name: [native]})
                    da_pressure_name = pressure_name
                elif da_pressure_name in da.dims:
                    native = _select_pressure_value(da[da_pressure_name].values, requested_pressure)
                    da = da.sel({da_pressure_name: [native]})
                else:
                    native = _select_pressure_value(np.asarray(da[da_pressure_name].values).reshape(-1), requested_pressure)
                    da = da.expand_dims({da_pressure_name: [native]})
                unique_dim = f'pressure_level__{array_name}'
                rename[da_pressure_name] = unique_dim
                coords[unique_dim] = np.asarray([native], dtype=da[da_pressure_name].dtype)
                coord_attrs[unique_dim] = dict(da[da_pressure_name].attrs)
                coord_attrs[unique_dim].setdefault('units', 'hPa')
                selector_paths['pressure_level'] = unique_dim
                selector_values['pressure_level'] = requested_pressure
        da = da.rename(rename)
        keep_dims = [d for d in da.dims if d in set(rename.values()) or d in ('time', 'latitude', 'longitude')]
        da = da.transpose(*keep_dims)
        attrs = dict(da.attrs)
        meta = inventory.get('option_metadata', {}).get('variable', {}).get(field['name'], {})
        if meta.get('units') and 'units' not in attrs:
            attrs['units'] = meta['units']
        if meta.get('description') and 'description' not in attrs:
            attrs['description'] = meta['description']
        attrs['field_id'] = field_id
        attrs['source_variable'] = field['name']
        attrs['selectors_json'] = json.dumps(selector_values, sort_keys=True, separators=(',', ':'))
        da.attrs = attrs
        da.encoding = {}
        data_vars[array_name] = da
        channels.append({
            'field_id': field_id,
            'array_path': array_name,
            'selectors': selector_values,
            'selector_coordinate_paths': selector_paths,
        })

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        'dataset_slug': DATASET_SLUG,
        'dataset_id': DATASET_ID,
        'provider': 'ECMWF',
        'publication_format': 'zarr',
    })
    for name, attrs in coord_attrs.items():
        if name in out.coords:
            out[name].attrs = attrs
            out[name].encoding = {}
    out.attrs['_artifact_channels'] = channels
    return out


def _find_name(ds: xr.Dataset, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    if required:
        raise ValueError(f'none of the required source names exist: {names}')
    return None


def _find_variable(ds: xr.Dataset, requested: str) -> str:
    for name in VAR_ALIASES.get(requested, [requested]):
        if name in ds.data_vars:
            return name
    raise ValueError(f'source fixture does not contain requested variable {requested}')


def _selected_timestamps(contract: dict[str, Any]) -> pd.DatetimeIndex:
    dr = contract['scope']['date_range']
    start_day = pd.Timestamp(dr['start_date'], tz='UTC').normalize()
    end_day = pd.Timestamp(dr['end_date'], tz='UTC').normalize()
    if bool(dr.get('inclusive', True)):
        end_exclusive = end_day + pd.Timedelta(days=1)
    else:
        end_exclusive = end_day
    selected_times = list(contract['scope']['time']['selected_times'])
    values = []
    day = start_day
    while day < end_exclusive:
        for hhmm in selected_times:
            hour, minute = [int(x) for x in hhmm.split(':')]
            ts = day + pd.Timedelta(hours=hour, minutes=minute)
            if start_day <= ts < end_exclusive:
                values.append(ts)
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(values).tz_convert(None)


def _subset_time(ds: xr.Dataset, time_name: str, timestamps: pd.DatetimeIndex) -> xr.Dataset:
    source_times = pd.DatetimeIndex(pd.to_datetime(ds[time_name].values)).tz_localize(None)
    missing = [str(t) for t in timestamps if t not in set(source_times)]
    if missing:
        raise ValueError(f'source fixture is missing selected timestamps, first missing {missing[0]}')
    return ds.sel({time_name: timestamps.to_numpy(dtype='datetime64[ns]')})


def _subset_area(ds: xr.Dataset, lat_name: str, lon_name: str, contract: dict[str, Any]) -> xr.Dataset:
    area = contract.get('scope', {}).get('geography', {}).get('cds_area', [90, -180, -90, 180])
    north, west, south, east = [float(x) for x in area]
    lat = ds[lat_name].values
    if lat[0] > lat[-1]:
        ds = ds.sel({lat_name: slice(north, south)})
    else:
        ds = ds.sel({lat_name: slice(south, north)})
    lon = np.asarray(ds[lon_name].values, dtype=float)
    full_global = west <= -180 and east >= 180
    if full_global:
        return ds
    if lon.min() >= 0 and (west < 0 or east < 0):
        west = west % 360
        east = east % 360
    if west <= east:
        return ds.sel({lon_name: slice(west, east)})
    left = ds.sel({lon_name: slice(west, float(lon.max()))})
    right = ds.sel({lon_name: slice(float(lon.min()), east)})
    return xr.concat([left, right], dim=lon_name)


def _select_pressure_value(values: np.ndarray, requested: str) -> Any:
    for val in values:
        if _pressure_token(val) == requested:
            return val.item() if hasattr(val, 'item') else val
    raise ValueError(f'source fixture is missing pressure_level {requested}')


def _pressure_token(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(value)


def _field_id(field: dict[str, Any]) -> str:
    selectors = field.get('selectors') or []
    if not selectors:
        return field['name']
    parts = []
    for sel in selectors:
        parts.append(sel['dimension'] + '=' + json.dumps(str(sel['value']), ensure_ascii=False, separators=(',', ':')))
    return field['name'] + '[' + ','.join(parts) + ']'


def _array_name(field_id: str, ordinal: int) -> str:
    base = re.sub('[^0-9A-Za-z_]+', '_', field_id).strip('_').lower()
    digest = hashlib.sha1(field_id.encode()).hexdigest()[:10]
    return f'v{ordinal:03d}_{base}_{digest}'


def _codec() -> BloscCodec:
    return BloscCodec(cname='zstd', clevel=7, shuffle=BloscShuffle.bitshuffle)


def _publish_zarr_atomic(ds: xr.Dataset, out_root: Path, contract: dict[str, Any]) -> Path:
    final = out_root / STORE_NAME
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()[:16]
    tmp = out_root / f'.tmp-{STORE_NAME}-{digest}'
    shutil.rmtree(tmp, ignore_errors=True)
    codec = _codec()
    enc: dict[str, dict[str, Any]] = {}
    lat_n = int(ds.sizes['latitude'])
    lon_n = int(ds.sizes['longitude'])
    for name, da in ds.data_vars.items():
        chunks = []
        for dim in da.dims:
            if dim == 'time':
                chunks.append(1)
            elif dim.startswith('pressure_level__'):
                chunks.append(1)
            elif dim == 'latitude':
                chunks.append(lat_n)
            elif dim == 'longitude':
                chunks.append(lon_n)
            else:
                chunks.append(int(ds.sizes[dim]))
        enc[name] = {'chunks': tuple(chunks), 'compressors': [codec]}
    for name in ds.coords:
        enc[name] = {'chunks': tuple(int(s) for s in ds[name].shape), 'compressors': [codec]}
    safe = ds.copy(deep=False)
    safe.attrs = {k: v for k, v in safe.attrs.items() if k != '_artifact_channels'}
    for var in list(safe.data_vars) + list(safe.coords):
        safe[var].encoding = {}
    safe.to_zarr(tmp, mode='w', zarr_format=3, consolidated=True, encoding=enc)
    _assert_codec_metadata(tmp, list(ds.data_vars))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return final


def _assert_codec_metadata(store: Path, data_vars: list[str]) -> None:
    root = zarr.open_group(str(store), mode='r')
    for name in data_vars:
        arr = root[name]
        meta = json.dumps(getattr(arr, 'metadata', {}), default=str).lower()
        if 'zstd' not in meta or 'blosc' not in meta:
            raise RuntimeError(f'published array {name} is missing explicit Blosc-Zstd compression')


def _validate_publication(store: Path, expected: xr.Dataset) -> None:
    root_json = json.loads((store / 'zarr.json').read_text())
    if root_json.get('zarr_format') != 3:
        raise RuntimeError('published store is not Zarr v3')
    if 'consolidated_metadata' not in root_json:
        raise RuntimeError('published Zarr v3 metadata is not consolidated')
    got = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    cmp = expected.copy(deep=False)
    cmp.attrs = {k: v for k, v in cmp.attrs.items() if k != '_artifact_channels'}
    xr.testing.assert_identical(got, cmp)
    for name, da in expected.data_vars.items():
        z = zarr.open_group(str(store), mode='r')[name]
        want = tuple(1 if d == 'time' or d.startswith('pressure_level__') else int(expected.sizes[d]) for d in da.dims)
        if tuple(z.chunks) != want:
            raise RuntimeError(f'unexpected chunks for {name}: {z.chunks}, expected {want}')
