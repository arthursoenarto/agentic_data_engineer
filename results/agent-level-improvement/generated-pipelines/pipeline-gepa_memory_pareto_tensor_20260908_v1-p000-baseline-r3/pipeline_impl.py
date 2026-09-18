from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p000-baseline-r3'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
PUBLIC_STORE_NAME = 'dataset.zarr'

_VAR_ALIASES = {
    't': 'temperature',
    'z': 'geopotential',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'w': 'vertical_velocity',
    'r': 'relative_humidity',
    'q': 'specific_humidity',
    'vo': 'vorticity',
    'd': 'divergence',
    'pv': 'potential_vorticity',
    'cc': 'fraction_of_cloud_cover',
    'o3': 'ozone_mass_mixing_ratio',
    'ciwc': 'specific_cloud_ice_water_content',
    'clwc': 'specific_cloud_liquid_water_content',
    'crwc': 'specific_rain_water_content',
    'cswc': 'specific_snow_water_content',
}


class ContractError(ValueError):
    pass


class FixtureError(RuntimeError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()

    manifest = _verify_fixture_manifest(cache_root)
    contract = _select_contract(contract_lock)
    request = _validate_contract(contract, inventory)

    datasets = [_open_source_entry(entry['path']) for entry in manifest['entries']]
    try:
        ds = _combine_sources(datasets)
    finally:
        for d in datasets:
            try:
                d.close()
            except Exception:
                pass

    ds = _normalize_dataset(ds, request, inventory)
    ds = _filter_dataset(ds, request)
    ds = _finalize_public_dataset(ds, contract, inventory, manifest, request)

    out_root.mkdir(parents=True, exist_ok=True)
    final_store = out_root / PUBLIC_STORE_NAME
    _publish_zarr_atomically(ds, final_store)
    opened = xr.open_zarr(final_store, consolidated=True)
    try:
        _validate_publication(opened, request)
    finally:
        opened.close()

    channels = []
    for field in request['fields']:
        channels.append({
            'field_id': _canonical_field_id(field),
            'array_path': field['name'],
            'selectors': {s['dimension']: s['value'] for s in field['selectors']},
            'selector_coordinate_paths': {s['dimension']: s['dimension'] for s in field['selectors']},
        })

    return {
        'cache': {
            'hits': len(manifest['entries']),
            'misses': 0,
            'acquired': 0,
            'reused_keys': [entry['entry_id'] for entry in manifest['entries']],
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': PUBLIC_STORE_NAME,
            'dimensions': {
                'sample': 'time',
                'y': 'latitude',
                'x': 'longitude',
            },
            'coordinates': {
                'sample': 'time',
                'y': 'latitude',
                'x': 'longitude',
            },
            'channels': channels,
        },
        'warnings': [],
    }


def _verify_fixture_manifest(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        raise FixtureError('cache_dir must contain source_fixture_manifest.json; this adapter is offline and never contacts CDS at runtime')
    raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    if raw.get('schema_version') != 'source_fixture_manifest.v1':
        raise FixtureError('unsupported source fixture manifest schema_version')
    entries = raw.get('entries')
    if not isinstance(entries, list) or not entries:
        raise FixtureError('source fixture manifest must contain at least one entry')
    verified = []
    root = cache_root.resolve()
    for entry in entries:
        rel = entry.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/'):
            raise FixtureError('fixture relative_path must be relative')
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise FixtureError('fixture entry escapes cache_dir') from exc
        if not path.exists() or not path.is_file() and not path.is_dir():
            raise FixtureError(f'fixture entry not found: {rel}')
        expected_size = int(entry.get('size_bytes'))
        expected_sha = str(entry.get('sha256', '')).lower()
        if path.is_file():
            size = path.stat().st_size
            sha = _sha256_file(path)
        else:
            size = _directory_size(path)
            sha = _sha256_directory(path)
        if expected_size <= 0 or size != expected_size:
            raise FixtureError(f'fixture size mismatch for {rel}')
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or sha != expected_sha:
            raise FixtureError(f'fixture sha256 mismatch for {rel}')
        verified.append({
            'entry_id': str(entry.get('entry_id') or expected_sha[:16]),
            'relative_path': rel,
            'source': str(entry.get('source', 'fixture')),
            'size_bytes': size,
            'sha256': sha,
            'path': path,
        })
    return {'schema_version': raw['schema_version'], 'entries': verified}


def _select_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    preferred_keys = ['selected_contract', 'contract', 'dataset_contract', 'runtime_contract']
    if isinstance(lock, dict):
        for key in preferred_keys:
            value = lock.get(key)
            if isinstance(value, dict) and value.get('schema_version') == 'dataset_contract.v1':
                return value
        selected_id = lock.get('selected_contract_id') or lock.get('contract_id')
        found = []
        _collect_contracts(lock, found)
        if selected_id:
            for c in found:
                if c.get('contract_id') == selected_id or c.get('id') == selected_id or c.get('dataset_slug') == selected_id:
                    return c
        if len(found) == 1:
            return found[0]
        matches = [c for c in found if c.get('dataset_slug') == DATASET_SLUG]
        if len(matches) == 1:
            return matches[0]
    raise ContractError('could not select exactly one dataset_contract.v1 from contract_lock')


def _collect_contracts(node: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if node.get('schema_version') == 'dataset_contract.v1':
            out.append(node)
        for value in node.values():
            _collect_contracts(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_contracts(value, out)


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise ContractError('inventory schema_version must be dataset_inventory.v1')
    if contract.get('dataset_slug') != inventory.get('dataset_slug') or contract.get('dataset_slug') != DATASET_SLUG:
        raise ContractError('contract dataset_slug does not match frozen inventory')
    if inventory.get('dataset_id') != DATASET_ID:
        raise ContractError('inventory dataset_id does not match this adapter')
    if contract.get('human_confirmed') is not True:
        raise ContractError('contract must be human_confirmed')

    options = inventory.get('options', {})
    advanced = contract.get('advanced_options') or {}
    if advanced.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ContractError('advanced_options.dataset_id is not supported')
    if advanced.get('data_format', inventory.get('defaults', {}).get('data_format')) != 'grib':
        raise ContractError('fixed policy requires acquisition_format grib')
    if advanced.get('download_format', inventory.get('defaults', {}).get('download_format')) not in options.get('download_format', []):
        raise ContractError('unsupported download_format')

    product_type = advanced.get('product_type') or contract.get('scope', {}).get('product_type') or inventory.get('defaults', {}).get('product_type')
    if isinstance(product_type, str):
        product_values = [product_type]
    else:
        product_values = list(product_type or [])
    if not product_values:
        raise ContractError('product_type is required')
    _require_options('product_type', product_values, options)

    fields = contract.get('fields')
    if not isinstance(fields, list) or not fields:
        raise ContractError('contract must request at least one field')
    clean_fields = []
    for field in fields:
        name = field.get('name')
        if name not in options.get('variable', []):
            raise ContractError(f'unsupported variable: {name}')
        selectors = field.get('selectors') or []
        if not isinstance(selectors, list):
            raise ContractError('field selectors must be a list')
        seen_dims = set()
        clean_selectors = []
        for selector in selectors:
            dim = selector.get('dimension')
            value = str(selector.get('value'))
            if dim in seen_dims:
                raise ContractError(f'duplicate selector dimension for {name}: {dim}')
            seen_dims.add(dim)
            if dim != 'pressure_level':
                raise ContractError(f'unsupported selector dimension: {dim}')
            if value not in options.get('pressure_level', []):
                raise ContractError(f'unsupported pressure_level: {value}')
            clean_selectors.append({'dimension': dim, 'value': value, 'unit': selector.get('unit'), 'label': selector.get('label')})
        if not clean_selectors:
            raise ContractError('ERA5 pressure-level fields must include a pressure_level selector')
        clean_fields.append({'name': name, 'selectors': clean_selectors})

    scope = contract.get('scope') or {}
    dr = scope.get('date_range') or {}
    start = _parse_date(dr.get('start_date'), 'start_date')
    end = _parse_date(dr.get('end_date'), 'end_date')
    if end < start:
        raise ContractError('end_date precedes start_date')
    years = [f'{y:04d}' for y in range(start.year, end.year + 1)]
    _require_options('year', years, options)

    time_scope = scope.get('time') or {}
    selected_times = time_scope.get('selected_times') or []
    if not isinstance(selected_times, list) or not selected_times:
        raise ContractError('scope.time.selected_times is required')
    _require_options('time', selected_times, options)
    if time_scope.get('timezone', 'UTC') != 'UTC':
        raise ContractError('only UTC selected_times are supported')

    geography = scope.get('geography') or {}
    area = geography.get('cds_area') or inventory.get('defaults', {}).get('area')
    if not isinstance(area, list) or len(area) != 4:
        raise ContractError('geography.cds_area must contain north, west, south, east')
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90) or not (-360 <= west <= 360) or not (-360 <= east <= 360):
        raise ContractError('invalid cds_area bounds')

    requested_levels = sorted({s['value'] for f in clean_fields for s in f['selectors']}, key=lambda v: float(v))
    return {
        'fields': clean_fields,
        'start_date': start,
        'end_date': end,
        'inclusive_end': bool(dr.get('inclusive', True)),
        'selected_times': list(selected_times),
        'area': [north, west, south, east],
        'pressure_levels': requested_levels,
        'levels_by_variable': _levels_by_variable(clean_fields),
        'variables': sorted({f['name'] for f in clean_fields}),
        'product_type': product_values,
    }


def _require_options(name: str, values: list[str], options: dict[str, Any]) -> None:
    allowed = set(options.get(name, []))
    missing = [v for v in values if str(v) not in allowed]
    if missing:
        raise ContractError(f'unsupported {name}: {missing}')


def _parse_date(value: Any, label: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ContractError(f'{label} is required')
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ContractError(f'invalid {label}')
    return pd.Timestamp(ts.date())


def _levels_by_variable(fields: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for field in fields:
        values = [s['value'] for s in field['selectors'] if s['dimension'] == 'pressure_level']
        out.setdefault(field['name'], [])
        for value in values:
            if value not in out[field['name']]:
                out[field['name']].append(value)
    return out


def _open_source_entry(path: Path) -> xr.Dataset:
    suffixes = ''.join(path.suffixes).lower()
    if path.is_dir() or suffixes.endswith('.zarr'):
        return xr.open_zarr(path, consolidated=None).load()
    if suffixes.endswith(('.nc', '.nc4', '.netcdf')):
        return xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
    if suffixes.endswith(('.grib', '.grb', '.grib2', '.grb2')):
        try:
            return xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True).load()
        except Exception as exc:
            raise FixtureError(f'could not open GRIB fixture {path.name}: {exc}') from exc
    try:
        return xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
    except Exception as netcdf_exc:
        try:
            return xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True).load()
        except Exception as grib_exc:
            raise FixtureError(f'unsupported fixture format for {path.name}: {netcdf_exc}; {grib_exc}') from grib_exc


def _combine_sources(datasets: list[xr.Dataset]) -> xr.Dataset:
    if len(datasets) == 1:
        return datasets[0].copy(deep=True)
    normalized = [d.copy(deep=True) for d in datasets]
    try:
        return xr.combine_by_coords(normalized, combine_attrs='drop_conflicts')
    except Exception:
        return xr.merge(normalized, compat='no_conflicts', combine_attrs='drop_conflicts')


def _normalize_dataset(ds: xr.Dataset, request: dict[str, Any], inventory: dict[str, Any]) -> xr.Dataset:
    rename = {}
    for name in list(ds.data_vars):
        canonical = _VAR_ALIASES.get(name, name)
        if canonical in request['variables'] and canonical != name and canonical not in ds:
            rename[name] = canonical
    if rename:
        ds = ds.rename(rename)

    dim_renames = {}
    for cand in ['valid_time', 'time']:
        if cand in ds.coords or cand in ds.dims:
            if cand != 'time':
                dim_renames[cand] = 'time'
            break
    for cand in ['latitude', 'lat']:
        if cand in ds.coords or cand in ds.dims:
            if cand != 'latitude':
                dim_renames[cand] = 'latitude'
            break
    for cand in ['longitude', 'lon']:
        if cand in ds.coords or cand in ds.dims:
            if cand != 'longitude':
                dim_renames[cand] = 'longitude'
            break
    for cand in ['pressure_level', 'isobaricInhPa', 'level', 'plev']:
        if cand in ds.coords or cand in ds.dims:
            if cand != 'pressure_level':
                dim_renames[cand] = 'pressure_level'
            break
    existing = {k: v for k, v in dim_renames.items() if k in ds and v not in ds}
    if existing:
        ds = ds.rename(existing)

    missing_vars = [v for v in request['variables'] if v not in ds.data_vars]
    if missing_vars:
        raise FixtureError(f'fixture does not contain requested variables: {missing_vars}')
    for coord in ['time', 'latitude', 'longitude']:
        if coord not in ds.coords and coord not in ds.dims:
            raise FixtureError(f'fixture is missing required coordinate {coord}')

    if 'pressure_level' not in ds.dims:
        scalar_level = None
        if 'pressure_level' in ds.coords and ds['pressure_level'].shape == ():
            scalar_level = ds['pressure_level'].item()
        elif len(request['pressure_levels']) == 1:
            scalar_level = float(request['pressure_levels'][0])
        if scalar_level is not None:
            ds = ds.expand_dims({'pressure_level': [scalar_level]})
        else:
            raise FixtureError('fixture is missing pressure_level dimension')

    for var, levels in request['levels_by_variable'].items():
        if var in ds.data_vars and 'pressure_level' not in ds[var].dims:
            if len(levels) != 1:
                raise FixtureError(f'fixture variable {var} does not preserve pressure_level dimension for multiple requested levels')
            ds[var] = ds[var].expand_dims({'pressure_level': [float(levels[0])]})

    for var in request['variables']:
        meta = inventory.get('option_metadata', {}).get('variable', {}).get(var, {})
        if meta.get('units') and not ds[var].attrs.get('units'):
            ds[var].attrs['units'] = meta['units']
        if meta.get('description') and not ds[var].attrs.get('description'):
            ds[var].attrs['description'] = meta['description']
    return ds[request['variables']]


def _filter_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    requested_times = _requested_timestamps(request)
    source_times = pd.to_datetime(ds['time'].values)
    source_ns = source_times.values.astype('datetime64[ns]')
    req_ns = requested_times.values.astype('datetime64[ns]')
    missing = sorted(set(req_ns.tolist()) - set(source_ns.tolist()))
    if missing:
        sample = [str(np.datetime64(m, 's')) for m in missing[:5]]
        raise FixtureError(f'fixture is incomplete for requested timestamps, first missing: {sample}')
    ds = ds.sel(time=req_ns)

    level_values = ds['pressure_level'].values
    indexers = []
    for level in request['pressure_levels']:
        matches = np.where(np.isclose(level_values.astype(float), float(level)))[0]
        if len(matches) == 0:
            raise FixtureError(f'fixture is missing requested pressure_level {level}')
        indexers.append(level_values[matches[0]])
    ds = ds.sel(pressure_level=indexers)

    for var, levels in request['levels_by_variable'].items():
        if var in ds and 'pressure_level' in ds[var].dims:
            allowed = np.array([float(v) for v in levels], dtype=float)
            mask = xr.apply_ufunc(lambda x: np.isin(x.astype(float), allowed), ds['pressure_level'])
            ds[var] = ds[var].where(mask)

    north, west, south, east = request['area']
    lat = ds['latitude']
    lon = ds['longitude']
    lat_mask = (lat >= south) & (lat <= north)
    lon_values = lon.values.astype(float)
    if west <= -180 and east >= 180:
        lon_mask = xr.ones_like(lon, dtype=bool)
    else:
        w, e = west, east
        if np.nanmin(lon_values) >= 0 and (w < 0 or e < 0):
            w = (w + 360) % 360
            e = (e + 360) % 360
        if w <= e:
            lon_mask = (lon >= w) & (lon <= e)
        else:
            lon_mask = (lon >= w) | (lon <= e)
    ds = ds.sel(latitude=lat[lat_mask], longitude=lon[lon_mask])
    if ds.sizes.get('latitude', 0) == 0 or ds.sizes.get('longitude', 0) == 0:
        raise FixtureError('geographic filter selected no grid cells')
    return ds


def _requested_timestamps(request: dict[str, Any]) -> pd.DatetimeIndex:
    start = request['start_date']
    end = request['end_date']
    if not request['inclusive_end']:
        end = end - pd.Timedelta(days=1)
    days = pd.date_range(start=start, end=end, freq='D')
    values = []
    for day in days:
        for hhmm in request['selected_times']:
            hour, minute = [int(x) for x in hhmm.split(':')]
            values.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(values).sort_values()


def _finalize_public_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], manifest: dict[str, Any], request: dict[str, Any]) -> xr.Dataset:
    ds = ds.copy(deep=True)
    ordered = ['time', 'pressure_level', 'latitude', 'longitude']
    for var in list(ds.data_vars):
        dims = [d for d in ordered if d in ds[var].dims] + [d for d in ds[var].dims if d not in ordered]
        ds[var] = ds[var].transpose(*dims)
    if 'pressure_level' in ds.coords:
        ds['pressure_level'].attrs.setdefault('units', 'hPa')
    ds['time'].attrs.setdefault('timezone', 'UTC')
    ds.attrs.update({
        'dataset_slug': DATASET_SLUG,
        'dataset_id': DATASET_ID,
        'provider': 'ECMWF',
        'publication_format': 'zarr',
        'zarr_format': '3',
        'pipeline_id': PIPELINE_ID,
        'source_fixture_entries': json.dumps([{k: e[k] for k in ['entry_id', 'relative_path', 'size_bytes', 'sha256']} for e in manifest['entries']], sort_keys=True),
        'requested_fields': json.dumps([_canonical_field_id(f) for f in request['fields']], sort_keys=True),
        'inventory_generated_at': str(inventory.get('generated_at', '')),
        'contract_title': str(contract.get('title', '')),
    })
    for name in list(ds.variables):
        ds[name].encoding.clear()
    return ds


def _publish_zarr_atomically(ds: xr.Dataset, final_store: Path) -> None:
    parent = final_store.parent
    tmp = parent / ('.tmp-' + next(tempfile._get_candidate_names()) + '.zarr')
    old = parent / ('.old-' + next(tempfile._get_candidate_names()) + '.zarr')
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        ds.to_zarr(tmp, mode='w', consolidated=True, zarr_format=3)
        opened = xr.open_zarr(tmp, consolidated=True)
        try:
            if set(ds.data_vars) - set(opened.data_vars):
                raise RuntimeError('written Zarr store is missing data variables')
        finally:
            opened.close()
        if final_store.exists():
            os.replace(final_store, old)
        os.replace(tmp, final_store)
        if old.exists():
            shutil.rmtree(old)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        if old.exists() and not final_store.exists():
            os.replace(old, final_store)
        elif old.exists():
            shutil.rmtree(old)
        raise


def _validate_publication(ds: xr.Dataset, request: dict[str, Any]) -> None:
    for dim in ['time', 'pressure_level', 'latitude', 'longitude']:
        if dim not in ds.dims:
            raise RuntimeError(f'published Zarr is missing dimension {dim}')
        if ds.sizes[dim] < 1:
            raise RuntimeError(f'published Zarr dimension {dim} is empty')
    for var in request['variables']:
        if var not in ds.data_vars:
            raise RuntimeError(f'published Zarr is missing variable {var}')
        if 'pressure_level' not in ds[var].dims:
            raise RuntimeError(f'published variable {var} does not preserve pressure_level as a dimension')
    expected_times = _requested_timestamps(request).values.astype('datetime64[ns]')
    actual_times = pd.to_datetime(ds['time'].values).values.astype('datetime64[ns]')
    if list(actual_times) != list(expected_times):
        raise RuntimeError('published time coordinate does not exactly match requested timestamps')
    zarr_json = Path(ds.encoding.get('source', '')).resolve() / 'zarr.json' if ds.encoding.get('source') else None
    if zarr_json and zarr_json.exists():
        meta = json.loads(zarr_json.read_text(encoding='utf-8'))
        if meta.get('zarr_format') != 3:
            raise RuntimeError('published store is not Zarr v3')


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get('selectors') or []
    if not selectors:
        return field['name']
    bits = [s['dimension'] + '=' + json.dumps(str(s['value']), ensure_ascii=False, sort_keys=True) for s in selectors]
    return field['name'] + '[' + ','.join(bits) + ']'


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in sorted(path.rglob('*')) if p.is_file())


def _sha256_directory(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_file():
            rel = p.relative_to(path).as_posix().encode('utf-8')
            h.update(rel + b'\0')
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
    return h.hexdigest()
