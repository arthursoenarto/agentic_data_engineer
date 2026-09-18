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

PIPELINE_ID = 'pipeline-expert-direct-workload-v9-matched-20260906'
POLICY_DATASET_ID = 'reanalysis-era5-pressure-levels'
POLICY_PROVIDER = 'ECMWF'
PUBLIC_STORE = 'dataset.zarr'

_FIELD_SHORT_NAMES = {
    'temperature': ['temperature', 't'],
    'geopotential': ['geopotential', 'z'],
}
_COORD_ALIASES = {
    'time': ['time', 'valid_time', 'datetime', 'sample'],
    'latitude': ['latitude', 'lat', 'y'],
    'longitude': ['longitude', 'lon', 'x'],
    'pressure_level': ['pressure_level', 'isobaricInhPa', 'level', 'plev'],
}


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class ChannelSpec:
    field_name: str
    field_id: str
    selectors: tuple[tuple[str, str], ...]
    array_name: str
    selector_coord_paths: dict[str, str]


@dataclass(frozen=True)
class RuntimeSpec:
    contract: dict[str, Any]
    target_times: tuple[np.datetime64, ...]
    fields: tuple[ChannelSpec, ...]
    area: tuple[float, float, float, float]


def run_pipeline(contract_lock, inventory, cache_dir, output_dir):
    '''Materialize a validated ERA5 pressure-level lock into one consolidated Zarr v3 dataset.'''
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fixture = _verify_fixture_first(cache_root)
    runtime = _validate_runtime(contract_lock, inventory)
    if not fixture['paths']:
        raise PipelineError('No verified source fixture was found at cache_dir/source_fixture_manifest.json')

    source = _open_sources(fixture['paths'])
    source = _canonicalize_dataset(source)

    arrays = {}
    channel_entries = []
    for spec in runtime.fields:
        da = _extract_channel(source, spec, runtime)
        arrays[spec.array_name] = da
        channel_entries.append({
            'field_id': spec.field_id,
            'array_path': spec.array_name,
            'selectors': {k: v for k, v in spec.selectors},
            'selector_coordinate_paths': spec.selector_coord_paths,
        })

    public = xr.Dataset(arrays)
    public.attrs.update({
        'dataset_slug': runtime.contract.get('dataset_slug'),
        'dataset_id': POLICY_DATASET_ID,
        'provider': POLICY_PROVIDER,
        'source_url': runtime.contract.get('source_url', ''),
        'publication_format': 'zarr',
        'zarr_format': '3',
    })
    public = _clear_encodings(public)

    store_path = out_root / PUBLIC_STORE
    _publish_zarr_atomic(public, store_path)
    _validate_published(store_path, runtime.fields)

    return {
        'cache': {
            'hits': len(fixture['entries']),
            'misses': 0,
            'acquired': 0,
            'reused_keys': [e['entry_id'] for e in fixture['entries']],
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': PUBLIC_STORE,
            'dimensions': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'coordinates': {'sample': 'time', 'y': 'latitude', 'x': 'longitude'},
            'channels': channel_entries,
        },
        'warnings': [],
    }


def _verify_fixture_first(cache_root: Path) -> dict[str, Any]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return {'entries': [], 'paths': []}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise PipelineError('Invalid source fixture manifest schema_version')
    entries = manifest.get('entries')
    if not isinstance(entries, list) or not entries:
        raise PipelineError('Fixture manifest must contain at least one entry')
    checked = []
    for entry in entries:
        rel = entry.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/') or '..' in Path(rel).parts:
            raise PipelineError('Fixture entry has unsafe relative_path')
        p = (cache_root / rel).resolve()
        if not str(p).startswith(str(cache_root) + os.sep) and p != cache_root:
            raise PipelineError('Fixture path escapes cache_dir')
        if not p.exists() or not p.is_file():
            raise PipelineError(f'Fixture file not found: {rel}')
        expected_size = int(entry.get('size_bytes'))
        if expected_size <= 0 or p.stat().st_size != expected_size:
            raise PipelineError(f'Fixture size mismatch: {rel}')
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        digest = h.hexdigest()
        if digest != entry.get('sha256'):
            raise PipelineError(f'Fixture sha256 mismatch: {rel}')
        entry_id = entry.get('entry_id') or digest[:16]
        checked.append({'entry_id': str(entry_id), 'path': p})
    return {'entries': [{'entry_id': e['entry_id']} for e in checked], 'paths': [e['path'] for e in checked]}


def _select_contract(envelope: Any) -> dict[str, Any]:
    if isinstance(envelope, dict) and envelope.get('schema_version') == 'dataset_contract.v1':
        return envelope
    if not isinstance(envelope, dict):
        raise PipelineError('contract_lock must be a dict-like lock envelope')
    for key in ('contract', 'selected_contract', 'runtime_contract', 'dataset_contract'):
        value = envelope.get(key)
        if isinstance(value, dict) and value.get('schema_version') == 'dataset_contract.v1':
            return value
    contracts = envelope.get('contracts')
    if isinstance(contracts, list):
        matches = [c for c in contracts if isinstance(c, dict) and c.get('schema_version') == 'dataset_contract.v1']
        if len(matches) == 1:
            return matches[0]
        confirmed = [c for c in matches if c.get('human_confirmed') is True]
        if len(confirmed) == 1:
            return confirmed[0]
    raise PipelineError('Could not find one selected dataset_contract.v1 in contract_lock')


def _validate_runtime(contract_lock: Any, inventory: dict[str, Any]) -> RuntimeSpec:
    c = _select_contract(contract_lock)
    if c.get('schema_version') != 'dataset_contract.v1':
        raise PipelineError('Unsupported contract schema')
    if c.get('dataset_slug') != inventory.get('dataset_slug'):
        raise PipelineError('Contract dataset_slug does not match inventory')
    adv = c.get('advanced_options') or {}
    dataset_id = adv.get('dataset_id') or inventory.get('dataset_id')
    if dataset_id != POLICY_DATASET_ID or inventory.get('dataset_id') != POLICY_DATASET_ID:
        raise PipelineError('Dataset id does not match fixed policy')
    if adv.get('data_format', 'grib') != 'grib':
        raise PipelineError('This policy requires GRIB acquisition')
    if adv.get('download_format', 'unarchived') != 'unarchived':
        raise PipelineError('This policy requires unarchived acquisition')

    opts = inventory.get('options') or {}
    scope = c.get('scope') or {}
    product_type = scope.get('product_type', 'reanalysis')
    if product_type not in opts.get('product_type', []):
        raise PipelineError('Invalid product_type for inventory')
    adv_product = adv.get('product_type', [product_type])
    if isinstance(adv_product, str):
        adv_product = [adv_product]
    if list(adv_product) != [product_type]:
        raise PipelineError('scope.product_type and advanced_options.product_type disagree')

    geography = scope.get('geography') or {}
    area = tuple(float(x) for x in geography.get('cds_area', opts.get('area', [90, -180, -90, 180])))
    if len(area) != 4:
        raise PipelineError('geography.cds_area must contain north, west, south, east')
    north, west, south, east = area
    if not (-90 <= south <= north <= 90) or not (-360 <= west <= 360) or not (-360 <= east <= 360):
        raise PipelineError('Invalid geographic bounds')

    target_times = _runtime_times(scope, opts)
    fields = _runtime_fields(c, inventory)
    return RuntimeSpec(contract=c, target_times=tuple(target_times), fields=tuple(fields), area=area)


def _runtime_times(scope: dict[str, Any], opts: dict[str, Any]) -> list[np.datetime64]:
    dr = scope.get('date_range') or {}
    start_s = dr.get('start_date')
    end_s = dr.get('end_date')
    if not start_s or not end_s:
        raise PipelineError('date_range.start_date and end_date are required')
    inclusive = bool(dr.get('inclusive', True))
    time_scope = scope.get('time') or {}
    if time_scope.get('timezone', 'UTC') != 'UTC':
        raise PipelineError('Only UTC selected times are supported')
    selected_times = time_scope.get('selected_times') or []
    if not selected_times:
        raise PipelineError('At least one selected time is required')
    for t in selected_times:
        if t not in opts.get('time', []):
            raise PipelineError(f'Selected time not in inventory options: {t}')

    start_dt = _parse_bound(start_s, end=False)
    end_dt = _parse_bound(end_s, end=True)
    if not inclusive:
        end_dt = end_dt - timedelta(microseconds=1)
    if start_dt > end_dt:
        raise PipelineError('date_range start is after end')

    out = []
    d = start_dt.date()
    while d <= end_dt.date():
        y = f'{d.year:04d}'
        m = f'{d.month:02d}'
        day = f'{d.day:02d}'
        if y not in opts.get('year', []) or m not in opts.get('month', []) or day not in opts.get('day', []):
            raise PipelineError(f'Date outside inventory options: {d.isoformat()}')
        for ts in selected_times:
            hh, mm = [int(x) for x in ts.split(':')]
            dt = datetime.combine(d, time(hh, mm, tzinfo=timezone.utc))
            if start_dt <= dt <= end_dt:
                out.append(np.datetime64(dt.replace(tzinfo=None), 'ns'))
        d += timedelta(days=1)
    if not out:
        raise PipelineError('Runtime selection produced zero samples')
    return out


def _parse_bound(value: str, end: bool) -> datetime:
    if 'T' in value:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    d = date.fromisoformat(value)
    if end:
        return datetime.combine(d, time(23, 59, 59, 999999, tzinfo=timezone.utc))
    return datetime.combine(d, time(0, 0, 0, 0, tzinfo=timezone.utc))


def _runtime_fields(contract: dict[str, Any], inventory: dict[str, Any]) -> list[ChannelSpec]:
    opts = inventory.get('options') or {}
    fields = contract.get('fields') or []
    if not fields:
        raise PipelineError('At least one field is required')
    out = []
    seen = set()
    for f in fields:
        name = f.get('name')
        if name not in opts.get('variable', []):
            raise PipelineError(f'Field not in inventory variable options: {name}')
        pairs = []
        for s in f.get('selectors') or []:
            dim = s.get('dimension')
            val = str(s.get('value'))
            if dim != 'pressure_level':
                raise PipelineError(f'Unsupported selector dimension: {dim}')
            if val not in opts.get('pressure_level', []):
                raise PipelineError(f'pressure_level not in inventory options: {val}')
            if s.get('unit') not in (None, inventory.get('option_units', {}).get('pressure_level', 'hPa'), 'hPa'):
                raise PipelineError('pressure_level selector unit must be hPa')
            pairs.append((dim, val))
        fid = _field_id(name, pairs)
        if fid in seen:
            raise PipelineError(f'Duplicate exact field-selector channel: {fid}')
        seen.add(fid)
        array_name = _array_name(name, pairs)
        selector_paths = {dim: _selector_coord_name(array_name, dim) for dim, _ in pairs}
        out.append(ChannelSpec(name, fid, tuple(pairs), array_name, selector_paths))
    return out


def _field_id(name: str, selectors: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> str:
    if not selectors:
        return name
    inside = ','.join(f'{d}={json.dumps(v, ensure_ascii=False)}' for d, v in selectors)
    return f'{name}[{inside}]'


def _array_name(name: str, selectors: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> str:
    suffix = '__'.join(f'{_safe(d)}_{_safe(v)}' for d, v in selectors)
    return _safe(name) if not suffix else f'{_safe(name)}__{suffix}'


def _selector_coord_name(array_name: str, dim: str) -> str:
    return f'{_safe(dim)}__{array_name}'


def _safe(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(text)).strip('_') or 'value'


def _open_sources(paths: list[Path]) -> xr.Dataset:
    datasets = []
    for p in paths:
        lower = p.name.lower()
        if lower.endswith(('.grib', '.grb', '.grib2', '.grb2')):
            try:
                import cfgrib
                parts = cfgrib.open_datasets(p, backend_kwargs={'indexpath': ''})
                datasets.extend(_canonicalize_dataset(part) for part in parts)
            except Exception as exc:
                raise PipelineError(f'Could not open GRIB fixture {p.name}: {exc}') from exc
        elif lower.endswith(('.nc', '.nc4', '.netcdf')):
            datasets.append(_canonicalize_dataset(xr.open_dataset(p, decode_cf=True, mask_and_scale=True)))
        else:
            raise PipelineError(f'Unsupported fixture file type: {p.name}')
    if not datasets:
        raise PipelineError('No source datasets opened')
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts')


def _canonicalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    present = set(ds.dims) | set(ds.coords)
    for canon, aliases in _COORD_ALIASES.items():
        if canon in present:
            continue
        for a in aliases:
            if a in present:
                ren[a] = canon
                present.add(canon)
                break
    if ren:
        ds = ds.rename(ren)
    if 'pressure_level' in ds.coords and 'pressure_level' not in ds.dims and np.ndim(ds['pressure_level'].values) == 0:
        ds = ds.expand_dims('pressure_level')
    for needed in ('time', 'latitude', 'longitude'):
        if needed not in ds.dims and needed not in ds.coords:
            raise PipelineError(f'Source is missing required coordinate: {needed}')
    return ds


def _extract_channel(ds: xr.Dataset, spec: ChannelSpec, runtime: RuntimeSpec) -> xr.DataArray:
    var = _find_var(ds, spec.field_name)
    da = ds[var]
    da = _select_times(da, runtime.target_times)
    da = _select_area(da, runtime.area)
    for dim, value in spec.selectors:
        da = _select_singleton_selector(da, dim, value, spec.selector_coord_paths[dim])
    desired = ['time'] + [spec.selector_coord_paths[d] for d, _ in spec.selectors] + ['latitude', 'longitude']
    missing = [d for d in desired if d not in da.dims]
    if missing:
        raise PipelineError(f'Channel {spec.field_id} missing dimensions after filtering: {missing}')
    da = da.transpose(*desired)
    da = da.astype('float32', keep_attrs=True)
    da.name = spec.array_name
    da.attrs = dict(da.attrs)
    da.attrs.update({'field_id': spec.field_id, 'source_field_name': spec.field_name})
    da.encoding = {}
    for c in da.coords:
        da.coords[c].encoding = {}
    return da


def _find_var(ds: xr.Dataset, field_name: str) -> str:
    names = _FIELD_SHORT_NAMES.get(field_name, [field_name])
    for n in names:
        if n in ds.data_vars:
            return n
    for v in ds.data_vars:
        attrs = ds[v].attrs
        candidates = [attrs.get('GRIB_shortName'), attrs.get('cfVarName'), attrs.get('standard_name'), attrs.get('long_name')]
        norm = {str(x).lower().replace(' ', '_') for x in candidates if x is not None}
        if field_name in norm or any(n in norm for n in names):
            return v
    raise PipelineError(f'Source fixture does not contain field: {field_name}')


def _select_times(da: xr.DataArray, target_times: tuple[np.datetime64, ...]) -> xr.DataArray:
    if 'time' not in da.coords:
        raise PipelineError('Source variable has no time coordinate')
    have = np.asarray(da['time'].values, dtype='datetime64[ns]')
    need = np.asarray(target_times, dtype='datetime64[ns]')
    missing = [str(t) for t in need if not np.any(have == t)]
    if missing:
        raise PipelineError(f'Source fixture missing selected UTC samples: {missing[:5]}')
    return da.sel(time=need)


def _select_area(da: xr.DataArray, area: tuple[float, float, float, float]) -> xr.DataArray:
    north, west, south, east = area
    if 'latitude' not in da.coords or 'longitude' not in da.coords:
        raise PipelineError('Source variable lacks latitude/longitude coordinates')
    if (north, west, south, east) != (90.0, -180.0, -90.0, 180.0):
        lat = da['latitude'].values
        if lat[0] > lat[-1]:
            da = da.sel(latitude=slice(north, south))
        else:
            da = da.sel(latitude=slice(south, north))
        lon = da['longitude'].values
        if west <= east:
            da = da.sel(longitude=slice(west, east))
        else:
            da = xr.concat([da.sel(longitude=slice(west, lon.max())), da.sel(longitude=slice(lon.min(), east))], dim='longitude')
    return da


def _select_singleton_selector(da: xr.DataArray, dim: str, value: str, public_coord: str) -> xr.DataArray:
    if dim not in da.coords and dim not in da.dims:
        raise PipelineError(f'Source variable lacks selector coordinate: {dim}')
    coord = da[dim]
    if dim in da.dims:
        vals = list(coord.values)
        matches = [i for i, v in enumerate(vals) if _selector_equal(v, value)]
        if len(matches) != 1:
            raise PipelineError(f'Source selector {dim}={value} matched {len(matches)} coordinates')
        da = da.isel({dim: [matches[0]]})
        da = da.rename({dim: public_coord})
        da = da.assign_coords({public_coord: [vals[matches[0]]]})
    else:
        scalar = coord.values.item() if np.ndim(coord.values) == 0 else coord.values
        if not _selector_equal(scalar, value):
            raise PipelineError(f'Source scalar selector {dim} does not equal requested {value}')
        da = da.drop_vars(dim)
        da = da.expand_dims({public_coord: [scalar]})
    da.coords[public_coord].attrs.update({'selector_dimension': dim, 'units': 'hPa'})
    return da


def _selector_equal(a: Any, b: str) -> bool:
    try:
        fa = float(np.asarray(a).item())
        fb = float(b)
        return abs(fa - fb) < 1e-9
    except Exception:
        return str(np.asarray(a).item() if np.ndim(a) == 0 else a) == b


def _clear_encodings(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {}
    return ds


def _chunk_encoding(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    enc = {}
    for name, da in ds.data_vars.items():
        chunks = []
        for d in da.dims:
            n = int(da.sizes[d])
            if d == 'time' or d.startswith('pressure_level__'):
                chunks.append(1)
            elif d == 'longitude':
                chunks.append(n)
            elif d == 'latitude':
                nx = int(da.sizes.get('longitude', 1))
                target = 1536 * 1024
                y = max(1, min(n, target // max(1, nx * 4)))
                if n <= 8:
                    y = n
                chunks.append(int(y))
            else:
                chunks.append(n)
        enc[name] = {'chunks': tuple(chunks), 'dtype': 'float32'}
    for name, da in ds.coords.items():
        if da.dims:
            enc[name] = {'chunks': tuple(int(da.sizes[d]) for d in da.dims)}
    return enc


def _publish_zarr_atomic(ds: xr.Dataset, final_store: Path) -> None:
    parent = final_store.parent
    tmp = Path(tempfile.mkdtemp(prefix='.tmp-zarr-', dir=parent))
    try:
        encoding = _chunk_encoding(ds)
        try:
            ds.to_zarr(tmp, mode='w', consolidated=True, zarr_format=3, encoding=encoding)
        except TypeError:
            ds.to_zarr(tmp, mode='w', consolidated=True, zarr_version=3, encoding=encoding)
        xr.open_zarr(tmp, consolidated=True).close()
        if final_store.exists():
            old = parent / (final_store.name + '.old')
            if old.exists():
                shutil.rmtree(old)
            final_store.rename(old)
            tmp.rename(final_store)
            shutil.rmtree(old)
        else:
            tmp.rename(final_store)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _validate_published(store: Path, fields: tuple[ChannelSpec, ...]) -> None:
    ds = xr.open_zarr(store, consolidated=True)
    try:
        for coord in ('time', 'latitude', 'longitude'):
            if coord not in ds.coords:
                raise PipelineError(f'Published dataset missing coordinate: {coord}')
        for spec in fields:
            if spec.array_name not in ds.data_vars:
                raise PipelineError(f'Published dataset missing channel array: {spec.array_name}')
            arr = ds[spec.array_name]
            expected = ['time'] + [spec.selector_coord_paths[d] for d, _ in spec.selectors] + ['latitude', 'longitude']
            if list(arr.dims) != expected:
                raise PipelineError(f'Published dimension order mismatch for {spec.array_name}')
            _ = arr.isel(time=0).values
    finally:
        ds.close()
