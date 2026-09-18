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

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p007-search-r1'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'era5_pressure_levels.zarr'
FIELD_ALIASES = {
    'temperature': ['temperature', 't'],
    'geopotential': ['geopotential', 'z'],
}
PRESSURE_NAMES = ['pressure_level', 'isobaricInhPa', 'isobaricInPa', 'level', 'plev']
LAT_NAMES = ['latitude', 'lat']
LON_NAMES = ['longitude', 'lon']


class ContractError(ValueError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    contract = _extract_contract(contract_lock)
    runtime = _validate_contract(contract, inventory)
    cache_root = Path(cache_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    fixture_entries = _verify_fixture_manifest(cache_root)
    source_paths = [e['path'] for e in fixture_entries]

    source = _open_source_dataset(source_paths)
    source = _normalise_source_dataset(source)
    filtered = _filter_and_project(source, runtime, inventory)

    final_store = out_root / STORE_NAME
    tmp_store = out_root / (STORE_NAME + '.tmp')
    if tmp_store.exists():
        shutil.rmtree(tmp_store)

    codec_mode, warnings = _write_zarr_v3(filtered['dataset'], tmp_store, filtered['data_vars'])
    _validate_written_store(tmp_store, filtered['dataset'], filtered['data_vars'], codec_mode)

    if final_store.exists():
        shutil.rmtree(final_store)
    os.replace(tmp_store, final_store)
    _validate_written_store(final_store, filtered['dataset'], filtered['data_vars'], codec_mode)

    reused = ['fixture:%s:%s' % (e['entry_id'], e['sha256'][:16]) for e in fixture_entries]
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
            'channels': filtered['channels'],
        },
        'warnings': warnings,
    }


def _extract_contract(lock: Any) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    if isinstance(lock, dict):
        for key in ['contract', 'dataset_contract', 'selected_contract', 'runtime_contract']:
            val = lock.get(key)
            if isinstance(val, dict) and val.get('schema_version') == 'dataset_contract.v1':
                return val
        found: list[dict[str, Any]] = []
        def walk(x: Any) -> None:
            if isinstance(x, dict):
                if x.get('schema_version') == 'dataset_contract.v1':
                    found.append(x)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(lock)
        if len(found) == 1:
            return found[0]
    raise ContractError('no selected dataset_contract.v1 found in lock envelope')


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise ContractError('unsupported inventory schema_version')
    if contract.get('dataset_slug') != DATASET_SLUG or inventory.get('dataset_slug') != DATASET_SLUG:
        raise ContractError('contract and inventory dataset_slug must be %s' % DATASET_SLUG)
    if inventory.get('dataset_id') != DATASET_ID:
        raise ContractError('inventory dataset_id mismatch')
    if contract.get('human_confirmed') is not True:
        raise ContractError('contract must be human_confirmed')

    options = inventory.get('options') or {}
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id') != DATASET_ID:
        raise ContractError('advanced_options.dataset_id mismatch')
    _require_option(options, 'data_format', adv.get('data_format'))
    _require_option(options, 'download_format', adv.get('download_format'))
    for pt in adv.get('product_type') or []:
        _require_option(options, 'product_type', pt)
    if (adv.get('product_type') or []) != (contract.get('scope', {}).get('product_type') and [contract.get('scope', {}).get('product_type')]):
        if contract.get('scope', {}).get('product_type') not in (adv.get('product_type') or []):
            raise ContractError('scope.product_type must match advanced_options.product_type')

    scope = contract.get('scope') or {}
    if scope.get('product_type') != 'reanalysis':
        raise ContractError('this adapter currently supports the ERA5 reanalysis product_type in this inventory')
    _require_option(options, 'product_type', scope.get('product_type'))

    time_scope = scope.get('time') or {}
    if time_scope.get('timezone') != 'UTC':
        raise ContractError('time.timezone must be UTC')
    selected_times = time_scope.get('selected_times') or []
    if not selected_times:
        raise ContractError('at least one selected time is required')
    for t in selected_times:
        _parse_hhmm(t)
        _require_option(options, 'time', t)

    dr = scope.get('date_range') or {}
    start_date = _parse_date(dr.get('start_date'))
    end_date = _parse_date(dr.get('end_date'))
    if end_date < start_date:
        raise ContractError('end_date precedes start_date')
    expected_times = _expected_timestamps(start_date, end_date, bool(dr.get('inclusive', True)), selected_times)
    if not expected_times:
        raise ContractError('date/time selection is empty')
    for ts in expected_times:
        _require_option(options, 'year', ts.strftime('%Y'))
        _require_option(options, 'month', ts.strftime('%m'))
        _require_option(options, 'day', ts.strftime('%d'))

    geo = scope.get('geography') or {}
    area = geo.get('cds_area') if 'cds_area' in geo else geo.get('area')
    if geo.get('area') == 'global':
        area = geo.get('cds_area', [90, -180, -90, 180])
    if geo.get('cds_area_order', ['north', 'west', 'south', 'east']) != ['north', 'west', 'south', 'east']:
        raise ContractError('cds_area_order must be north/west/south/east')
    area = _validate_area(area)

    fields = contract.get('fields') or []
    if not fields:
        raise ContractError('at least one field is required')
    seen: set[str] = set()
    field_specs = []
    for f in fields:
        name = f.get('name')
        _require_option(options, 'variable', name)
        selectors = f.get('selectors') or []
        if len(selectors) != 1 or selectors[0].get('dimension') != 'pressure_level':
            raise ContractError('ERA5 pressure-level fields require exactly one pressure_level selector')
        val = str(selectors[0].get('value'))
        _require_option(options, 'pressure_level', val)
        if selectors[0].get('unit') not in (None, 'hPa'):
            raise ContractError('pressure_level selector unit must be hPa when supplied')
        fid = _field_id(name, selectors)
        if fid in seen:
            raise ContractError('duplicate requested field-selector channel: %s' % fid)
        seen.add(fid)
        field_specs.append({'name': name, 'pressure_level': val, 'selectors': selectors, 'field_id': fid})

    return {
        'expected_times': expected_times,
        'area': area,
        'fields': field_specs,
        'selected_times': selected_times,
        'contract': contract,
    }


def _require_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value not in (options.get(key) or []):
        raise ContractError('invalid %s option: %r' % (key, value))


def _parse_date(s: Any) -> date:
    if not isinstance(s, str):
        raise ContractError('date bounds must be ISO date strings')
    try:
        return date.fromisoformat(s)
    except Exception as exc:
        raise ContractError('invalid date: %r' % s) from exc


def _parse_hhmm(s: Any) -> time:
    if not isinstance(s, str) or not re.match(r'^\d{2}:\d{2}$', s):
        raise ContractError('times must be HH:MM strings')
    try:
        return time.fromisoformat(s)
    except Exception as exc:
        raise ContractError('invalid time: %r' % s) from exc


def _expected_timestamps(start: date, end: date, inclusive: bool, selected_times: list[str]) -> list[pd.Timestamp]:
    stop = end if inclusive else end - timedelta(days=1)
    out: list[pd.Timestamp] = []
    d = start
    while d <= stop:
        for hhmm in selected_times:
            tt = _parse_hhmm(hhmm)
            out.append(pd.Timestamp(datetime.combine(d, tt, tzinfo=timezone.utc)).tz_convert(None))
        d += timedelta(days=1)
    return out


def _validate_area(area: Any) -> list[float]:
    if not isinstance(area, list) or len(area) != 4:
        raise ContractError('cds_area must contain four numeric values')
    vals = [float(v) for v in area]
    north, west, south, east = vals
    if not (-90 <= south <= north <= 90):
        raise ContractError('invalid latitude bounds')
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ContractError('invalid longitude bounds')
    return vals


def _field_id(name: str, selectors: list[dict[str, Any]]) -> str:
    if not selectors:
        return name
    body = ','.join('%s=%s' % (s['dimension'], json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':'))) for s in selectors)
    return '%s[%s]' % (name, body)


def _safe_name(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9_]+', '_', text).strip('_')
    return text or 'array'


def _verify_fixture_manifest(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('complete local source fixture is required: source_fixture_manifest.json is missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ContractError('invalid source fixture manifest schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise ContractError('source fixture manifest contains no entries')
    verified = []
    for e in entries:
        rel = e.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/'):
            raise ContractError('fixture relative_path must be relative')
        path = (cache_root / rel).resolve()
        try:
            path.relative_to(cache_root)
        except ValueError as exc:
            raise ContractError('fixture path escapes cache_dir') from exc
        if not path.is_file() and not path.is_dir():
            raise FileNotFoundError('fixture entry missing: %s' % rel)
        actual_size = _path_size(path)
        expected_size = int(e.get('size_bytes'))
        if actual_size != expected_size:
            raise ContractError('fixture size mismatch for %s' % rel)
        sha = _sha256_path(path)
        expected_sha = str(e.get('sha256', '')).lower()
        if sha != expected_sha:
            raise ContractError('fixture sha256 mismatch for %s' % rel)
        verified.append({'entry_id': str(e.get('entry_id') or rel), 'path': path, 'sha256': sha, 'size_bytes': actual_size})
    return verified


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in sorted(x for x in path.rglob('*') if x.is_file()):
        total += p.stat().st_size
    return total


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        _hash_file(h, path)
    else:
        for p in sorted(x for x in path.rglob('*') if x.is_file()):
            rel = p.relative_to(path).as_posix().encode('utf-8')
            h.update(rel + b'\0')
            _hash_file(h, p)
    return h.hexdigest()


def _hash_file(h: Any, path: Path) -> None:
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)


def _open_source_dataset(paths: list[Path]) -> xr.Dataset:
    datasets = []
    for p in paths:
        if p.is_dir() or p.suffix.lower() == '.zarr':
            ds = xr.open_zarr(p, consolidated=False, mask_and_scale=True)
        elif p.suffix.lower() in ['.nc', '.netcdf', '.h5', '.hdf5']:
            try:
                ds = xr.open_dataset(p, engine='h5netcdf', decode_cf=True, mask_and_scale=True)
            except Exception:
                ds = xr.open_dataset(p, decode_cf=True, mask_and_scale=True)
        else:
            try:
                ds = xr.open_dataset(p, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True)
            except Exception as exc:
                raise RuntimeError('failed to decode fixture source file %s as GRIB/CF dataset' % p.name) from exc
        datasets.append(ds.load())
        ds.close()
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, data_vars='all', coords='minimal', compat='no_conflicts', combine_attrs='drop_conflicts').load()


def _normalise_source_dataset(ds: xr.Dataset) -> xr.Dataset:
    ren: dict[str, str] = {}
    for names, target in [(PRESSURE_NAMES, 'pressure_level'), (LAT_NAMES, 'latitude'), (LON_NAMES, 'longitude')]:
        for n in names:
            if n in ds.dims or n in ds.coords:
                if n != target and target not in ds.dims and target not in ds.coords:
                    ren[n] = target
                break
    if 'valid_time' in ds.coords and 'time' not in ds.coords and 'time' not in ds.dims:
        ren['valid_time'] = 'time'
    if ren:
        ds = ds.rename(ren)
    for need in ['time', 'latitude', 'longitude', 'pressure_level']:
        if need not in ds.coords and need not in ds.dims:
            raise ContractError('source fixture is missing required coordinate/dimension %s' % need)
    return ds


def _filter_and_project(ds: xr.Dataset, runtime: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    expected_np = np.array([np.datetime64(t.to_datetime64(), 'ns') for t in runtime['expected_times']])
    source_times = ds['time'].values.astype('datetime64[ns]')
    missing = [str(t) for t in expected_np if t not in set(source_times)]
    if missing:
        raise ContractError('source fixture does not contain all requested timestamps; first missing %s' % missing[0])
    ds = ds.sel(time=expected_np)
    ds = _select_area(ds, runtime['area'])

    out_vars: dict[str, xr.DataArray] = {}
    channels: list[dict[str, Any]] = []
    data_var_names: list[str] = []
    for spec in runtime['fields']:
        array_name = _safe_name(spec['name'] + '__pressure_level_' + spec['pressure_level'])
        src_name = array_name if array_name in ds.data_vars else _find_source_variable(ds, spec['name'])
        arr = ds[src_name]
        if 'pressure_level' in arr.dims:
            level_value = _pressure_value(arr['pressure_level'], spec['pressure_level'])
            arr = arr.sel(pressure_level=[level_value])
        elif 'pressure_level' in arr.coords:
            level_value = _pressure_value(arr['pressure_level'], spec['pressure_level'])
            arr = arr.expand_dims({'pressure_level': [level_value]})
        elif 'pressure_level' in ds.coords:
            level_value = _pressure_value(ds['pressure_level'], spec['pressure_level'])
            arr = arr.expand_dims({'pressure_level': [level_value]})
        else:
            raise ContractError('source fixture is missing required coordinate/dimension pressure_level for %s' % src_name)
        dim_name = array_name + '__pressure_level'
        arr = arr.rename({'pressure_level': dim_name})
        wanted_dims = [d for d in ['time', dim_name, 'latitude', 'longitude'] if d in arr.dims]
        arr = arr.transpose(*wanted_dims)
        arr.name = array_name
        arr.attrs = dict(arr.attrs)
        arr.attrs.setdefault('source_variable_name', src_name)
        arr.attrs['field_id'] = spec['field_id']
        arr.attrs['selector_dimension'] = 'pressure_level'
        arr.attrs['selector_value'] = spec['pressure_level']
        arr.encoding = {}
        out_vars[array_name] = arr
        data_var_names.append(array_name)
        channels.append({
            'field_id': spec['field_id'],
            'array_path': array_name,
            'selectors': {'pressure_level': spec['pressure_level']},
            'selector_coordinate_paths': {'pressure_level': dim_name},
        })

    out = xr.Dataset(out_vars)
    for coord in ['time', 'latitude', 'longitude']:
        out[coord].attrs = dict(ds[coord].attrs)
        out[coord].encoding = {}
    for name in data_var_names:
        sel_dim = [d for d in out[name].dims if d.endswith('__pressure_level')][0]
        out[sel_dim].attrs = dict(ds['pressure_level'].attrs)
        out[sel_dim].attrs.setdefault('units', 'hPa')
        out[sel_dim].encoding = {}
    out.attrs = {
        'dataset_slug': DATASET_SLUG,
        'dataset_id': DATASET_ID,
        'publication_format': 'zarr_v3_consolidated',
        'grid_mapping': 'regular_rectilinear_latitude_longitude',
        'channel_field_ids': json.dumps([c['field_id'] for c in channels], ensure_ascii=False, separators=(',', ':')),
    }
    return {'dataset': out, 'channels': channels, 'data_vars': data_var_names}


def _find_source_variable(ds: xr.Dataset, requested: str) -> str:
    aliases = FIELD_ALIASES.get(requested, [requested])
    for name in aliases:
        if name in ds.data_vars:
            return name
    for var in ds.data_vars:
        attrs = {str(k).lower(): str(v).lower() for k, v in ds[var].attrs.items()}
        values = set(attrs.values())
        if requested.lower() in values or requested.lower().replace('_', ' ') in values:
            return var
    raise ContractError('source fixture is missing requested variable %s' % requested)


def _pressure_value(coord: xr.DataArray, requested: str) -> Any:
    vals = np.asarray(coord.values).ravel()
    for v in vals:
        if str(v) == requested:
            return v.item() if hasattr(v, 'item') else v
    try:
        req = float(requested)
        for v in vals:
            if float(v) == req:
                return v.item() if hasattr(v, 'item') else v
    except Exception:
        pass
    raise ContractError('source fixture is missing requested pressure_level %s' % requested)


def _select_area(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    north, west, south, east = area
    if [north, west, south, east] == [90.0, -180.0, -90.0, 180.0]:
        return ds
    lat = ds['latitude']
    ds = ds.where((lat >= south) & (lat <= north), drop=True)
    lon = ds['longitude']
    lon_vals = lon.values.astype(float)
    if lon_vals.min() >= 0 and (west < 0 or east < 0):
        w = west % 360
        e = east % 360
    else:
        w, e = west, east
    if w <= e:
        mask = (lon >= w) & (lon <= e)
    else:
        mask = (lon >= w) | (lon <= e)
    return ds.where(mask, drop=True)


def _write_zarr_v3(ds: xr.Dataset, tmp_store: Path, data_vars: list[str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for mode in ['uncompressed_empty_tuple', 'uncompressed_none']:
        if tmp_store.exists():
            shutil.rmtree(tmp_store)
        enc = _encoding(ds, data_vars, mode)
        try:
            ds2 = ds.copy(deep=False)
            ds2.attrs = dict(ds.attrs)
            ds2.attrs['publication_data_chunk_codecs'] = 'uncompressed'
            ds2.to_zarr(tmp_store, mode='w', consolidated=True, zarr_format=3, encoding=enc)
            return 'uncompressed', warnings
        except TypeError:
            continue
        except ValueError:
            continue
    if tmp_store.exists():
        shutil.rmtree(tmp_store)
    enc = _encoding(ds, data_vars, 'blosc_lz4')
    ds2 = ds.copy(deep=False)
    ds2.attrs = dict(ds.attrs)
    ds2.attrs['publication_data_chunk_codecs'] = 'blosc_lz4_clevel1_shuffle_lossless'
    ds2.to_zarr(tmp_store, mode='w', consolidated=True, zarr_format=3, encoding=enc)
    warnings.append('uncompressed Zarr v3 chunks were not accepted by the host xarray/zarr API; used lossless Blosc-LZ4 clevel=1 byte-shuffle fallback')
    return 'blosc_lz4', warnings


def _encoding(ds: xr.Dataset, data_vars: list[str], mode: str) -> dict[str, dict[str, Any]]:
    enc: dict[str, dict[str, Any]] = {}
    for name in ds.variables:
        var = ds[name]
        e: dict[str, Any] = {'chunks': _chunks_for_var(var, name in data_vars)}
        if mode == 'uncompressed_empty_tuple':
            e['compressors'] = ()
        elif mode == 'uncompressed_none':
            e['compressors'] = None
        else:
            from zarr.codecs import BloscCodec
            e['compressors'] = (BloscCodec(cname='lz4', clevel=1, shuffle='shuffle'),)
        if name in data_vars:
            e['_FillValue'] = None
        enc[name] = e
    return enc


def _chunks_for_var(var: xr.DataArray, is_data: bool) -> tuple[int, ...]:
    if not is_data:
        return tuple(int(var.sizes[d]) for d in var.dims)
    chunks = []
    for d in var.dims:
        if d == 'time' or d.endswith('__pressure_level'):
            chunks.append(1)
        else:
            chunks.append(int(var.sizes[d]))
    return tuple(chunks)


def _validate_written_store(store: Path, expected: xr.Dataset, data_vars: list[str], codec_mode: str) -> None:
    root_meta = json.loads((store / 'zarr.json').read_text(encoding='utf-8'))
    if 'consolidated_metadata' not in root_meta:
        raise RuntimeError('Zarr v3 root metadata is not consolidated')
    reopened = xr.open_zarr(store, consolidated=True, mask_and_scale=True)
    try:
        for name in expected.data_vars:
            if name not in reopened.data_vars:
                raise RuntimeError('published data variable missing: %s' % name)
            if tuple(reopened[name].dims) != tuple(expected[name].dims):
                raise RuntimeError('dimension mismatch for %s' % name)
            if not _array_equal_exact(reopened[name].values, expected[name].values):
                raise RuntimeError('value mismatch for %s' % name)
            for d in expected[name].dims:
                if not _array_equal_exact(reopened[d].values, expected[d].values):
                    raise RuntimeError('coordinate mismatch for %s' % d)
            meta = json.loads((store / name / 'zarr.json').read_text(encoding='utf-8'))
            if tuple(meta['chunk_grid']['configuration']['chunk_shape']) != _chunks_for_var(expected[name], True):
                raise RuntimeError('chunk layout mismatch for %s' % name)
            compressed = _metadata_has_compression(meta)
            if codec_mode == 'uncompressed' and compressed:
                raise RuntimeError('data array %s is compressed despite uncompressed publication mode' % name)
            if codec_mode == 'blosc_lz4' and not _metadata_has_lz4(meta):
                raise RuntimeError('data array %s lacks declared LZ4 fallback codec' % name)
    finally:
        reopened.close()


def _array_equal_exact(a: Any, b: Any) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _metadata_has_compression(meta: dict[str, Any]) -> bool:
    text = json.dumps(meta.get('codecs', []), sort_keys=True).lower()
    return any(x in text for x in ['blosc', 'gzip', 'zstd', 'zlib', 'lz4'])


def _metadata_has_lz4(meta: dict[str, Any]) -> bool:
    text = json.dumps(meta.get('codecs', []), sort_keys=True).lower()
    return 'blosc' in text and 'lz4' in text and ('clevel' in text or 'level' in text)
