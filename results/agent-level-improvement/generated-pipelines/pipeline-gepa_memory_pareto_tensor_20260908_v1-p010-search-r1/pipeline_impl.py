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

PIPELINE_ID = 'pipeline-gepa_memory_pareto_tensor_20260908_v1-p010-search-r1'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
STORE_NAME = 'data.zarr'
TMP_STORE_NAME = '_tmp_publish_reanalysis_era5_pressure_levels.zarr'
BLOSC_CNAME = 'zstd'
BLOSC_CLEVEL = 9
BLOSC_SHUFFLE_NAME = 'bitshuffle'
SHORT_NAMES = {'t': 'temperature', 'z': 'geopotential'}
CANONICAL_DIMS = {'sample': 'time', 'y': 'latitude', 'x': 'longitude'}


class PipelineValidationError(ValueError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str, output_dir: str) -> dict[str, Any]:
    contract, policy = _extract_contract_and_policy(contract_lock)
    _validate_inventory(inventory)
    _validate_contract(contract, inventory)
    _validate_policy(policy, contract)

    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = _verify_source_fixture(cache_root)
    if not manifest_entries:
        raise PipelineValidationError('No complete verified local source fixture is present; network acquisition is not implemented or attempted.')

    source = _open_fixture_dataset(cache_root, manifest_entries)
    source = _normalise_dataset(source)
    requested_times = _selected_datetimes(contract)
    source = _filter_time_and_area(source, contract, requested_times)
    published, channels = _materialise_requested_channels(source, contract, inventory)
    published = _strip_repacking_encodings(published)

    tmp_store = output_root / TMP_STORE_NAME
    final_store = output_root / STORE_NAME
    _remove_tree(tmp_store)
    _write_zarr_v3(published, tmp_store)
    reopened = xr.open_zarr(tmp_store, consolidated=True)
    try:
        _assert_semantically_equal(published, reopened)
        _assert_codec_configuration(tmp_store, channels)
    finally:
        reopened.close()

    _publish_directory_atomically(tmp_store, final_store)
    reopened_final = xr.open_zarr(final_store, consolidated=True)
    try:
        _assert_semantically_equal(published, reopened_final)
    finally:
        reopened_final.close()

    reused = [_safe_cache_key(e) for e in manifest_entries]
    return {
        'cache': {
            'hits': len(manifest_entries),
            'misses': 0,
            'acquired': 0,
            'reused_keys': reused,
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': STORE_NAME,
            'dimensions': dict(CANONICAL_DIMS),
            'coordinates': dict(CANONICAL_DIMS),
            'channels': channels,
        },
        'warnings': [],
    }


def _extract_contract_and_policy(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(lock, dict):
        raise PipelineValidationError('contract_lock must be a mapping')
    if lock.get('schema_version') == 'dataset_contract.v1':
        return lock, lock.get('fixed_pipeline_policy') or lock.get('policy')
    candidates = ['contract', 'dataset_contract', 'selected_contract', 'locked_contract']
    contract = None
    for key in candidates:
        val = lock.get(key)
        if isinstance(val, dict) and val.get('schema_version') == 'dataset_contract.v1':
            contract = val
            break
    if contract is None:
        for val in lock.values():
            if isinstance(val, dict) and val.get('schema_version') == 'dataset_contract.v1':
                contract = val
                break
    if contract is None:
        raise PipelineValidationError('No dataset_contract.v1 object found in contract_lock')
    policy = lock.get('fixed_pipeline_policy') or lock.get('pipeline_policy') or lock.get('policy')
    return contract, policy if isinstance(policy, dict) else None


def _validate_inventory(inv: dict[str, Any]) -> None:
    if inv.get('schema_version') != 'dataset_inventory.v1':
        raise PipelineValidationError('Unsupported inventory schema_version')
    if inv.get('dataset_slug') != DATASET_SLUG or inv.get('dataset_id') != DATASET_ID:
        raise PipelineValidationError('Inventory identity does not match ERA5 pressure-level adapter')


def _validate_policy(policy: dict[str, Any] | None, contract: dict[str, Any]) -> None:
    adv = contract.get('advanced_options') or {}
    if adv.get('dataset_id') != DATASET_ID:
        raise PipelineValidationError('advanced_options.dataset_id mismatch')
    if adv.get('data_format') != 'grib' or adv.get('download_format') != 'unarchived':
        raise PipelineValidationError('This adapter is locked to unarchived GRIB acquisition semantics')
    if adv.get('product_type') != ['reanalysis']:
        raise PipelineValidationError('Only reanalysis product_type is supported by the fixed policy')
    if policy is None:
        return
    if policy.get('provider') not in (None, 'ECMWF'):
        raise PipelineValidationError('Fixed policy provider mismatch')
    if policy.get('dataset_id') not in (None, DATASET_ID):
        raise PipelineValidationError('Fixed policy dataset_id mismatch')
    if policy.get('acquisition_format') not in (None, 'grib'):
        raise PipelineValidationError('Fixed policy acquisition format mismatch')
    if policy.get('publication_format') not in (None, 'zarr'):
        raise PipelineValidationError('Fixed policy publication format mismatch')
    zpol = policy.get('zarr') or policy.get('zarr_output_policy') or {}
    if zpol:
        if zpol.get('format_version') != 3 or zpol.get('consolidated_metadata') is not True:
            raise PipelineValidationError('Fixed Zarr v3 consolidated publication policy mismatch')


def _validate_contract(contract: dict[str, Any], inv: dict[str, Any]) -> None:
    if contract.get('dataset_slug') != DATASET_SLUG:
        raise PipelineValidationError('Contract dataset_slug mismatch')
    if not contract.get('human_confirmed'):
        raise PipelineValidationError('Contract must be human_confirmed')
    scope = contract.get('scope') or {}
    if scope.get('product_type') != 'reanalysis':
        raise PipelineValidationError('Only product_type=reanalysis is valid for this fixed policy')
    geo = scope.get('geography') or {}
    area = geo.get('cds_area')
    if not (isinstance(area, list) and len(area) == 4 and all(isinstance(x, (int, float)) for x in area)):
        raise PipelineValidationError('scope.geography.cds_area must contain four numeric bounds')
    north, west, south, east = [float(x) for x in area]
    if north < south or north > 90 or south < -90 or west < -360 or east > 360:
        raise PipelineValidationError('Invalid geographic bounds')
    t = scope.get('time') or {}
    if t.get('timezone') != 'UTC':
        raise PipelineValidationError('Only UTC time selections are supported')
    selected_times = t.get('selected_times') or []
    inv_times = set(inv.get('options', {}).get('time', []))
    if not selected_times or any(x not in inv_times for x in selected_times):
        raise PipelineValidationError('Invalid selected time')
    dr = scope.get('date_range') or {}
    _parse_date(dr.get('start_date'))
    _parse_date(dr.get('end_date'))
    if _parse_date(dr.get('end_date')) < _parse_date(dr.get('start_date')):
        raise PipelineValidationError('date_range end before start')
    years = set(inv.get('options', {}).get('year', []))
    for ts in _selected_datetimes(contract):
        if f'{ts.year:04d}' not in years:
            raise PipelineValidationError('Selected year outside frozen inventory options')
    inv_vars = set(inv.get('options', {}).get('variable', []))
    inv_levels = set(inv.get('options', {}).get('pressure_level', []))
    seen = set()
    fields = contract.get('fields') or []
    if not fields:
        raise PipelineValidationError('At least one field must be requested')
    for f in fields:
        name = f.get('name')
        if name not in inv_vars:
            raise PipelineValidationError(f'Invalid variable requested: {name}')
        sels = f.get('selectors') or []
        if len(sels) != 1 or sels[0].get('dimension') != 'pressure_level':
            raise PipelineValidationError('ERA5 pressure-level fields must declare one pressure_level selector')
        val = str(sels[0].get('value'))
        if val not in inv_levels:
            raise PipelineValidationError(f'Invalid pressure_level requested: {val}')
        if sels[0].get('unit') not in (None, 'hPa'):
            raise PipelineValidationError('pressure_level selector unit must be hPa')
        fid = _field_id(f)
        if fid in seen:
            raise PipelineValidationError(f'Duplicate requested field selector: {fid}')
        seen.add(fid)


def _parse_date(x: str) -> date:
    try:
        return date.fromisoformat(str(x))
    except Exception as exc:
        raise PipelineValidationError(f'Invalid date: {x}') from exc


def _selected_datetimes(contract: dict[str, Any]) -> list[pd.Timestamp]:
    scope = contract['scope']
    dr = scope['date_range']
    start = _parse_date(dr['start_date'])
    end = _parse_date(dr['end_date'])
    inclusive = bool(dr.get('inclusive', True))
    if not inclusive:
        end = end - timedelta(days=1)
    selected_times = sorted(scope['time']['selected_times'])
    out: list[pd.Timestamp] = []
    d = start
    while d <= end:
        for s in selected_times:
            hh, mm = [int(p) for p in s.split(':')]
            out.append(pd.Timestamp(datetime.combine(d, time(hh, mm), tzinfo=timezone.utc)).tz_convert(None))
        d += timedelta(days=1)
    return out


def _verify_source_fixture(cache_root: Path) -> list[dict[str, Any]]:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise PipelineValidationError('Invalid source fixture manifest schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise PipelineValidationError('Fixture manifest has no entries')
    verified: list[dict[str, Any]] = []
    for e in entries:
        rel = e.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/') or '..' in Path(rel).parts:
            raise PipelineValidationError('Unsafe fixture relative_path')
        p = (cache_root / rel).resolve()
        if not str(p).startswith(str(cache_root) + os.sep) and p != cache_root:
            raise PipelineValidationError('Fixture path escapes cache root')
        if not p.exists() or not p.is_file():
            raise PipelineValidationError('Fixture file listed in manifest is missing')
        size = p.stat().st_size
        if size != int(e.get('size_bytes')):
            raise PipelineValidationError('Fixture file size mismatch')
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha != e.get('sha256'):
            raise PipelineValidationError('Fixture file sha256 mismatch')
        ve = dict(e)
        ve['_path'] = p
        verified.append(ve)
    return sorted(verified, key=lambda x: str(x.get('relative_path')))


def _open_fixture_dataset(cache_root: Path, entries: list[dict[str, Any]]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for e in entries:
        p = e['_path']
        suffix = p.suffix.lower()
        if suffix in ('.nc', '.nc4', '.cdf', '.netcdf'):
            datasets.append(xr.open_dataset(p, decode_cf=True, mask_and_scale=True, engine='h5netcdf').load())
        elif suffix == '.zarr':
            datasets.append(xr.open_zarr(p, consolidated=False).load())
        elif suffix in ('.grib', '.grb', '.grb2'):
            try:
                datasets.append(xr.open_dataset(p, engine='cfgrib', backend_kwargs={'indexpath': ''}).load())
            except Exception as exc:
                raise PipelineValidationError('GRIB fixture could not be decoded with cfgrib/eccodes') from exc
        else:
            raise PipelineValidationError(f'Unsupported fixture file extension: {suffix}')
    datasets = [_normalise_dataset(ds) for ds in datasets]
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts').load()


def _normalise_dataset(ds: xr.Dataset) -> xr.Dataset:
    ren: dict[str, str] = {}
    for name in list(ds.dims) + list(ds.coords):
        low = str(name).lower()
        if low in ('valid_time', 'time') and name != 'time':
            if low != 'valid_time' or ('time' not in ds.dims and 'time' not in ds.coords):
                ren[name] = 'time'
        elif low in ('isobaricinhpa', 'level', 'pressure', 'plev') and name != 'pressure_level':
            ren[name] = 'pressure_level'
        elif low in ('lat', 'latitude') and name != 'latitude':
            ren[name] = 'latitude'
        elif low in ('lon', 'longitude') and name != 'longitude':
            ren[name] = 'longitude'
    if ren:
        ds = ds.rename(ren)
    vren = {k: v for k, v in SHORT_NAMES.items() if k in ds.data_vars and v not in ds.data_vars}
    if vren:
        ds = ds.rename(vren)
    if 'pressure_level' in ds.coords and 'pressure_level' not in ds.dims:
        if np.ndim(ds['pressure_level'].values) == 0:
            ds = ds.expand_dims('pressure_level')
    required = {'time', 'pressure_level', 'latitude', 'longitude'}
    if not required.issubset(set(ds.coords) | set(ds.dims)):
        raise PipelineValidationError('Source fixture lacks required ERA5 grid coordinates')
    return ds


def _filter_time_and_area(ds: xr.Dataset, contract: dict[str, Any], times: list[pd.Timestamp]) -> xr.Dataset:
    wanted = pd.DatetimeIndex(times)
    available = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
    missing = wanted.difference(available)
    if len(missing):
        raise PipelineValidationError('Source fixture does not cover all selected timestamps')
    ds = ds.sel(time=wanted)
    area = [float(x) for x in contract['scope']['geography']['cds_area']]
    north, west, south, east = area
    lat = ds['latitude']
    lon = ds['longitude']
    if not (north == 90 and south == -90 and west == -180 and east == 180):
        if lat.values[0] > lat.values[-1]:
            ds = ds.sel(latitude=slice(north, south))
        else:
            ds = ds.sel(latitude=slice(south, north))
        lon_vals = lon.values.astype(float)
        w, e = west, east
        if lon_vals.min() >= 0 and w < 0:
            w = (w + 360) % 360
            e = (e + 360) % 360
        if w <= e:
            ds = ds.sel(longitude=slice(w, e))
        else:
            left = ds.sel(longitude=slice(w, float(lon_vals.max())))
            right = ds.sel(longitude=slice(float(lon_vals.min()), e))
            ds = xr.concat([left, right], dim='longitude')
    if ds.sizes.get('latitude', 0) == 0 or ds.sizes.get('longitude', 0) == 0:
        raise PipelineValidationError('Selected geography produced an empty grid')
    return ds


def _field_id(field: dict[str, Any]) -> str:
    sels = field.get('selectors') or []
    if not sels:
        return str(field['name'])
    parts = []
    for s in sels:
        parts.append(str(s['dimension']) + '=' + json.dumps(str(s['value']), ensure_ascii=False, separators=(',', ':')))
    return str(field['name']) + '[' + ','.join(parts) + ']'


def _safe_dim_name(fid: str, selector_name: str) -> str:
    clean = re.sub('[^0-9A-Za-z_]+', '_', fid).strip('_')
    return selector_name + '__' + clean[:80]


def _materialise_requested_channels(ds: xr.Dataset, contract: dict[str, Any], inv: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    data_vars: dict[str, xr.DataArray] = {}
    coords: dict[str, xr.DataArray] = {
        'time': ds['time'],
        'latitude': ds['latitude'],
        'longitude': ds['longitude'],
    }
    channels: list[dict[str, Any]] = []
    for field in contract['fields']:
        name = field['name']
        if name not in ds.data_vars:
            raise PipelineValidationError(f'Source fixture lacks requested variable {name}')
        sel = field['selectors'][0]
        level_str = str(sel['value'])
        level_num: int | float = int(level_str) if level_str.isdigit() else float(level_str)
        fid = _field_id(field)
        sdim = _safe_dim_name(fid, 'pressure_level')
        da = ds[name].sel(pressure_level=[level_num])
        da = da.rename({'pressure_level': sdim})
        da = da.transpose('time', sdim, 'latitude', 'longitude')
        da.name = fid
        da.attrs = dict(ds[name].attrs)
        meta = inv.get('option_metadata', {}).get('variable', {}).get(name, {})
        if 'units' not in da.attrs and meta.get('units'):
            da.attrs['units'] = meta['units']
        da.attrs['field_name'] = name
        da.attrs['pressure_level'] = level_str
        da.attrs['pressure_level_units'] = 'hPa'
        data_vars[fid] = da
        coord = da[sdim]
        coord.attrs = dict(coord.attrs)
        coord.attrs['units'] = 'hPa'
        coords[sdim] = coord
        channels.append({
            'field_id': fid,
            'array_path': fid,
            'selectors': {'pressure_level': level_str},
            'selector_coordinate_paths': {'pressure_level': sdim},
        })
    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        'dataset_slug': DATASET_SLUG,
        'dataset_id': DATASET_ID,
        'publication_format': 'zarr_v3_consolidated',
    })
    return out, channels


def _strip_repacking_encodings(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    for v in list(ds.variables):
        ds[v].encoding = {k: val for k, val in ds[v].encoding.items() if k in ('dtype',)}
        for bad in ('scale_factor', 'add_offset', '_FillValue', 'missing_value', 'compressor', 'compressors'):
            ds[v].encoding.pop(bad, None)
    return ds


def _blosc_codec() -> Any:
    from zarr.codecs import BloscCodec, BloscShuffle
    shuffle = getattr(BloscShuffle, BLOSC_SHUFFLE_NAME)
    return BloscCodec(cname=BLOSC_CNAME, clevel=BLOSC_CLEVEL, shuffle=shuffle)


def _write_zarr_v3(ds: xr.Dataset, store: Path) -> None:
    codec = _blosc_codec()
    encoding: dict[str, dict[str, Any]] = {}
    y = int(ds.sizes['latitude'])
    x = int(ds.sizes['longitude'])
    for name, da in ds.data_vars.items():
        if da.dims[1].startswith('pressure_level__'):
            chunks = (1, 1, y, x)
        else:
            chunks = (1, y, x)
        encoding[name] = {'chunks': chunks, 'compressors': (codec,)}
    for cname in ds.coords:
        encoding.setdefault(cname, {})['chunks'] = tuple(int(ds.sizes[d]) for d in ds[cname].dims) or None
    ds.to_zarr(store, mode='w', zarr_format=3, consolidated=True, encoding=encoding, compute=True)


def _assert_semantically_equal(expected: xr.Dataset, actual: xr.Dataset) -> None:
    if set(expected.data_vars) != set(actual.data_vars):
        raise AssertionError('Published data variables differ after read-back')
    if set(expected.coords) != set(actual.coords):
        raise AssertionError('Published coordinates differ after read-back')
    for name in expected.variables:
        e = expected[name]
        a = actual[name]
        if e.dims != a.dims:
            raise AssertionError(f'Dimensions differ for {name}')
        np.testing.assert_array_equal(e.values, a.values)
        for key in ('units', 'field_name', 'pressure_level', 'pressure_level_units'):
            if key in e.attrs and a.attrs.get(key) != e.attrs.get(key):
                raise AssertionError(f'Attribute {key} differs for {name}')


def _assert_codec_configuration(store: Path, channels: list[dict[str, Any]]) -> None:
    import zarr
    root = zarr.open_group(store, mode='r')
    for ch in channels:
        arr = root[ch['array_path']]
        chunks = tuple(arr.chunks)
        if len(chunks) == 4 and (chunks[0] != 1 or chunks[1] != 1):
            raise AssertionError('Data chunks are not sample/selector aligned')
        meta = arr.metadata
        codecs = getattr(meta, 'codecs', None) or []
        found = False
        for c in codecs:
            rep = c.to_dict() if hasattr(c, 'to_dict') else (c if isinstance(c, dict) else {})
            text = json.dumps(rep, sort_keys=True, default=str).lower()
            if 'blosc' in text and BLOSC_CNAME in text and str(BLOSC_CLEVEL) in text and BLOSC_SHUFFLE_NAME in text:
                found = True
        if not found:
            raise AssertionError('Published data array does not use Blosc-Zstd clevel=9 bitshuffle')


def _safe_cache_key(e: dict[str, Any]) -> str:
    entry = re.sub('[^0-9A-Za-z_.:-]+', '_', str(e.get('entry_id') or 'fixture'))[:96]
    sha = str(e.get('sha256', ''))[:16]
    return f'{entry}:{sha}'


def _remove_tree(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _publish_directory_atomically(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + '.old')
    _remove_tree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    _remove_tree(old)
