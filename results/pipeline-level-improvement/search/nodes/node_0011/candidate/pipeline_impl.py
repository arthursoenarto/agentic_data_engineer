from __future__ import annotations

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
from zarr.codecs import BloscCodec, BloscShuffle

PIPELINE_ID = 'pipeline-era5-pareto-baseline-r1-20260907'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
DATASET_ID = 'reanalysis-era5-pressure-levels'
ZARR_STORE_NAME = 'dataset.zarr'

FIELD_ALIASES = {
    'temperature': ('temperature', 't'),
    'geopotential': ('geopotential', 'z'),
    'u_component_of_wind': ('u_component_of_wind', 'u'),
    'v_component_of_wind': ('v_component_of_wind', 'v'),
    'relative_humidity': ('relative_humidity', 'r'),
    'specific_humidity': ('specific_humidity', 'q'),
    'vertical_velocity': ('vertical_velocity', 'w'),
    'vorticity': ('vorticity', 'vo'),
    'divergence': ('divergence', 'd'),
}

TIME_NAMES = ('time', 'valid_time', 'timestamp')
LAT_NAMES = ('latitude', 'lat', 'y')
LON_NAMES = ('longitude', 'lon', 'x')
PLEV_NAMES = ('pressure_level', 'level', 'isobaricInhPa', 'plev')


class ContractError(ValueError):
    pass


def run_pipeline(contract_lock: dict[str, Any], inventory: dict[str, Any], cache_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    '''Materialize a validated ERA5 pressure-level regular-grid contract to consolidated Zarr v3.

    The function is intentionally framework-independent.  If cache_dir contains
    source_fixture_manifest.json, every listed raw source object is verified and
    consumed before any provider or credential path.  This adapter does not use
    credentials and deliberately performs no network access when no complete
    fixture is supplied.
    '''
    contract = _extract_contract(contract_lock)
    request = _validate_contract(contract, inventory)

    cache_root = Path(cache_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    fixture = _verify_fixture(cache_root)
    if not fixture:
        raise RuntimeError('No verified source_fixture_manifest.json was found in cache_dir; network/provider acquisition is intentionally disabled for this fixture-authoritative adapter.')

    ds = _open_fixture_dataset([entry['path'] for entry in fixture['entries']])
    ds = _standardize_grid_names(ds)
    ds = _filter_dataset(ds, request)
    public_ds, channels = _build_public_dataset(ds, contract, inventory, request)

    final_store = out_root / ZARR_STORE_NAME
    _publish_zarr_atomically(public_ds, final_store, request, channels)
    _validate_published_zarr(final_store, request, channels)

    return {
        'cache': {
            'hits': len(fixture['entries']),
            'misses': 0,
            'acquired': 0,
            'reused_keys': [entry['entry_id'] for entry in fixture['entries']],
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': ZARR_STORE_NAME,
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
        'warnings': fixture['warnings'],
    }


def _extract_contract(lock: dict[str, Any]) -> dict[str, Any]:
    if isinstance(lock, dict) and lock.get('schema_version') == 'dataset_contract.v1':
        return lock
    preferred_keys = ('contract', 'dataset_contract', 'selected_contract', 'runtime_contract')
    for key in preferred_keys:
        value = lock.get(key) if isinstance(lock, dict) else None
        if isinstance(value, dict) and value.get('schema_version') == 'dataset_contract.v1':
            return value
    found: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get('schema_version') == 'dataset_contract.v1':
                found.append(obj)
                return
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(lock)
    matching = [c for c in found if c.get('dataset_slug') == DATASET_SLUG]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise ContractError('No dataset_contract.v1 for reanalysis_era5_pressure_levels was found in the supplied lock envelope.')
    raise ContractError('Multiple matching contracts were found; the lock envelope must identify one selected contract.')


def _validate_contract(contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get('schema_version') != 'dataset_inventory.v1':
        raise ContractError('inventory.schema_version must be dataset_inventory.v1')
    if contract.get('schema_version') != 'dataset_contract.v1':
        raise ContractError('contract.schema_version must be dataset_contract.v1')
    if contract.get('dataset_slug') != inventory.get('dataset_slug') or contract.get('dataset_slug') != DATASET_SLUG:
        raise ContractError('contract dataset_slug must match the frozen inventory')
    if inventory.get('dataset_id', DATASET_ID) != DATASET_ID:
        raise ContractError('inventory dataset_id does not match the fixed policy')
    if contract.get('human_confirmed') is not True:
        raise ContractError('contract.human_confirmed must be true')

    options = inventory.get('options') or {}
    fields = contract.get('fields') or []
    if not fields:
        raise ContractError('At least one field is required')

    selected_levels: list[str] = []
    field_specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for field in fields:
        name = field.get('name')
        if name not in options.get('variable', []):
            raise ContractError(f'Unsupported variable: {name!r}')
        selectors = field.get('selectors') or []
        plev = None
        for selector in selectors:
            dim = selector.get('dimension')
            if dim != 'pressure_level':
                raise ContractError(f'Unsupported selector dimension for ERA5 pressure-level data: {dim!r}')
            value = str(selector.get('value'))
            if value not in options.get('pressure_level', []):
                raise ContractError(f'Unsupported pressure_level: {value!r}')
            plev = value
        if plev is None:
            raise ContractError(f'Field {name!r} must include a pressure_level selector')
        fid = _canonical_field_id(field)
        if fid in seen:
            raise ContractError(f'Duplicate field-selector request: {fid}')
        seen.add(fid)
        selected_levels.append(plev)
        field_specs.append({'field': field, 'name': name, 'pressure_level': plev, 'field_id': fid})

    scope = contract.get('scope') or {}
    product_type = scope.get('product_type') or (contract.get('advanced_options') or {}).get('product_type') or (inventory.get('defaults') or {}).get('product_type')
    if isinstance(product_type, str):
        product_types = [product_type]
    else:
        product_types = list(product_type or [])
    if not product_types:
        raise ContractError('product_type is required')
    invalid_pt = [pt for pt in product_types if pt not in options.get('product_type', [])]
    if invalid_pt:
        raise ContractError(f'Unsupported product_type values: {invalid_pt!r}')
    if product_types != ['reanalysis']:
        raise ContractError('The fixed publication policy for this adapter supports product_type=[reanalysis] only.')

    adv = contract.get('advanced_options') or {}
    data_format = adv.get('data_format', (inventory.get('defaults') or {}).get('data_format'))
    download_format = adv.get('download_format', (inventory.get('defaults') or {}).get('download_format'))
    if data_format not in options.get('data_format', []):
        raise ContractError(f'Unsupported data_format: {data_format!r}')
    if data_format != 'grib':
        raise ContractError('The fixed acquisition policy requires data_format=grib in the runtime lock.')
    if download_format not in options.get('download_format', []):
        raise ContractError(f'Unsupported download_format: {download_format!r}')

    date_range = (scope.get('date_range') or {})
    start_date = _parse_date(date_range.get('start_date'), 'start_date')
    end_date = _parse_date(date_range.get('end_date'), 'end_date')
    if end_date < start_date:
        raise ContractError('date_range.end_date must be on or after start_date')
    inclusive = bool(date_range.get('inclusive', True))

    time_scope = scope.get('time') or {}
    if time_scope.get('timezone', 'UTC') != 'UTC':
        raise ContractError('Only UTC selected_times are supported')
    selected_times = list(time_scope.get('selected_times') or [])
    if not selected_times:
        raise ContractError('scope.time.selected_times is required')
    invalid_times = [t for t in selected_times if t not in options.get('time', [])]
    if invalid_times:
        raise ContractError(f'Unsupported selected_times: {invalid_times!r}')

    requested_timestamps = _requested_timestamps(start_date, end_date, inclusive, selected_times)
    _validate_temporal_options(requested_timestamps, options)
    _validate_temporal_extent(requested_timestamps, inventory)

    geography = scope.get('geography') or {}
    area = geography.get('cds_area', (inventory.get('defaults') or {}).get('area'))
    if not isinstance(area, list) or len(area) != 4:
        raise ContractError('scope.geography.cds_area must be [north, west, south, east]')
    north, west, south, east = [float(v) for v in area]
    if not (-90 <= south <= north <= 90):
        raise ContractError('Latitude bounds must satisfy -90 <= south <= north <= 90')
    if not (-360 <= west <= 360 and -360 <= east <= 360):
        raise ContractError('Longitude bounds must be within [-360, 360]')

    return {
        'field_specs': field_specs,
        'variables': sorted({spec['name'] for spec in field_specs}),
        'pressure_levels': _unique_preserve_order(selected_levels),
        'timestamps': requested_timestamps,
        'area': [north, west, south, east],
        'product_type': product_types,
        'data_format': data_format,
        'download_format': download_format,
    }


def _parse_date(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ContractError(f'date_range.{name} must be an ISO date string')
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.normalize()


def _requested_timestamps(start: pd.Timestamp, end: pd.Timestamp, inclusive: bool, selected_times: list[str]) -> pd.DatetimeIndex:
    final_day = end if inclusive else end - pd.Timedelta(days=1)
    if final_day < start:
        return pd.DatetimeIndex([], dtype='datetime64[ns]')
    days = pd.date_range(start, final_day, freq='D')
    values: list[pd.Timestamp] = []
    for day in days:
        for time_text in selected_times:
            hour, minute = [int(part) for part in time_text.split(':')]
            values.append(day + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(values).astype('datetime64[ns]')


def _validate_temporal_options(timestamps: pd.DatetimeIndex, options: dict[str, Any]) -> None:
    years = set(options.get('year', []))
    months = set(options.get('month', []))
    days = set(options.get('day', []))
    for ts in timestamps:
        if f'{ts.year:04d}' not in years or f'{ts.month:02d}' not in months or f'{ts.day:02d}' not in days:
            raise ContractError(f'Requested date is outside inventory enumerations: {ts.date()}')


def _validate_temporal_extent(timestamps: pd.DatetimeIndex, inventory: dict[str, Any]) -> None:
    intervals = (((inventory.get('catalogue_metadata') or {}).get('extent') or {}).get('temporal') or {}).get('interval') or []
    if not intervals or not timestamps.size:
        return
    start_raw, end_raw = intervals[0]
    start = pd.Timestamp(start_raw).tz_convert('UTC').tz_localize(None) if pd.Timestamp(start_raw).tzinfo else pd.Timestamp(start_raw)
    end = pd.Timestamp(end_raw).tz_convert('UTC').tz_localize(None) if pd.Timestamp(end_raw).tzinfo else pd.Timestamp(end_raw)
    if timestamps.min() < start or timestamps.max() > end + pd.Timedelta(days=1):
        raise ContractError('Requested timestamps are outside the inventory temporal extent')


def _verify_fixture(cache_root: Path) -> dict[str, Any] | None:
    manifest_path = cache_root / 'source_fixture_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'source_fixture_manifest.v1':
        raise ContractError('source_fixture_manifest.json has an unsupported schema_version')
    entries = manifest.get('entries') or []
    if not entries:
        raise ContractError('source_fixture_manifest.json must contain at least one entry')
    verified: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, entry in enumerate(entries):
        rel = entry.get('relative_path')
        if not isinstance(rel, str) or rel.startswith('/'):
            raise ContractError(f'Fixture entry {i} has an invalid relative_path')
        path = (cache_root / rel).resolve()
        if cache_root.resolve() not in (path, *path.parents):
            raise ContractError(f'Fixture entry {i} escapes cache_dir')
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'Fixture file not found: {rel}')
        expected_size = int(entry.get('size_bytes'))
        actual_size = path.stat().st_size
        if expected_size <= 0 or actual_size != expected_size:
            raise ContractError(f'Fixture size mismatch for {rel}')
        expected_sha = str(entry.get('sha256', '')).lower()
        actual_sha = _sha256(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or actual_sha != expected_sha:
            raise ContractError(f'Fixture SHA-256 mismatch for {rel}')
        suffix = ''.join(path.suffixes).lower()
        if not any(suffix.endswith(ext) for ext in ('.grib', '.grb', '.grib2', '.nc', '.nc4', '.cdf', '.zarr')):
            warnings.append(f'Verified fixture entry {entry.get("entry_id", rel)} has an uncommon extension: {path.name}')
        verified.append({'entry_id': str(entry.get('entry_id') or rel), 'path': path, 'sha256': actual_sha})
    return {'entries': verified, 'warnings': warnings}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _open_fixture_dataset(paths: list[Path]) -> xr.Dataset:
    datasets: list[xr.Dataset] = []
    for path in paths:
        suffixes = ''.join(path.suffixes).lower()
        if suffixes.endswith('.zarr'):
            ds = xr.open_zarr(path, consolidated=False).load()
        elif suffixes.endswith(('.grib', '.grb', '.grib2')):
            ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}, decode_cf=True, mask_and_scale=True).load()
        else:
            ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True).load()
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, combine_attrs='drop_conflicts').load()


def _standardize_grid_names(ds: xr.Dataset) -> xr.Dataset:
    renames: dict[str, str] = {}
    for wanted, candidates in [('time', TIME_NAMES), ('latitude', LAT_NAMES), ('longitude', LON_NAMES), ('pressure_level', PLEV_NAMES)]:
        if wanted in ds.dims or wanted in ds.coords:
            continue
        for name in candidates:
            if name in ds.dims or name in ds.coords:
                renames[name] = wanted
                break
    if renames:
        ds = ds.rename(renames)
    missing = [name for name in ('time', 'latitude', 'longitude') if name not in ds.coords and name not in ds.dims]
    if missing:
        raise ContractError(f'Source fixture is missing required grid coordinates: {missing!r}')
    if 'time' in ds.coords:
        times = pd.to_datetime(ds['time'].values, utc=True).tz_convert(None).to_numpy(dtype='datetime64[ns]')
        ds = ds.assign_coords(time=times)
    return ds


def _filter_dataset(ds: xr.Dataset, request: dict[str, Any]) -> xr.Dataset:
    wanted_times = request['timestamps']
    if not wanted_times.size:
        raise ContractError('Runtime lock selected no timestamps')
    source_times = pd.DatetimeIndex(pd.to_datetime(ds['time'].values).astype('datetime64[ns]'))
    missing_times = wanted_times.difference(source_times)
    if len(missing_times):
        raise ContractError(f'Source fixture does not contain all requested timestamps; first missing timestamp is {missing_times[0]}')
    ds = ds.sel(time=wanted_times.to_numpy(dtype='datetime64[ns]'))

    north, west, south, east = request['area']
    lat_values = np.asarray(ds['latitude'].values, dtype=float)
    lat_idx = np.where((lat_values >= south) & (lat_values <= north))[0]
    if lat_idx.size == 0:
        raise ContractError('Requested latitude area selects no source grid cells')
    lon_values = np.asarray(ds['longitude'].values, dtype=float)
    norm_lon = ((lon_values + 180.0) % 360.0) - 180.0
    norm_west = ((west + 180.0) % 360.0) - 180.0
    norm_east = ((east + 180.0) % 360.0) - 180.0
    if abs(east - west) >= 360 or (west == -180 and east == 180):
        lon_mask = np.ones_like(norm_lon, dtype=bool)
    elif norm_west <= norm_east:
        lon_mask = (norm_lon >= norm_west) & (norm_lon <= norm_east)
    else:
        lon_mask = (norm_lon >= norm_west) | (norm_lon <= norm_east)
    lon_idx = np.where(lon_mask)[0]
    if lon_idx.size == 0:
        raise ContractError('Requested longitude area selects no source grid cells')
    return ds.isel(latitude=lat_idx, longitude=lon_idx)


def _build_public_dataset(ds: xr.Dataset, contract: dict[str, Any], inventory: dict[str, Any], request: dict[str, Any]) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    data_vars: dict[str, xr.DataArray] = {}
    channels: list[dict[str, Any]] = []
    for spec in request['field_specs']:
        source_name = _find_source_variable(ds, spec['name'])
        da = ds[source_name]
        plev_dim = _find_pressure_level_name(da, ds)
        unique_plev_dim = _safe_name(f'pressure_level__{spec["field_id"]}')
        if plev_dim:
            idx = _pressure_level_index(ds[plev_dim].values if plev_dim in ds.coords else da[plev_dim].values, spec['pressure_level'])
            da = da.isel({plev_dim: [idx]})
            da = da.rename({plev_dim: unique_plev_dim})
            da = da.assign_coords({unique_plev_dim: np.asarray([_native_level_value(da[unique_plev_dim].values[0], spec['pressure_level'])])})
            da[unique_plev_dim].attrs.update({'long_name': 'pressure level', 'units': 'hPa', 'selector_dimension': 'pressure_level'})
        else:
            da = da.expand_dims({unique_plev_dim: np.asarray([int(spec['pressure_level'])])})
            da[unique_plev_dim].attrs.update({'long_name': 'pressure level', 'units': 'hPa', 'selector_dimension': 'pressure_level'})
        order = [dim for dim in ('time', unique_plev_dim, 'latitude', 'longitude') if dim in da.dims]
        da = da.transpose(*order, ...)
        out_name = _safe_name(spec['field_id'])
        da = da.copy(deep=False)
        da.attrs = dict(da.attrs)
        da.attrs['field_id'] = spec['field_id']
        da.attrs['source_variable_name'] = source_name
        units = (((inventory.get('option_metadata') or {}).get('variable') or {}).get(spec['name']) or {}).get('units')
        if units and not da.attrs.get('units'):
            da.attrs['units'] = units
        da.encoding = {}
        data_vars[out_name] = da
        channels.append({
            'field_id': spec['field_id'],
            'array_path': out_name,
            'selectors': {'pressure_level': spec['pressure_level']},
            'selector_coordinate_paths': {'pressure_level': unique_plev_dim},
        })
    public = xr.Dataset(data_vars=data_vars, attrs={
        'dataset_slug': contract.get('dataset_slug'),
        'dataset_id': DATASET_ID,
        'pipeline_id': PIPELINE_ID,
        'publication_format': 'zarr',
        'zarr_format_version': 3,
        'source_provider': 'ECMWF',
    })
    for coord in public.coords:
        public[coord].encoding = {}
    return public, channels


def _find_source_variable(ds: xr.Dataset, requested_name: str) -> str:
    candidates = FIELD_ALIASES.get(requested_name, (requested_name,))
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {name.lower(): name for name in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ContractError(f'Source fixture does not contain requested variable {requested_name!r}; available variables are {list(ds.data_vars)!r}')


def _find_pressure_level_name(da: xr.DataArray, ds: xr.Dataset) -> str | None:
    for name in PLEV_NAMES:
        if name in da.dims or name in da.coords:
            return name
    if 'pressure_level' in ds.coords and 'pressure_level' in da.dims:
        return 'pressure_level'
    return None


def _pressure_level_index(values: Any, requested: str) -> int:
    arr = np.asarray(values)
    as_text = [str(v.item() if hasattr(v, 'item') else v) for v in arr]
    if requested in as_text:
        return as_text.index(requested)
    try:
        req_float = float(requested)
        diffs = np.abs(arr.astype(float) - req_float)
        idx = int(np.argmin(diffs))
        if float(diffs[idx]) < 1e-9:
            return idx
    except Exception:
        pass
    raise ContractError(f'Source fixture does not contain pressure_level={requested!r}')


def _native_level_value(value: Any, fallback: str) -> Any:
    try:
        return value.item()
    except Exception:
        try:
            return int(fallback)
        except Exception:
            return fallback


def _publish_zarr_atomically(ds: xr.Dataset, final_store: Path, request: dict[str, Any], channels: list[dict[str, Any]]) -> None:
    tmp_parent = final_store.parent
    tmp_path = Path(tempfile.mkdtemp(prefix=f'.{final_store.name}.tmp-', dir=tmp_parent))
    backup_path = final_store.with_name(f'.{final_store.name}.bak-{os.getpid()}')
    try:
        encoding: dict[str, dict[str, Any]] = {}
        data_compressors = (BloscCodec(cname='lz4', clevel=5, shuffle=BloscShuffle.noshuffle, typesize=4),)
        for name, da in ds.data_vars.items():
            encoding[name] = {
                'chunks': tuple(1 if dim == 'time' or dim.startswith('pressure_level__') else int(da.sizes[dim]) for dim in da.dims),
                'compressors': data_compressors,
            }
        for name in ds.coords:
            encoding[name] = {}
        ds.to_zarr(tmp_path, mode='w', consolidated=True, zarr_format=3, encoding=encoding)
        _validate_published_zarr(tmp_path, request, channels)
        if backup_path.exists():
            shutil.rmtree(backup_path)
        if final_store.exists():
            final_store.rename(backup_path)
        tmp_path.rename(final_store)
        if backup_path.exists():
            shutil.rmtree(backup_path)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        if backup_path.exists() and not final_store.exists():
            backup_path.rename(final_store)
        raise


def _validate_published_zarr(store: Path, request: dict[str, Any], channels: list[dict[str, Any]]) -> None:
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        for coord in ('time', 'latitude', 'longitude'):
            if coord not in reopened.coords:
                raise RuntimeError(f'Published Zarr is missing coordinate {coord}')
        published_times = pd.DatetimeIndex(pd.to_datetime(reopened['time'].values).astype('datetime64[ns]'))
        if not published_times.equals(request['timestamps']):
            raise RuntimeError('Published Zarr time coordinate does not match requested timestamps exactly')
        for channel in channels:
            array_path = channel['array_path']
            if array_path not in reopened.data_vars:
                raise RuntimeError(f'Published Zarr is missing channel array {array_path}')
            selector_coord = channel['selector_coordinate_paths']['pressure_level']
            if selector_coord not in reopened[array_path].dims:
                raise RuntimeError(f'Channel {array_path} does not preserve pressure_level as an output dimension')
            if reopened.sizes[selector_coord] != 1:
                raise RuntimeError(f'Channel {array_path} pressure_level selector dimension must have cardinality one')
    finally:
        reopened.close()


def _canonical_field_id(field: dict[str, Any]) -> str:
    selectors = field.get('selectors') or []
    if not selectors:
        return str(field.get('name'))
    parts = [f'{sel.get("dimension")}={json.dumps(str(sel.get("value")), ensure_ascii=False)}' for sel in selectors]
    return f'{field.get("name")}[{",".join(parts)}]'


def _safe_name(value: str) -> str:
    out = re.sub(r'[^0-9A-Za-z_]+', '_', value).strip('_')
    if not out or out[0].isdigit():
        out = f'v_{out}'
    return out


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
