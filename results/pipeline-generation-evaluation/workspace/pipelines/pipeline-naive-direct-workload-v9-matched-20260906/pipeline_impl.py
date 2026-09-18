from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PIPELINE_ID = 'pipeline-naive-direct-workload-v9-matched-20260906'
DATASET_ID = 'reanalysis-era5-pressure-levels'
DATASET_SLUG = 'reanalysis_era5_pressure_levels'
STORE_NAME = 'era5_pressure_levels_20240101_20240107.zarr'

FIELD_SPECS = [
    {
        'field_id': 'temperature[pressure_level="500"]',
        'variable': 'temperature',
        'level_hpa': 500.0,
        'array_name': 'temperature_500hPa',
        'selectors': {'pressure_level': '500'},
    },
    {
        'field_id': 'temperature[pressure_level="850"]',
        'variable': 'temperature',
        'level_hpa': 850.0,
        'array_name': 'temperature_850hPa',
        'selectors': {'pressure_level': '850'},
    },
    {
        'field_id': 'geopotential[pressure_level="500"]',
        'variable': 'geopotential',
        'level_hpa': 500.0,
        'array_name': 'geopotential_500hPa',
        'selectors': {'pressure_level': '500'},
    },
]

EXPECTED_TIMES = [
    f'2024-01-{day:02d}T{hour:02d}:00:00'
    for day in range(1, 8)
    for hour in (0, 6, 12, 18)
]

VARIABLE_ALIASES = {
    'temperature': {'temperature', 't', 'air_temperature'},
    'geopotential': {'geopotential', 'z'},
}

PRESSURE_NAMES = {
    'isobaricinhpa', 'isobaricinpa', 'pressure_level', 'pressurelevel',
    'level', 'plev', 'isobaric', 'isobaricinhpa0'
}


def run_pipeline(contract_lock: Any, inventory: Any, cache_dir: str, output_dir: str) -> Dict[str, Any]:
    cache_root = Path(cache_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve_under(cache_root, 'source_fixture_manifest.json')
    manifest = _read_json_file(manifest_path)
    entries = manifest.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('source_fixture_manifest.json must contain a non-empty entries list')

    fixture_paths: List[Path] = []
    reused_keys: List[str] = []
    for idx, entry in enumerate(entries):
        rel = entry.get('relative_path')
        if not isinstance(rel, str) or not rel:
            raise ValueError(f'manifest entry {idx} has invalid relative_path')
        path = _resolve_under(cache_root, rel)
        _verify_fixture_entry(path, entry, idx)
        fixture_paths.append(path)
        reused_keys.append(str(entry.get('entry_id') or rel))

    warnings: List[str] = []
    _validate_inventory_hint(inventory, warnings)
    _validate_contract_hint(contract_lock, warnings)

    work_dir = _resolve_under(output_root, '_decode_work')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    store_path = _resolve_under(output_root, STORE_NAME)
    if store_path.exists():
        shutil.rmtree(store_path)

    try:
        decode_paths = _expand_decode_inputs(fixture_paths, work_dir)
        ds = _build_output_dataset(decode_paths)
        _write_request_sidecar(output_root)
        _write_zarr_v3(ds, store_path, warnings)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)

    channels = [
        {
            'field_id': spec['field_id'],
            'array_path': spec['array_name'],
            'selectors': dict(spec['selectors']),
            'selector_coordinate_paths': {},
        }
        for spec in FIELD_SPECS
    ]

    return {
        'cache': {
            'hits': len(fixture_paths),
            'misses': 0,
            'acquired': 0,
            'reused_keys': reused_keys,
            'acquired_keys': [],
        },
        'dataset_artifact': {
            'schema_version': 'dataset_artifact_layout.v1',
            'storage_format': 'zarr',
            'store_path': STORE_NAME,
            'dimensions': {'sample': 'sample', 'y': 'y', 'x': 'x'},
            'coordinates': {'sample': 'sample', 'y': 'y', 'x': 'x'},
            'channels': channels,
        },
        'warnings': warnings,
    }


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


def _resolve_under(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'path escapes authorized root: {candidate}') from exc
    return candidate


def _verify_fixture_entry(path: Path, entry: Dict[str, Any], idx: int) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f'source fixture entry {idx} not found: {path}')
    expected_size = entry.get('size_bytes')
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError(f'manifest entry {idx} has invalid size_bytes')
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f'manifest entry {idx} size mismatch: expected {expected_size}, got {actual_size}')
    expected_sha = entry.get('sha256')
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or expected_sha.lower() != expected_sha:
        raise ValueError(f'manifest entry {idx} has invalid lowercase sha256')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f'manifest entry {idx} sha256 mismatch')


def _expand_decode_inputs(paths: Sequence[Path], work_dir: Path) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == '.zip':
            with zipfile.ZipFile(path, 'r') as zf:
                for member in sorted(zf.infolist(), key=lambda m: m.filename):
                    if member.is_dir():
                        continue
                    name = Path(member.filename).name
                    if not name:
                        continue
                    if Path(name).suffix.lower() not in {'.grib', '.grb', '.grib2', '.nc', '.nc4', '.netcdf', '.zarr'}:
                        continue
                    target = _resolve_under(work_dir, f'{len(out):04d}_{name}')
                    with zf.open(member, 'r') as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    out.append(target)
        else:
            out.append(path)
    if not out:
        raise ValueError('no decodable GRIB/NetCDF/Zarr source files found in verified fixture entries')
    return out


def _build_output_dataset(paths: Sequence[Path]):
    import numpy as np
    import xarray as xr

    opened = _open_source_datasets(paths)
    try:
        arrays: Dict[str, Any] = {}
        sample_coord = None
        y_coord = None
        x_coord = None

        for spec in FIELD_SPECS:
            pieces = []
            for ds in opened:
                da = _find_matching_data_array(ds, spec['variable'], spec['level_hpa'])
                if da is not None:
                    pieces.append(_normalize_channel(da, spec['level_hpa']))
            if not pieces:
                raise ValueError(f'could not find requested field {spec["field_id"]} in source fixture')
            if len(pieces) == 1:
                channel = pieces[0]
            else:
                channel = xr.concat(pieces, dim='sample')
                channel = channel.sortby('sample')
                _, unique_index = np.unique(channel['sample'].values, return_index=True)
                channel = channel.isel(sample=np.sort(unique_index))

            channel = channel.transpose('sample', 'y', 'x')
            channel = channel.astype('float32')
            _assert_expected_samples(channel)

            if sample_coord is None:
                sample_coord = channel['sample'].values
                y_coord = channel['y'].values
                x_coord = channel['x'].values
            else:
                if not np.array_equal(sample_coord, channel['sample'].values):
                    raise ValueError(f'time/sample coordinate mismatch for {spec["field_id"]}')
                if not np.allclose(y_coord, channel['y'].values, rtol=0.0, atol=1e-7):
                    raise ValueError(f'latitude/y coordinate mismatch for {spec["field_id"]}')
                if not np.allclose(x_coord, channel['x'].values, rtol=0.0, atol=1e-7):
                    raise ValueError(f'longitude/x coordinate mismatch for {spec["field_id"]}')

            arrays[spec['array_name']] = (('sample', 'y', 'x'), np.asarray(channel.values, dtype='float32'))

        ds_out = xr.Dataset(
            data_vars=arrays,
            coords={
                'sample': ('sample', sample_coord),
                'y': ('y', y_coord),
                'x': ('x', x_coord),
            },
            attrs={
                'title': 'ERA5 pressure-level tensors for 1-7 January 2024',
                'provider': 'ECMWF',
                'dataset_id': DATASET_ID,
                'dataset_slug': DATASET_SLUG,
                'source': 'Verified local source fixture; no network acquisition performed by generated pipeline.',
                'publication_format': 'zarr',
                'zarr_format_version': 3,
                'sample_layout': 'Each data variable is float32 with dimensions sample,y,x for warm shuffled per-sample full-field reads.',
            },
        )
        ds_out['sample'].attrs.update({'standard_name': 'time', 'long_name': 'time', 'timezone': 'UTC'})
        ds_out['y'].attrs.update({'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north'})
        ds_out['x'].attrs.update({'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east'})
        for spec in FIELD_SPECS:
            ds_out[spec['array_name']].attrs.update({
                'field_id': spec['field_id'],
                'source_variable': spec['variable'],
                'pressure_level_hPa': int(spec['level_hpa']),
                'layout': 'sample,y,x',
            })
        return ds_out
    finally:
        for ds in opened:
            close = getattr(ds, 'close', None)
            if callable(close):
                close()


def _open_source_datasets(paths: Sequence[Path]) -> List[Any]:
    import xarray as xr

    datasets: List[Any] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == '.zarr' or path.is_dir():
            datasets.append(xr.open_zarr(path, consolidated=False))
        elif suffix in {'.nc', '.nc4', '.netcdf'}:
            datasets.append(xr.open_dataset(path))
        elif suffix in {'.grib', '.grb', '.grib2', ''}:
            try:
                datasets.append(xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''}))
            except Exception:
                try:
                    import cfgrib
                except Exception as exc:
                    raise RuntimeError('GRIB fixture decoding requires cfgrib/eccodes in the runtime environment') from exc
                try:
                    groups = cfgrib.open_datasets(str(path), backend_kwargs={'indexpath': ''})
                except TypeError:
                    groups = cfgrib.open_datasets(str(path), indexpath='')
                datasets.extend(groups)
        else:
            try:
                datasets.append(xr.open_dataset(path))
            except Exception as exc:
                raise ValueError(f'unsupported source fixture format: {path.name}') from exc
    if not datasets:
        raise ValueError('no datasets could be opened from source fixture')
    return datasets


def _find_matching_data_array(ds: Any, requested_variable: str, level_hpa: float) -> Optional[Any]:
    for name in sorted(ds.data_vars):
        da = ds[name]
        if not _matches_variable(name, da, requested_variable):
            continue
        selected = _select_pressure_level_if_present(da, level_hpa, require_match=True)
        if selected is not None:
            return selected
    return None


def _norm_token(value: Any) -> str:
    return str(value).strip().lower().replace(' ', '_').replace('-', '_')


def _matches_variable(name: str, da: Any, requested_variable: str) -> bool:
    aliases = VARIABLE_ALIASES.get(requested_variable, {requested_variable})
    tokens = {_norm_token(name)}
    for key in ['GRIB_shortName', 'shortName', 'cfVarName', 'standard_name', 'long_name', 'GRIB_name']:
        if key in da.attrs:
            tokens.add(_norm_token(da.attrs[key]))
    return bool(tokens.intersection(aliases))


def _select_pressure_level_if_present(da: Any, level_hpa: float, require_match: bool) -> Optional[Any]:
    import numpy as np

    candidates = []
    for name in list(da.coords) + list(da.dims):
        coord = da.coords.get(name) if name in da.coords else None
        lname = _norm_token(name)
        units = _norm_token(getattr(coord, 'attrs', {}).get('units', '')) if coord is not None else ''
        if lname in PRESSURE_NAMES or 'hpa' in units or units == 'pa':
            candidates.append(name)

    for name in candidates:
        coord = da.coords.get(name)
        if coord is None:
            continue
        values = np.asarray(coord.values)
        units = _norm_token(coord.attrs.get('units', ''))
        target = level_hpa * 100.0 if units == 'pa' or 'pa' == units else level_hpa
        if values.ndim == 0:
            val = float(values)
            if abs(val - target) < 1e-4:
                return da
            continue
        diffs = np.abs(values.astype('float64') - target)
        idx = int(diffs.argmin())
        if float(diffs[idx]) < 1e-4:
            if name in da.dims:
                return da.isel({name: idx}).squeeze(drop=True)
            return da

    if require_match:
        return None
    return da


def _normalize_channel(da: Any, level_hpa: float):
    import numpy as np

    da = _select_pressure_level_if_present(da, level_hpa, require_match=False)
    da = da.squeeze(drop=True)

    time_dim = _infer_time_dim(da)
    y_dim = _infer_spatial_dim(da, latitude=True)
    x_dim = _infer_spatial_dim(da, latitude=False)
    if time_dim is None or y_dim is None or x_dim is None:
        raise ValueError(f'could not infer time/latitude/longitude dimensions for {da.name}')

    rename = {time_dim: 'sample', y_dim: 'y', x_dim: 'x'}
    da = da.rename({k: v for k, v in rename.items() if k != v})

    if 'sample' not in da.coords:
        da = da.assign_coords(sample=np.arange(da.sizes['sample']))
    if 'y' not in da.coords:
        da = da.assign_coords(y=np.arange(da.sizes['y'], dtype='float64'))
    if 'x' not in da.coords:
        da = da.assign_coords(x=np.arange(da.sizes['x'], dtype='float64'))

    da = da.transpose('sample', 'y', 'x')
    da = da.sortby('sample')
    da.load()
    return da


def _infer_time_dim(da: Any) -> Optional[str]:
    import numpy as np

    preferred = ['time', 'valid_time', 'forecast_reference_time', 'sample']
    for name in preferred:
        if name in da.dims:
            return name
    for cname, coord in da.coords.items():
        if coord.dims and coord.dims[0] in da.dims:
            if np.issubdtype(coord.dtype, np.datetime64):
                return coord.dims[0]
    for dim in da.dims:
        if dim not in {_infer_spatial_dim(da, True), _infer_spatial_dim(da, False)}:
            return dim
    return None


def _infer_spatial_dim(da: Any, latitude: bool) -> Optional[str]:
    names = ['latitude', 'lat', 'y'] if latitude else ['longitude', 'lon', 'x']
    for name in names:
        if name in da.dims:
            return name
    for cname, coord in da.coords.items():
        lname = _norm_token(cname)
        standard = _norm_token(coord.attrs.get('standard_name', ''))
        units = _norm_token(coord.attrs.get('units', ''))
        is_match = False
        if latitude:
            is_match = lname in {'latitude', 'lat', 'y'} or standard == 'latitude' or 'degrees_north' in units
        else:
            is_match = lname in {'longitude', 'lon', 'x'} or standard == 'longitude' or 'degrees_east' in units
        if is_match and len(coord.dims) == 1 and coord.dims[0] in da.dims:
            return coord.dims[0]
    return None


def _assert_expected_samples(da: Any) -> None:
    import numpy as np

    if da.sizes.get('sample') != 28:
        raise ValueError(f'expected 28 six-hourly samples, found {da.sizes.get("sample")}')
    sample_values = da['sample'].values
    if np.issubdtype(sample_values.dtype, np.datetime64):
        got = [str(v.astype('datetime64[s]')) for v in sample_values]
        if got != EXPECTED_TIMES:
            raise ValueError('source fixture time coordinate does not match 2024-01-01..2024-01-07 at 00/06/12/18 UTC')


def _write_zarr_v3(ds: Any, store_path: Path, warnings: List[str]) -> None:
    encoding: Dict[str, Dict[str, Any]] = {}
    y_len = int(ds.sizes['y'])
    x_len = int(ds.sizes['x'])
    for spec in FIELD_SPECS:
        encoding[spec['array_name']] = {'dtype': 'float32', 'chunks': (1, y_len, x_len)}
    for coord in ['sample', 'y', 'x']:
        encoding[coord] = {'chunks': (int(ds.sizes[coord]),)}

    def clean_store() -> None:
        if store_path.exists():
            shutil.rmtree(store_path)

    try:
        ds.to_zarr(str(store_path), mode='w', consolidated=True, zarr_format=3, encoding=encoding)
        return
    except TypeError:
        clean_store()
    except Exception as first_exc:
        clean_store()
        try:
            ds.to_zarr(str(store_path), mode='w', consolidated=True, zarr_version=3, encoding=encoding)
            return
        except TypeError:
            clean_store()
        except Exception:
            clean_store()
            try:
                ds.to_zarr(str(store_path), mode='w', consolidated=False, zarr_format=3, encoding=encoding)
                import zarr
                if hasattr(zarr, 'consolidate_metadata'):
                    zarr.consolidate_metadata(str(store_path))
                warnings.append('Runtime xarray/zarr stack did not accept consolidated=True directly for Zarr v3; metadata consolidation was attempted after writing.')
                return
            except Exception as final_exc:
                clean_store()
                raise RuntimeError('failed to write required Zarr v3 output') from first_exc if first_exc else final_exc

    try:
        ds.to_zarr(str(store_path), mode='w', consolidated=True, zarr_version=3, encoding=encoding)
        return
    except Exception as exc:
        clean_store()
        raise RuntimeError('failed to write required Zarr v3 output') from exc


def _write_request_sidecar(output_root: Path) -> None:
    request = {
        'dataset_id': DATASET_ID,
        'product_type': ['reanalysis'],
        'variable': ['temperature', 'geopotential'],
        'pressure_level': ['500', '850'],
        'year': ['2024'],
        'month': ['01'],
        'day': ['01', '02', '03', '04', '05', '06', '07'],
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [90, -180, -90, 180],
        'data_format': 'grib',
        'download_format': 'unarchived',
    }
    sidecar = _resolve_under(output_root, 'request_parameters.json')
    with sidecar.open('w', encoding='utf-8') as f:
        json.dump(request, f, indent=2, sort_keys=True)
        f.write('\n')


def _validate_inventory_hint(inventory: Any, warnings: List[str]) -> None:
    inv = _coerce_jsonish(inventory)
    if isinstance(inv, dict):
        dataset_id = inv.get('dataset_id')
        if dataset_id and dataset_id != DATASET_ID:
            warnings.append(f'inventory dataset_id differs from expected {DATASET_ID}; using verified local fixture selection')


def _validate_contract_hint(contract_lock: Any, warnings: List[str]) -> None:
    contract = _coerce_jsonish(contract_lock)
    if isinstance(contract, dict):
        slug = contract.get('dataset_slug') or contract.get('dataset', {}).get('dataset_slug') if isinstance(contract.get('dataset'), dict) else None
        if slug and slug != DATASET_SLUG:
            warnings.append(f'contract dataset_slug differs from expected {DATASET_SLUG}; using verified local fixture selection')


def _coerce_jsonish(value: Any) -> Any:
    if isinstance(value, (str, os.PathLike)):
        p = Path(value)
        if p.exists() and p.is_file():
            try:
                return _read_json_file(p)
            except Exception:
                return value
    return value
