import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline_impl


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'generated_at': '2026-07-06T14:00:28.158844+00:00',
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'options': {
            'product_type': ['ensemble_mean', 'ensemble_members', 'ensemble_spread', 'reanalysis'],
            'variable': ['temperature', 'geopotential', 'relative_humidity'],
            'year': ['2023', '2024', '2025'],
            'month': ['01', '02'],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['zip', 'unarchived'],
        },
        'option_metadata': {'variable': {'temperature': {'units': 'K'}, 'geopotential': {'units': 'm2 s-2'}}},
    }


def _contract(start='2024-01-01', end='2024-01-01', times=None, area=None, fields=None):
    if times is None:
        times = ['00:00', '06:00']
    if area is None:
        area = [1, 0, 0, 1]
    if fields is None:
        fields = [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ]
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'title': 'test contract',
        'fields': fields,
        'scope': {
            'date_range': {'start_date': start, 'end_date': end, 'inclusive': True},
            'geography': {'area': 'custom', 'cds_area': area, 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': times, 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'dataset_id': 'reanalysis-era5-pressure-levels', 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def _make_fixture(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range('2024-01-01T00:00:00', '2024-01-01T23:00:00', freq='6h')
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    shape = (len(times), len(levels), len(lat), len(lon))
    temp = 250.0 + np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10.0
    geop = 5000.0 + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    temp[1, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), temp, {'units': 'K'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), geop, {'units': 'm2 s-2'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
    )
    path = cache_dir / 'era5_fixture.nc'
    ds.to_netcdf(path, encoding={'temperature': {'dtype': 'int16', 'scale_factor': 0.1, 'add_offset': 250.0, '_FillValue': -32767}})
    size = path.stat().st_size
    sha = _sha256(path)
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'local-era5-nc', 'relative_path': path.name, 'source': 'unit-test', 'size_bytes': size, 'sha256': sha}],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return path


def _hash_tree(path: Path):
    h = hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            h.update(b'\0')
            h.update(p.read_bytes())
    return h.hexdigest()


def test_contract_validation_rejects_bad_selector(tmp_path):
    cache = tmp_path / 'cache'
    _make_fixture(cache)
    bad = _contract(fields=[{'name': 'temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '700'}]}])
    with pytest.raises(pipeline_impl.ContractError):
        pipeline_impl.run_pipeline({'contract': bad}, _inventory(), cache, tmp_path / 'out')


def test_fixture_manifest_verified_before_readback(tmp_path):
    cache = tmp_path / 'cache'
    fixture = _make_fixture(cache)
    manifest = json.loads((cache / 'source_fixture_manifest.json').read_text())
    manifest['entries'][0]['sha256'] = '0' * 64
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(pipeline_impl.FixtureError, match='sha256'):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / 'out')
    assert fixture.exists()


def test_filtering_publication_and_readback_preserve_dimensions_and_missingness(tmp_path):
    cache = tmp_path / 'cache'
    _make_fixture(cache)
    out = tmp_path / 'out'
    result = pipeline_impl.run_pipeline({'locks': [{'kind': 'other'}, {'contract': _contract()}]}, _inventory(), cache, out)
    assert result['cache']['hits'] == 1
    assert result['cache']['acquired'] == 0
    assert result['dataset_artifact']['storage_format'] == 'zarr'
    store = out / result['dataset_artifact']['store_path']
    assert (store / 'zarr.json').exists()
    meta = json.loads((store / 'zarr.json').read_text())
    assert meta['zarr_format'] == 3
    ds = xr.open_zarr(store, consolidated=True)
    try:
        assert set(ds.dims) >= {'time', 'pressure_level', 'latitude', 'longitude'}
        assert ds.sizes['time'] == 2
        assert ds.sizes['pressure_level'] == 1
        assert list(pd.to_datetime(ds.time.values).strftime('%H:%M')) == ['00:00', '06:00']
        assert list(ds.latitude.values) == [1.0, 0.0]
        assert list(ds.longitude.values) == [0.0, 1.0]
        assert np.isnan(ds['temperature'].isel(time=1, pressure_level=0, latitude=0, longitude=0).item())
        assert ds['temperature'].encoding.get('scale_factor') is None
    finally:
        ds.close()


def test_inclusive_date_only_end_includes_final_day_selected_times(tmp_path):
    cache = tmp_path / 'cache'
    _make_fixture(cache)
    contract = _contract(start='2024-01-01', end='2024-01-01', times=['18:00'])
    pipeline_impl.run_pipeline(contract, _inventory(), cache, tmp_path / 'out')
    ds = xr.open_zarr(tmp_path / 'out' / 'dataset.zarr', consolidated=True)
    try:
        assert list(pd.to_datetime(ds.time.values).strftime('%Y-%m-%dT%H:%M')) == ['2024-01-01T18:00']
    finally:
        ds.close()


def test_exact_rerun_replaces_publication_deterministically(tmp_path):
    cache = tmp_path / 'cache'
    _make_fixture(cache)
    out = tmp_path / 'out'
    contract = _contract()
    r1 = pipeline_impl.run_pipeline(contract, _inventory(), cache, out)
    h1 = _hash_tree(out / 'dataset.zarr')
    r2 = pipeline_impl.run_pipeline(contract, _inventory(), cache, out)
    h2 = _hash_tree(out / 'dataset.zarr')
    assert h1 == h2
    assert r1 == r2


def test_missing_complete_fixture_fails_without_network(tmp_path):
    with pytest.raises(pipeline_impl.FixtureError, match='source_fixture_manifest'):
        pipeline_impl.run_pipeline(_contract(), _inventory(), tmp_path / 'empty-cache', tmp_path / 'out')
