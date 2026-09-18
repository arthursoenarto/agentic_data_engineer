from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pipeline_impl


def inventory() -> dict:
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'options': {
            'product_type': ['ensemble_mean', 'ensemble_members', 'ensemble_spread', 'reanalysis'],
            'variable': ['temperature', 'geopotential'],
            'year': ['2024'],
            'month': ['01'],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['zip', 'unarchived'],
        },
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'catalogue_metadata': {'extent': {'temporal': {'interval': [['1940-01-01T00:00:00+00:00', '2026-06-30T00:00:00+00:00']]}}},
        'option_metadata': {'variable': {'temperature': {'units': 'K'}, 'geopotential': {'units': 'm2 s-2'}}},
    }


def contract(days: int = 2, times: list[str] | None = None, area: list[float] | None = None) -> dict:
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'title': 'test',
        'fields': [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': f'2024-01-{days:02d}', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': area or [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': times or ['00:00', '06:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'data_format': 'grib', 'dataset_id': 'reanalysis-era5-pressure-levels', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def write_fixture(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(['2024-01-01T00:00:00', '2024-01-01T06:00:00', '2024-01-02T00:00:00', '2024-01-02T06:00:00'])
    levels = np.array([500, 850], dtype='int32')
    lats = np.array([1.0, 0.0, -1.0], dtype='float64')
    lons = np.array([0.0, 1.0, 359.0], dtype='float64')
    shape = (len(times), len(levels), len(lats), len(lons))
    t = (250.0 + np.arange(np.prod(shape), dtype='float64').reshape(shape) / 10.0)
    z = (5000.0 + np.arange(np.prod(shape), dtype='float64').reshape(shape))
    t[0, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        data_vars={
            't': (('time', 'pressure_level', 'latitude', 'longitude'), t, {'units': 'K'}),
            'z': (('time', 'pressure_level', 'latitude', 'longitude'), z, {'units': 'm2 s-2'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lats, 'longitude': lons},
    )
    path = cache_dir / 'era5_fixture.nc'
    ds.to_netcdf(path, engine='h5netcdf', encoding={'t': {'dtype': 'int16', 'scale_factor': 0.1, 'add_offset': 250.0, '_FillValue': -32768}})
    data = path.read_bytes()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'local-era5-test-object', 'relative_path': path.name, 'source': 'unit-test', 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return path


def digest_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        h.update(str(file.relative_to(path)).encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def test_contract_validation_rejects_invalid_selector(tmp_path: Path):
    c = contract()
    c['fields'][0]['selectors'][0]['value'] = '925'
    write_fixture(tmp_path / 'cache')
    with pytest.raises(pipeline_impl.ContractError):
        pipeline_impl.run_pipeline({'contract': c}, inventory(), tmp_path / 'cache', tmp_path / 'out')


def test_fixture_reuse_filtering_publication_and_readback(tmp_path: Path):
    cache = tmp_path / 'cache'
    write_fixture(cache)
    out = tmp_path / 'out'
    result = pipeline_impl.run_pipeline({'runtime': {'contract': contract()}}, inventory(), cache, out)
    assert result['cache']['hits'] == 1
    assert result['cache']['misses'] == 0
    assert result['dataset_artifact']['store_path'] == 'dataset.zarr'
    assert len(result['dataset_artifact']['channels']) == 3

    ds = xr.open_zarr(out / 'dataset.zarr', consolidated=True, zarr_format=3)
    assert ds.sizes['time'] == 4
    assert list(pd.to_datetime(ds['time'].values).strftime('%Y-%m-%dT%H:%M:%S'))[-1] == '2024-01-02T06:00:00'
    for channel in result['dataset_artifact']['channels']:
        arr = ds[channel['array_path']]
        selector_dim = channel['selector_coordinate_paths']['pressure_level']
        assert selector_dim in arr.dims
        assert arr.sizes[selector_dim] == 1
    assert np.isnan(ds['temperature_pressure_level_500'].isel(time=0, latitude=0, longitude=0).values).item()
    ds.close()


def test_spatial_filter_preserves_native_longitudes(tmp_path: Path):
    cache = tmp_path / 'cache'
    write_fixture(cache)
    out = tmp_path / 'out'
    c = contract(area=[1, -2, -1, 0.5])
    pipeline_impl.run_pipeline(c, inventory(), cache, out)
    ds = xr.open_zarr(out / 'dataset.zarr', consolidated=True, zarr_format=3)
    assert set(ds['longitude'].values.tolist()) == {0.0, 359.0}
    ds.close()


def test_exact_rerun_reuses_fixture_and_replaces_output(tmp_path: Path):
    cache = tmp_path / 'cache'
    write_fixture(cache)
    out = tmp_path / 'out'
    c = contract()
    r1 = pipeline_impl.run_pipeline(c, inventory(), cache, out)
    d1 = digest_tree(out / 'dataset.zarr')
    r2 = pipeline_impl.run_pipeline(c, inventory(), cache, out)
    d2 = digest_tree(out / 'dataset.zarr')
    assert r1['cache'] == r2['cache']
    assert d1 == d2


def test_manifest_is_verified_before_decoding(tmp_path: Path):
    cache = tmp_path / 'cache'
    fixture = write_fixture(cache)
    manifest_path = cache / 'source_fixture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['entries'][0]['sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(pipeline_impl.ContractError, match='SHA-256'):
        pipeline_impl.run_pipeline(contract(), inventory(), cache, tmp_path / 'out')
    assert fixture.exists()


def test_no_fixture_does_not_attempt_network(tmp_path: Path):
    with pytest.raises(RuntimeError, match='network/provider acquisition is intentionally disabled'):
        pipeline_impl.run_pipeline(contract(), inventory(), tmp_path / 'empty-cache', tmp_path / 'out')
