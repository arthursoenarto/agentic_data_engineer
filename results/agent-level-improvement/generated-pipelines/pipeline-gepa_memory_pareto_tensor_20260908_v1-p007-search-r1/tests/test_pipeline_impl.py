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


def _inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'options': {
            'product_type': ['ensemble_mean', 'ensemble_members', 'ensemble_spread', 'reanalysis'],
            'variable': ['geopotential', 'temperature'],
            'year': ['2024'],
            'month': ['01'],
            'day': ['01', '02', '03', '04', '05', '06', '07'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['zip', 'unarchived'],
        },
        'option_metadata': {
            'variable': {
                'temperature': {'units': 'K'},
                'geopotential': {'units': 'm2 s-2'},
            }
        },
    }


def _contract():
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
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-07', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00', '12:00', '18:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'data_format': 'grib', 'dataset_id': 'reanalysis-era5-pressure-levels', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def _sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_fixture(cache: Path):
    times = pd.date_range('2024-01-01T00:00:00', '2024-01-07T18:00:00', freq='6h')
    pressure = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 89.75, 89.5], dtype=np.float32)
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype=np.float32)
    shape = (len(times), len(pressure), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = 250.0 + base * 0.1
    geop = 50000.0 + base * 2.0
    temp[2, 1, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), temp, {'units': 'K', 'long_name': 'temperature'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), geop, {'units': 'm2 s-2', 'long_name': 'geopotential'}),
        },
        coords={
            'time': times.values.astype('datetime64[ns]'),
            'pressure_level': ('pressure_level', pressure, {'units': 'hPa'}),
            'latitude': ('latitude', lat, {'units': 'degrees_north'}),
            'longitude': ('longitude', lon, {'units': 'degrees_east'}),
        },
    )
    cache.mkdir()
    src = cache / 'era5_fixture.nc'
    ds.to_netcdf(src, engine='h5netcdf', encoding={'temperature': {'dtype': 'int16', 'scale_factor': 0.1, 'add_offset': 250.0, '_FillValue': -32767}})
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'era5-test-nc', 'relative_path': 'era5_fixture.nc', 'source': 'offline-test', 'size_bytes': src.stat().st_size, 'sha256': _sha(src)}],
    }
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return ds


def _codec_text(store: Path, var: str):
    return json.dumps(json.loads((store / var / 'zarr.json').read_text())['codecs'], sort_keys=True).lower()


def test_contract_validation_rejects_invalid_time(tmp_path):
    cache = tmp_path / 'cache'
    _write_fixture(cache)
    contract = _contract()
    contract['scope']['time']['selected_times'] = ['05:00']
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({'dataset_contract': contract}, _inventory(), cache, tmp_path / 'out')


def test_fixture_verification_rejects_tamper_before_decode(tmp_path):
    cache = tmp_path / 'cache'
    _write_fixture(cache)
    mf = json.loads((cache / 'source_fixture_manifest.json').read_text())
    mf['entries'][0]['sha256'] = '0' * 64
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(mf), encoding='utf-8')
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / 'out')


def test_publication_semantics_chunks_codecs_and_read_path(tmp_path):
    cache = tmp_path / 'cache'
    source = _write_fixture(cache)
    out = tmp_path / 'out'
    result = pipeline_impl.run_pipeline({'lock': {'selected_contract': _contract()}}, _inventory(), cache, out)

    assert result['cache']['hits'] == 1
    assert result['cache']['misses'] == 0
    assert result['cache']['acquired'] == 0
    assert result['dataset_artifact']['store_path'] == 'era5_pressure_levels.zarr'
    assert len(result['dataset_artifact']['channels']) == 3

    store = out / 'era5_pressure_levels.zarr'
    root_meta = json.loads((store / 'zarr.json').read_text())
    assert 'consolidated_metadata' in root_meta

    ds = xr.open_zarr(store, consolidated=True)
    try:
        channels = result['dataset_artifact']['channels']
        for ch in channels:
            arr = ds[ch['array_path']]
            selector_dim = ch['selector_coordinate_paths']['pressure_level']
            assert selector_dim in arr.dims
            assert arr.sizes['time'] == 28
            assert arr.sizes[selector_dim] == 1
            assert arr.sizes['latitude'] == 3
            assert arr.sizes['longitude'] == 4
            meta = json.loads((store / ch['array_path'] / 'zarr.json').read_text())
            assert tuple(meta['chunk_grid']['configuration']['chunk_shape']) == (1, 1, 3, 4)
            text = _codec_text(store, ch['array_path'])
            mode = ds.attrs['publication_data_chunk_codecs']
            if mode == 'uncompressed':
                assert all(x not in text for x in ['blosc', 'gzip', 'zstd', 'zlib', 'lz4'])
            else:
                assert mode == 'blosc_lz4_clevel1_shuffle_lossless'
                assert 'blosc' in text and 'lz4' in text

        temp500 = ds['temperature__pressure_level_500'].values
        expected_temp500 = source['temperature'].sel(time=source.time.values, pressure_level=[500]).values
        np.testing.assert_array_equal(temp500, expected_temp500)
        temp850 = ds['temperature__pressure_level_850'].values
        expected_temp850 = source['temperature'].sel(time=source.time.values, pressure_level=[850]).values
        np.testing.assert_array_equal(temp850, expected_temp850)
        geop500 = ds['geopotential__pressure_level_500'].values
        expected_geop500 = source['geopotential'].sel(time=source.time.values, pressure_level=[500]).values
        np.testing.assert_array_equal(geop500, expected_geop500)
        np.testing.assert_array_equal(ds['latitude'].values, source['latitude'].values)
        np.testing.assert_array_equal(ds['longitude'].values, source['longitude'].values)

        planes = []
        for ch in channels:
            selector_dim = ch['selector_coordinate_paths']['pressure_level']
            plane = ds[ch['array_path']].isel(time=0, **{selector_dim: 0}).values
            assert plane.shape == (3, 4)
            planes.append(plane)
        chw = np.stack(planes, axis=0)
        assert chw.shape == (3, 3, 4)
    finally:
        ds.close()


def test_exact_rerun_behavior_values_and_artifact(tmp_path):
    cache = tmp_path / 'cache'
    _write_fixture(cache)
    out = tmp_path / 'out'
    r1 = pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    first = xr.open_zarr(out / 'era5_pressure_levels.zarr', consolidated=True).load()
    r2 = pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    second = xr.open_zarr(out / 'era5_pressure_levels.zarr', consolidated=True).load()
    assert r1 == r2
    assert set(first.data_vars) == set(second.data_vars)
    for name in first.data_vars:
        np.testing.assert_array_equal(first[name].values, second[name].values)
    first.close()
    second.close()


def test_missing_fixture_manifest_disables_network_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline_impl.run_pipeline(_contract(), _inventory(), tmp_path / 'empty-cache', tmp_path / 'out')
