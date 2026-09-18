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
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'options': {
            'product_type': ['reanalysis'],
            'variable': ['temperature', 'geopotential'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['unarchived', 'zip'],
            'year': ['2024'],
            'month': ['01'],
            'day': ['01', '02'],
        },
        'option_metadata': {
            'variable': {
                'temperature': {'units': 'K', 'description': 'air temperature'},
                'geopotential': {'units': 'm**2 s**-2', 'description': 'geopotential'},
            }
        },
    }


def _contract():
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'fields': [
            {'name': 'temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa'}]},
            {'name': 'temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa'}]},
            {'name': 'geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa'}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-02', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'dataset_id': 'reanalysis-era5-pressure-levels', 'data_format': 'netcdf', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def _make_fixture(cache_dir: Path) -> xr.Dataset:
    times = pd.to_datetime(['2024-01-01T00:00:00', '2024-01-01T06:00:00', '2024-01-02T00:00:00', '2024-01-02T06:00:00'])
    levels = np.array([500, 850], dtype='int32')
    lat = np.array([1.0, 0.0, -1.0], dtype='float32')
    lon = np.array([10.0, 11.0, 12.0, 13.0], dtype='float32')
    base = np.arange(len(times) * len(levels) * len(lat) * len(lon), dtype='float32').reshape(len(times), len(levels), len(lat), len(lon))
    t = 250.0 + base * 0.5
    z = 1000.0 + base * 2.0
    t[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        data_vars={
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), t, {'units': 'K'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), z, {'units': 'm**2 s**-2'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
    )
    path = cache_dir / 'source.nc'
    ds.to_netcdf(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'local-era5-test', 'relative_path': 'source.nc', 'source': 'offline-test', 'size_bytes': path.stat().st_size, 'sha256': digest}],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return ds


def _run(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    out.mkdir()
    source = _make_fixture(cache)
    result = pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(cache), str(out))
    return source, result, out


def test_contract_validation_rejects_bad_selector(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    _make_fixture(cache)
    contract = _contract()
    contract['fields'][0]['selectors'][0]['value'] = '700'
    with pytest.raises(ValueError, match='unsupported pressure_level'):
        pipeline_impl.run_pipeline(contract, _inventory(), str(cache), str(tmp_path / 'out'))


def test_fixture_verification_and_publication_readback(tmp_path):
    source, result, out = _run(tmp_path)
    assert result['cache']['hits'] == 1
    assert result['cache']['misses'] == 0
    assert result['cache']['acquired'] == 0
    artifact = result['dataset_artifact']
    assert artifact['store_path'] == 'era5_pressure_levels.zarr'
    assert artifact['dimensions'] == {'sample': 'time', 'y': 'latitude', 'x': 'longitude'}
    assert [c['field_id'] for c in artifact['channels']] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    store = out / artifact['store_path']
    root_meta = json.loads((store / 'zarr.json').read_text())
    assert root_meta['zarr_format'] == 3
    assert 'consolidated_metadata' in root_meta
    reopened = xr.open_zarr(store, consolidated=True)
    try:
        assert set(reopened.data_vars) == {'temperature__pressure_level_500', 'temperature__pressure_level_850', 'geopotential__pressure_level_500'}
        assert np.array_equal(reopened['time'].values, source['time'].values)
        assert np.array_equal(reopened['latitude'].values, source['latitude'].values)
        assert np.array_equal(reopened['longitude'].values, source['longitude'].values)
        assert reopened['temperature__pressure_level_500'].dims == ('time', 'temperature__pressure_level_500__pressure_level', 'latitude', 'longitude')
        assert reopened['temperature__pressure_level_500'].shape == (4, 1, 3, 4)
        np.testing.assert_array_equal(reopened['temperature__pressure_level_500'].values[:, 0], source['temperature'].sel(pressure_level=500).values)
        np.testing.assert_array_equal(reopened['temperature__pressure_level_850'].values[:, 0], source['temperature'].sel(pressure_level=850).values)
        np.testing.assert_array_equal(reopened['geopotential__pressure_level_500'].values[:, 0], source['geopotential'].sel(pressure_level=500).values)
        assert np.isnan(reopened['temperature__pressure_level_500'].values[1, 0, 1, 2])
        assert reopened['temperature__pressure_level_500'].attrs['units'] == 'K'
    finally:
        reopened.close()


def test_chunks_are_full_field_sample_aligned_and_uncompressed(tmp_path):
    _source, result, out = _run(tmp_path)
    store = out / result['dataset_artifact']['store_path']
    forbidden = ('blosc', 'zstd', 'zlib', 'gzip', 'lz4', 'quant', 'delta', 'scale_factor', 'add_offset', 'shuffle')
    for channel in result['dataset_artifact']['channels']:
        meta = json.loads((store / channel['array_path'] / 'zarr.json').read_text())
        assert meta['chunk_grid']['configuration']['chunk_shape'] == [1, 1, 3, 4]
        text = json.dumps(meta).lower()
        assert not any(token in text for token in forbidden)
        assert [c['name'] for c in meta.get('codecs', [])] == ['bytes']


def test_rerun_is_exact_and_replaces_output_deterministically(tmp_path):
    _source, result1, out = _run(tmp_path)
    store = out / result1['dataset_artifact']['store_path']
    first_files = sorted(p.relative_to(store).as_posix() for p in store.rglob('*') if p.is_file())
    first_meta = (store / 'zarr.json').read_bytes()
    result2 = pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(tmp_path / 'cache'), str(out))
    second_files = sorted(p.relative_to(store).as_posix() for p in store.rglob('*') if p.is_file())
    assert result1 == result2
    assert first_files == second_files
    assert first_meta == (store / 'zarr.json').read_bytes()


def test_offline_full_chw_read_path_across_declared_channels(tmp_path):
    source, result, out = _run(tmp_path)
    ds = xr.open_zarr(out / result['dataset_artifact']['store_path'], consolidated=True)
    try:
        planes = []
        for channel in result['dataset_artifact']['channels']:
            arr = ds[channel['array_path']]
            selector_dim = next(d for d in arr.dims if d.endswith('__pressure_level'))
            plane = arr.isel(time=0, **{selector_dim: 0}).values
            assert plane.shape == (3, 4)
            planes.append(plane)
        chw = np.stack(planes, axis=0)
        assert chw.shape == (3, 3, 4)
        expected = np.stack([
            source['temperature'].sel(pressure_level=500).isel(time=0).values,
            source['temperature'].sel(pressure_level=850).isel(time=0).values,
            source['geopotential'].sel(pressure_level=500).isel(time=0).values,
        ])
        np.testing.assert_array_equal(chw, expected)
    finally:
        ds.close()


def test_no_network_fallback_without_fixture(tmp_path):
    with pytest.raises(RuntimeError, match='source_fixture_manifest'):
        pipeline_impl.run_pipeline(_contract(), _inventory(), str(tmp_path / 'empty-cache'), str(tmp_path / 'out'))
