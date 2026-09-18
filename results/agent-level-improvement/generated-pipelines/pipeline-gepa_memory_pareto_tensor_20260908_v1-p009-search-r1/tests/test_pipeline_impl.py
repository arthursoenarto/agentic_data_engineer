import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import pipeline_impl


def seed_contract():
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'fields': [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-02', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'data_format': 'grib', 'dataset_id': 'reanalysis-era5-pressure-levels', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'provider': 'ECMWF',
        'options': {
            'product_type': ['ensemble_mean', 'ensemble_members', 'ensemble_spread', 'reanalysis'],
            'variable': ['temperature', 'geopotential', 'relative_humidity'],
            'year': ['2024'],
            'month': ['01'],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['zip', 'unarchived'],
        },
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'option_metadata': {'variable': {'temperature': {'units': 'K'}, 'geopotential': {'units': 'm2 s-2'}}},
    }


def write_fixture(cache_dir: Path):
    times = np.array(['2024-01-01T00:00:00', '2024-01-01T06:00:00', '2024-01-02T00:00:00', '2024-01-02T06:00:00'], dtype='datetime64[ns]')
    levels = np.array([500, 850], dtype='int32')
    lat = np.array([90.0, 89.75, 89.5], dtype='float32')
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype='float32')
    shape = (len(times), len(levels), len(lat), len(lon))
    t = np.arange(np.prod(shape), dtype='float32').reshape(shape)
    z = (1000 + np.arange(np.prod(shape), dtype='float32')).reshape(shape)
    t[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), t, {'units': 'K'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), z, {'units': 'm2 s-2'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
        attrs={'source_fixture': 'unit-test'},
    )
    path = cache_dir / 'era5_fixture.nc'
    ds.to_netcdf(path)
    data = path.read_bytes()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'fixture-1', 'relative_path': 'era5_fixture.nc', 'source': 'unit-test', 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest))
    return ds


def test_contract_validation_rejects_invalid_selector(tmp_path):
    c = seed_contract()
    c['fields'][0]['selectors'][0]['value'] = '700'
    with pytest.raises(ValueError):
        pipeline_impl.run_pipeline(c, inventory(), str(tmp_path / 'cache'), str(tmp_path / 'out'))


def test_no_network_fallback_without_fixture(tmp_path):
    (tmp_path / 'cache').mkdir()
    with pytest.raises(RuntimeError, match='network'):
        pipeline_impl.run_pipeline(seed_contract(), inventory(), str(tmp_path / 'cache'), str(tmp_path / 'out'))


def test_fixture_reuse_publication_readback_chunks_and_no_compressor(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    source = write_fixture(cache)
    result = pipeline_impl.run_pipeline({'contract': seed_contract()}, inventory(), str(cache), str(out))

    assert result['cache']['hits'] == 1
    assert result['cache']['misses'] == 0
    assert result['cache']['acquired'] == 0
    assert result['dataset_artifact']['store_path'] == 'dataset.zarr'
    assert [c['field_id'] for c in result['dataset_artifact']['channels']] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert all(c['selector_coordinate_paths'] == {'pressure_level': 'pressure_level'} for c in result['dataset_artifact']['channels'])

    store = out / 'dataset.zarr'
    assert json.loads((store / 'zarr.json').read_text())['zarr_format'] == 3
    assert 'consolidated_metadata' in json.loads((store / 'zarr.json').read_text())
    reopened = xr.open_zarr(store, consolidated=True).load()

    np.testing.assert_array_equal(reopened['time'].values, source['time'].values)
    np.testing.assert_array_equal(reopened['latitude'].values, source['latitude'].values)
    np.testing.assert_array_equal(reopened['longitude'].values, source['longitude'].values)
    np.testing.assert_array_equal(reopened['pressure_level'].values, source['pressure_level'].values)
    assert reopened['temperature'].dims == ('time', 'pressure_level', 'latitude', 'longitude')
    assert reopened['geopotential'].dims == ('time', 'pressure_level', 'latitude', 'longitude')
    np.testing.assert_array_equal(reopened['temperature'].values, source['temperature'].values)
    np.testing.assert_array_equal(reopened['geopotential'].values[:, :1], source['geopotential'].sel(pressure_level=[500]).values)
    assert np.isnan(reopened['temperature'].values[1, 0, 1, 2])
    assert reopened['temperature'].attrs['units'] == 'K'

    for name in ['temperature', 'geopotential']:
        arr = zarr.open_array(str(store / name), mode='r')
        assert tuple(arr.chunks) == (1, 1, 3, 4)
        codec_text = json.dumps([repr(c).lower() for c in getattr(arr.metadata, 'codecs', [])])
        assert not any(x in codec_text for x in ['blosc', 'zstd', 'gzip', 'lz4', 'zlib', 'compressor'])


def test_exact_rerun_behavior(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    write_fixture(cache)
    r1 = pipeline_impl.run_pipeline(seed_contract(), inventory(), str(cache), str(out))
    first_files = sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file())
    first_hashes = {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in first_files}
    r2 = pipeline_impl.run_pipeline(seed_contract(), inventory(), str(cache), str(out))
    second_files = sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file())
    second_hashes = {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in second_files}
    assert r1 == r2
    assert first_files == second_files
    assert first_hashes == second_hashes


def test_filtering_preserves_singleton_pressure_dimension(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    write_fixture(cache)
    c = seed_contract()
    c['fields'] = [c['fields'][2]]
    c['scope']['date_range']['end_date'] = '2024-01-01'
    result = pipeline_impl.run_pipeline(c, inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / 'dataset.zarr', consolidated=True)
    assert ds['geopotential'].dims == ('time', 'pressure_level', 'latitude', 'longitude')
    assert ds.sizes['pressure_level'] == 1
    assert result['dataset_artifact']['channels'][0]['selectors'] == {'pressure_level': '500'}
