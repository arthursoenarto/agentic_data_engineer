import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import pipeline_impl


def inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'provider': 'ECMWF',
        'options': {
            'product_type': ['reanalysis'],
            'variable': ['temperature', 'geopotential'],
            'year': ['2024'],
            'month': ['01'],
            'day': [f'{i:02d}' for i in range(1, 8)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['zip', 'unarchived'],
        },
        'option_units': {'pressure_level': 'hPa'},
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
    }


def contract(days=2):
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'source_url': 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download',
        'provider': 'Copernicus Climate Data Store',
        'fields': [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': f'2024-01-{days:02d}', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00', '12:00', '18:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'data_format': 'grib', 'dataset_id': 'reanalysis-era5-pressure-levels', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def write_fixture(cache_dir: Path, scalar_level=False):
    times = np.array([np.datetime64(f'2024-01-{d:02d}T{h:02d}:00:00') for d in range(1, 8) for h in (0, 6, 12, 18)], dtype='datetime64[ns]')
    lat = np.linspace(90, -90, 9, dtype='float32')
    lon = np.linspace(-180, 179, 16, dtype='float32')
    levels = np.array([500, 850], dtype='int32')
    base = np.arange(times.size * levels.size * lat.size * lon.size, dtype='float32').reshape(times.size, levels.size, lat.size, lon.size)
    ds = xr.Dataset(
        {
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), base),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), base + 100000),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
    )
    if scalar_level:
        ds = ds.sel(pressure_level=500)
    path = cache_dir / 'source.nc'
    ds.to_netcdf(path, engine='netcdf4')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'fixture-source', 'relative_path': 'source.nc', 'source': 'test', 'size_bytes': path.stat().st_size, 'sha256': digest}],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest))
    return path


def run(tmp_path, c=None, scalar_level=False):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    out.mkdir()
    write_fixture(cache, scalar_level=scalar_level)
    result = pipeline_impl.run_pipeline({'schema_version': 'lock.v1', 'contract': c or contract()}, inventory(), cache, out)
    return result, out


def test_publication_mapping_and_readback(tmp_path):
    result, out = run(tmp_path)
    art = result['dataset_artifact']
    assert art['store_path'] == 'dataset.zarr'
    assert art['dimensions'] == {'sample': 'time', 'y': 'latitude', 'x': 'longitude'}
    store = out / art['store_path']
    ds = xr.open_zarr(store, consolidated=True)
    try:
        assert ds.sizes['time'] == 8
        assert set(art['coordinates'].values()) <= set(ds.coords)
        for ch in art['channels']:
            assert ch['array_path'] in ds.data_vars
            assert Path(store, ch['array_path']).exists()
            arr = ds[ch['array_path']]
            assert arr.dtype == np.dtype('float32')
            assert arr.dims[0] == 'time'
            assert arr.dims[-2:] == ('latitude', 'longitude')
            assert np.isfinite(arr.isel(time=0).values).all()
    finally:
        ds.close()


def test_singleton_selector_dimensions_are_reconstructed(tmp_path):
    c = contract(days=1)
    c['fields'] = [c['fields'][0], c['fields'][2]]
    result, out = run(tmp_path, c=c, scalar_level=True)
    ds = xr.open_zarr(out / result['dataset_artifact']['store_path'], consolidated=True)
    try:
        for ch in result['dataset_artifact']['channels']:
            coord = ch['selector_coordinate_paths']['pressure_level']
            assert coord in ds.coords
            assert ds.sizes[coord] == 1
            assert int(ds[coord].values[0]) == 500
            assert coord in ds[ch['array_path']].dims
    finally:
        ds.close()


def test_bad_selector_is_rejected_before_publication(tmp_path):
    c = contract(days=1)
    c['fields'][0]['selectors'][0]['value'] = '700'
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    out.mkdir()
    write_fixture(cache)
    with pytest.raises(pipeline_impl.PipelineError):
        pipeline_impl.run_pipeline({'contract': c}, inventory(), cache, out)
    assert not (out / 'dataset.zarr').exists()


def test_fixture_integrity_is_checked(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    out.mkdir()
    write_fixture(cache)
    manifest = json.loads((cache / 'source_fixture_manifest.json').read_text())
    manifest['entries'][0]['sha256'] = '0' * 64
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(pipeline_impl.PipelineError, match='sha256'):
        pipeline_impl.run_pipeline({'contract': contract(days=1)}, inventory(), cache, out)


def test_exact_rerun_replaces_output_and_reuses_fixture(tmp_path):
    result1, out = run(tmp_path)
    ds1 = xr.open_zarr(out / 'dataset.zarr', consolidated=True)
    v1 = float(ds1['temperature__pressure_level_500'].isel(time=0, latitude=0, longitude=0).values[0])
    ds1.close()
    cache = tmp_path / 'cache'
    result2 = pipeline_impl.run_pipeline({'contract': contract()}, inventory(), cache, out)
    ds2 = xr.open_zarr(out / 'dataset.zarr', consolidated=True)
    try:
        v2 = float(ds2['temperature__pressure_level_500'].isel(time=0, latitude=0, longitude=0).values[0])
        assert v1 == v2
        assert result1['cache']['reused_keys'] == result2['cache']['reused_keys']
        assert result2['cache']['acquired'] == 0
    finally:
        ds2.close()


def test_chunk_geometry_matches_workload(tmp_path):
    result, out = run(tmp_path)
    group = zarr.open_group(out / result['dataset_artifact']['store_path'], mode='r')
    for ch in result['dataset_artifact']['channels']:
        arr = group[ch['array_path']]
        assert arr.chunks[0] == 1
        assert arr.chunks[1] == 1
        assert arr.chunks[-1] == arr.shape[-1]
        assert arr.chunks[-2] >= 1


def test_date_only_end_includes_complete_final_day(tmp_path):
    result, out = run(tmp_path, c=contract(days=7))
    ds = xr.open_zarr(out / result['dataset_artifact']['store_path'], consolidated=True)
    try:
        assert ds.sizes['time'] == 28
        assert str(ds['time'].values[-1]).startswith('2024-01-07T18:00:00')
    finally:
        ds.close()
