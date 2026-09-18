from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline_impl


def inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'options': {
            'variable': ['temperature', 'geopotential'],
            'pressure_level': ['500', '850'],
            'year': ['2024'],
            'month': ['01'],
            'day': ['01', '02', '03', '04', '05', '06', '07'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
        },
        'defaults': {'area': [90, -180, -90, 180]},
        'option_metadata': {
            'variable': {
                'temperature': {'label': 'Temperature', 'units': 'K'},
                'geopotential': {'label': 'Geopotential', 'units': 'm<sup>2</sup> s<sup>-2</sup>'},
            }
        },
    }


def contract():
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'fields': [
            {'name': 'temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-07', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00', '12:00', '18:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {
            'dataset_id': 'reanalysis-era5-pressure-levels',
            'data_format': 'grib',
            'download_format': 'unarchived',
            'product_type': ['reanalysis'],
        },
        'human_confirmed': True,
    }


def lock(c=None):
    return {
        'selected_contract': c or contract(),
        'fixed_pipeline_policy': {
            'provider': 'ECMWF',
            'dataset_id': 'reanalysis-era5-pressure-levels',
            'acquisition_format': 'grib',
            'publication_format': 'zarr',
            'zarr': {'format_version': 3, 'consolidated_metadata': True},
        },
    }


def make_fixture(cache_dir: Path):
    times = pd.date_range('2024-01-01T00:00:00', '2024-01-07T18:00:00', freq='6h')
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 0.0, -90.0], dtype=np.float32)
    lon = np.array([-180.0, 0.0, 180.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    t = base.copy()
    z = base + 10000.0
    t[2, 0, 1, 1] = np.nan
    ds = xr.Dataset(
        {
            't': (('time', 'pressure_level', 'latitude', 'longitude'), t, {'units': 'K', 'long_name': 'Temperature'}),
            'z': (('time', 'pressure_level', 'latitude', 'longitude'), z, {'units': 'm2 s-2', 'long_name': 'Geopotential'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
        attrs={'source': 'local fixture'},
    )
    path = cache_dir / 'era5_fixture.nc'
    ds.to_netcdf(path, engine='h5netcdf')
    raw = path.read_bytes()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [
            {
                'entry_id': 'fixture-era5',
                'relative_path': 'era5_fixture.nc',
                'source': 'local-test-fixture',
                'size_bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    (cache_dir / 'source_fixture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return ds


def test_publication_semantics_chunks_uncompressed_and_rerun(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    source = make_fixture(cache)
    result1 = pipeline_impl.run_pipeline(lock(), inventory(), str(cache), str(out))
    result2 = pipeline_impl.run_pipeline(lock(), inventory(), str(cache), str(out))
    assert result1 == result2
    assert result1['cache'] == {'hits': 1, 'misses': 0, 'acquired': 0, 'reused_keys': ['fixture-era5'], 'acquired_keys': []}
    artifact = result1['dataset_artifact']
    assert artifact['store_path'] == 'dataset.zarr'
    assert artifact['dimensions'] == {'sample': 'time', 'y': 'latitude', 'x': 'longitude'}
    assert len(artifact['channels']) == 3

    store = out / 'dataset.zarr'
    root_meta = json.loads((store / 'zarr.json').read_text(encoding='utf-8'))
    assert root_meta['zarr_format'] == 3
    assert 'consolidated_metadata' in root_meta
    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    group = zarr.open_group(store, mode='r')
    expected_times = pd.date_range('2024-01-01T00:00:00', '2024-01-07T18:00:00', freq='6h')
    assert np.array_equal(reopened['time'].values, expected_times.values)
    assert np.array_equal(reopened['latitude'].values, source['latitude'].values)
    assert np.array_equal(reopened['longitude'].values, source['longitude'].values)

    for ch in artifact['channels']:
        name = ch['array_path']
        arr = group[name]
        assert tuple(arr.chunks) == (1, 1, 3, 3)
        codec_text = ' '.join([c.__class__.__name__.lower() for c in arr.metadata.codecs])
        assert 'blosc' not in codec_text
        assert 'zstd' not in codec_text
        assert 'lz4' not in codec_text
        assert 'gzip' not in codec_text
        assert 'zlib' not in codec_text
        sel = ch['selector_coordinate_paths']['pressure_level']
        assert sel in reopened.coords
        assert reopened[name].dims == ('time', sel, 'latitude', 'longitude')
        assert reopened.sizes[sel] == 1
        level = int(ch['selectors']['pressure_level'])
        src_name = 't' if ch['field_id'].startswith('temperature') else 'z'
        expected = source[src_name].sel(time=expected_times, pressure_level=[level]).values
        assert np.array_equal(reopened[name].values, expected, equal_nan=True)


def test_inclusive_date_only_end_keeps_final_day(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    make_fixture(cache)
    result = pipeline_impl.run_pipeline(lock(), inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / result['dataset_artifact']['store_path'], consolidated=True, zarr_format=3)
    assert str(pd.Timestamp(ds['time'].values[-1])) == '2024-01-07 18:00:00'
    assert ds.sizes['time'] == 28


def test_invalid_selector_rejected_before_publication(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    make_fixture(cache)
    c = contract()
    c['fields'][0]['selectors'][0]['value'] = '925'
    with pytest.raises(pipeline_impl.PipelineError, match='unsupported pressure_level'):
        pipeline_impl.run_pipeline(lock(c), inventory(), str(cache), str(tmp_path / 'out'))


def test_fixture_manifest_verification_and_no_network_required(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    cache.mkdir()
    make_fixture(cache)
    manifest_path = cache / 'source_fixture_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['entries'][0]['sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(pipeline_impl.PipelineError, match='sha256 mismatch'):
        pipeline_impl.run_pipeline(lock(), inventory(), str(cache), str(tmp_path / 'out'))


def test_no_fixture_fails_safely_without_acquisition(tmp_path):
    with pytest.raises(pipeline_impl.PipelineError, match='local source fixture is required'):
        pipeline_impl.run_pipeline(lock(), inventory(), str(tmp_path / 'empty'), str(tmp_path / 'out'))


def test_subregion_filtering_preserves_coordinates_and_values(tmp_path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    source = make_fixture(cache)
    c = contract()
    c['scope']['geography']['cds_area'] = [90, -1, -1, 1]
    result = pipeline_impl.run_pipeline(lock(c), inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / result['dataset_artifact']['store_path'], consolidated=True, zarr_format=3).load()
    assert np.array_equal(ds['latitude'].values, np.array([90.0, 0.0], dtype=np.float32))
    assert np.array_equal(ds['longitude'].values, np.array([0.0], dtype=np.float32))
    ch = result['dataset_artifact']['channels'][0]
    level = int(ch['selectors']['pressure_level'])
    expected = source['t'].sel(time=ds['time'].values, pressure_level=[level], latitude=[90.0, 0.0], longitude=[0.0]).values
    assert np.array_equal(ds[ch['array_path']].values, expected, equal_nan=True)
