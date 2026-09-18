import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pipeline_impl


def inventory():
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'options': {
            'variable': ['temperature', 'geopotential'],
            'pressure_level': ['500', '850'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'year': ['2024'],
        },
        'option_metadata': {
            'variable': {
                'temperature': {'units': 'K'},
                'geopotential': {'units': 'm2 s-2'},
            }
        },
    }


def contract():
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'source_url': 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download',
        'fields': [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-02', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': ['00:00', '06:00', '12:00', '18:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'data_format': 'grib', 'dataset_id': 'reanalysis-era5-pressure-levels', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def policy():
    return {
        'provider': 'ECMWF',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'acquisition_format': 'grib',
        'publication_format': 'zarr',
        'data_model': 'xarray_dataset',
        'zarr': {'schema_version': 'regular_grid_zarr_output_policy.v1', 'dataset_class': 'regular_rectilinear_grid', 'format_version': 3, 'consolidated_metadata': True},
    }


def make_fixture(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    times = pd.date_range('2024-01-01T00:00', '2024-01-02T18:00', freq='6h')
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 89.75], dtype=np.float32)
    lon = np.array([-180.0, -179.75, -179.5], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    t = base + np.float32(273.15)
    z = base * np.float32(10.0)
    t[1, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        data_vars={
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), t, {'units': 'K', 'quality_flag': 'decoded'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), z, {'units': 'm2 s-2'}),
        },
        coords={'time': times, 'pressure_level': levels, 'latitude': lat, 'longitude': lon},
    )
    src = cache / 'era5_fixture.nc'
    ds.to_netcdf(src, engine='h5netcdf')
    blob = src.read_bytes()
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'era5-local-fixture', 'relative_path': 'era5_fixture.nc', 'source': 'local-test', 'size_bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}],
    }
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(manifest))
    return cache, ds


def run(tmp_path, c=None):
    cache, source = make_fixture(tmp_path)
    out = tmp_path / 'out'
    lock = {'contract': c or contract(), 'fixed_pipeline_policy': policy()}
    result = pipeline_impl.run_pipeline(lock, inventory(), str(cache), str(out))
    return result, out, source


def test_offline_fixture_reuse_publication_and_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv('CDSAPI_KEY', raising=False)
    result, out, source = run(tmp_path)
    assert result['cache']['hits'] == 1
    assert result['cache']['acquired'] == 0
    assert result['dataset_artifact']['store_path'] == 'data.zarr'
    assert (out / 'data.zarr').exists()
    reopened = xr.open_zarr(out / 'data.zarr', consolidated=True)
    assert set(reopened.data_vars) == {'temperature[pressure_level="500"]', 'temperature[pressure_level="850"]', 'geopotential[pressure_level="500"]'}
    assert reopened.sizes['time'] == 8
    assert '2024-01-02T18:00:00' in str(reopened.time.values[-1])
    for ch in result['dataset_artifact']['channels']:
        sdim = ch['selector_coordinate_paths']['pressure_level']
        assert sdim in reopened.dims
        assert reopened.sizes[sdim] == 1
        assert reopened[ch['array_path']].dims == ('time', sdim, 'latitude', 'longitude')
    np.testing.assert_array_equal(reopened['temperature[pressure_level="500"]'].values[:, 0], source['temperature'].sel(pressure_level=500).values)
    assert np.isnan(reopened['temperature[pressure_level="500"]'].values[1, 0, 0, 0])


def test_invalid_selector_rejected(tmp_path):
    c = contract()
    c['fields'][0]['selectors'][0]['value'] = '700'
    cache, _ = make_fixture(tmp_path)
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({'contract': c, 'fixed_pipeline_policy': policy()}, inventory(), str(cache), str(tmp_path / 'out'))


def test_manifest_verification_rejects_tamper(tmp_path):
    cache, _ = make_fixture(tmp_path)
    (cache / 'era5_fixture.nc').write_bytes(b'corrupt')
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({'contract': contract(), 'fixed_pipeline_policy': policy()}, inventory(), str(cache), str(tmp_path / 'out'))


def test_no_manifest_no_network_fails_safely(tmp_path):
    cache = tmp_path / 'empty_cache'
    cache.mkdir()
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({'contract': contract(), 'fixed_pipeline_policy': policy()}, inventory(), str(cache), str(tmp_path / 'out'))


def test_rerun_equivalence_and_codec_layout(tmp_path):
    result1, out, _ = run(tmp_path)
    result2 = pipeline_impl.run_pipeline({'contract': contract(), 'fixed_pipeline_policy': policy()}, inventory(), str(tmp_path / 'cache'), str(out))
    assert result1 == result2
    import zarr
    root = zarr.open_group(out / 'data.zarr', mode='r')
    for ch in result2['dataset_artifact']['channels']:
        arr = root[ch['array_path']]
        assert tuple(arr.chunks) == (1, 1, 2, 3)
        meta = arr.metadata
        text = json.dumps([c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in meta.codecs], sort_keys=True, default=str).lower()
        assert 'blosc' in text
        assert 'zstd' in text
        assert '9' in text
        assert 'bitshuffle' in text


def test_subarea_filtering_preserves_native_coordinates(tmp_path):
    c = contract()
    c['scope']['geography']['area'] = 'subset'
    c['scope']['geography']['cds_area'] = [90, -180, 89.75, -179.75]
    result, out, source = run(tmp_path, c)
    reopened = xr.open_zarr(out / 'data.zarr', consolidated=True)
    np.testing.assert_array_equal(reopened.latitude.values, source.latitude.values[:2])
    np.testing.assert_array_equal(reopened.longitude.values, source.longitude.values[:2])
