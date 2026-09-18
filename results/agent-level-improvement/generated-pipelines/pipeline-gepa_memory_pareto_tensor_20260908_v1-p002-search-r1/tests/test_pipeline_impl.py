from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import pipeline_impl


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob('*') if q.is_file()):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(b'\0')
        h.update(_sha(p).encode())
        h.update(b'\0')
    return h.hexdigest()


def _inventory() -> dict:
    return {
        'schema_version': 'dataset_inventory.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'dataset_id': 'reanalysis-era5-pressure-levels',
        'options': {
            'product_type': ['reanalysis'],
            'variable': ['temperature', 'geopotential', 'relative_humidity'],
            'year': ['2024'],
            'month': ['01'],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': [f'{i:02d}:00' for i in range(24)],
            'pressure_level': ['500', '850'],
            'data_format': ['grib', 'netcdf'],
            'download_format': ['unarchived', 'zip'],
        },
        'defaults': {'area': [90, -180, -90, 180], 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'option_metadata': {'variable': {'temperature': {'units': 'K'}, 'geopotential': {'units': 'm2 s-2'}}},
    }


def _contract(times=None, fields=None) -> dict:
    return {
        'schema_version': 'dataset_contract.v1',
        'dataset_slug': 'reanalysis_era5_pressure_levels',
        'fields': fields or [
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
            {'name': 'temperature', 'display_name': 'Temperature', 'selectors': [{'dimension': 'pressure_level', 'value': '850', 'unit': 'hPa', 'label': None}]},
            {'name': 'geopotential', 'display_name': 'Geopotential', 'selectors': [{'dimension': 'pressure_level', 'value': '500', 'unit': 'hPa', 'label': None}]},
        ],
        'scope': {
            'date_range': {'start_date': '2024-01-01', 'end_date': '2024-01-02', 'inclusive': True},
            'geography': {'area': 'global', 'cds_area': [90, -180, -90, 180], 'cds_area_order': ['north', 'west', 'south', 'east']},
            'product_type': 'reanalysis',
            'time': {'selected_times': times or ['00:00', '06:00', '12:00', '18:00'], 'timestep': '6 hours', 'timezone': 'UTC'},
        },
        'advanced_options': {'dataset_id': 'reanalysis-era5-pressure-levels', 'data_format': 'grib', 'download_format': 'unarchived', 'product_type': ['reanalysis']},
        'human_confirmed': True,
    }


def _make_fixture(cache: Path) -> xr.Dataset:
    times = pd.date_range('2024-01-01', '2024-01-02 18:00', freq='6h')
    pressure = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 89.75, 89.5], dtype=np.float32)
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype=np.float32)
    shape = (len(times), len(pressure), len(lat), len(lon))
    vals = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    vals[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            'temperature': (('time', 'pressure_level', 'latitude', 'longitude'), vals + np.float32(250.0), {'units': 'K', 'quality_flag_meaning': 'synthetic fixture'}),
            'geopotential': (('time', 'pressure_level', 'latitude', 'longitude'), vals + np.float32(50000.0), {'units': 'm2 s-2'}),
        },
        coords={
            'time': times.to_numpy(dtype='datetime64[ns]'),
            'pressure_level': ('pressure_level', pressure, {'units': 'hPa'}),
            'latitude': ('latitude', lat, {'units': 'degrees_north'}),
            'longitude': ('longitude', lon, {'units': 'degrees_east'}),
        },
        attrs={'source': 'unit-test fixture'},
    )
    nc = cache / 'era5_fixture.nc'
    enc = {
        'temperature': {'dtype': 'int16', 'scale_factor': 0.5, '_FillValue': -32768},
        'geopotential': {'dtype': 'int16', 'scale_factor': 2.0, '_FillValue': -32768},
    }
    ds.to_netcdf(nc, engine='h5netcdf', encoding=enc)
    manifest = {
        'schema_version': 'source_fixture_manifest.v1',
        'entries': [{'entry_id': 'era5-small', 'relative_path': nc.name, 'source': 'offline-test', 'size_bytes': nc.stat().st_size, 'sha256': _sha(nc)}],
    }
    (cache / 'source_fixture_manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    return xr.open_dataset(nc, decode_cf=True, mask_and_scale=True).load()


def test_fixture_reuse_filtering_publication_readback_chunks_and_compression(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    source = _make_fixture(cache)
    result = pipeline_impl.run_pipeline({'envelope': {'selected_contract': _contract()}}, _inventory(), str(cache), str(out))

    assert result['cache']['hits'] == 1
    assert result['cache']['misses'] == 0
    assert result['cache']['acquired'] == 0
    assert result['dataset_artifact']['store_path'] == 'dataset.zarr'
    assert len(result['dataset_artifact']['channels']) == 3

    store = out / 'dataset.zarr'
    root_meta = json.loads((store / 'zarr.json').read_text())
    assert root_meta['zarr_format'] == 3
    assert 'consolidated_metadata' in root_meta

    got = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    assert set(got.coords) >= {'time', 'latitude', 'longitude'}
    assert len(got.data_vars) == 3

    for ch in result['dataset_artifact']['channels']:
        arr = got[ch['array_path']]
        assert ch['field_id'] in arr.attrs['field_id']
        assert ch['selector_coordinate_paths']['pressure_level'] in got.coords
        pcoord = ch['selector_coordinate_paths']['pressure_level']
        assert pcoord in arr.dims
        assert got.sizes[pcoord] == 1
        assert arr.dims == ('time', pcoord, 'latitude', 'longitude')
        z = zarr.open_group(str(store), mode='r')[ch['array_path']]
        assert tuple(z.chunks) == (1, 1, got.sizes['latitude'], got.sizes['longitude'])
        meta = json.dumps(getattr(z, 'metadata', {}), default=str).lower()
        assert 'blosc' in meta and 'zstd' in meta and '7' in meta

    first = result['dataset_artifact']['channels'][0]
    expected = source['temperature'].sel(time=got.time.values, pressure_level=[500]).values
    np.testing.assert_array_equal(got[first['array_path']].values, expected)
    assert np.isnan(got[first['array_path']].values).any()
    assert got[first['array_path']].attrs['units'] == 'K'


def test_exact_rerun_behavior(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    _make_fixture(cache)
    pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(cache), str(out))
    digest1 = _dir_digest(out / 'dataset.zarr')
    pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(cache), str(out))
    digest2 = _dir_digest(out / 'dataset.zarr')
    assert digest1 == digest2


def test_contract_validation_rejects_invalid_time_before_open(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    _make_fixture(cache)
    bad = _contract(times=['25:00'])
    with pytest.raises(ValueError, match='invalid selected time'):
        pipeline_impl.run_pipeline({'contract': bad}, _inventory(), str(cache), str(out))


def test_manifest_verification_is_mandatory_and_secret_free(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    with pytest.raises(FileNotFoundError, match='source_fixture_manifest'):
        pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(cache), str(out))


def test_manifest_hash_mismatch_fails_before_decoding(tmp_path: Path):
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    cache.mkdir()
    _make_fixture(cache)
    manifest_path = cache / 'source_fixture_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['entries'][0]['sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='sha256 mismatch'):
        pipeline_impl.run_pipeline({'contract': _contract()}, _inventory(), str(cache), str(out))
