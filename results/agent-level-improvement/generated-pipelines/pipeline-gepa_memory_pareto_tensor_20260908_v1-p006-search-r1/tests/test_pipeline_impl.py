from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", ROOT / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_impl)


def _inventory() -> dict:
    variables = ["temperature", "geopotential", "u_component_of_wind"]
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "title": "ERA5 hourly data on pressure levels from 1940 to present",
        "provider": "ECMWF",
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": variables,
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "option_units": {"pressure_level": "hPa", "time": "UTC", "area": "north/west/south/east degrees"},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Temperature."},
                "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>", "description": "Geopotential."},
                "u_component_of_wind": {"label": "U-component of wind", "units": "m s<sup>-1</sup>"},
            }
        },
    }


def _contract(*, end_date="2024-01-02", fields=None, times=None) -> dict:
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test contract",
        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "access_methods": [],
        "credential_requirements": [],
        "fields": fields
        or [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": end_date, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": times or ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _write_fixture(cache: Path) -> xr.Dataset:
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00", "2024-01-01T18:00", "2024-01-02T00:00", "2024-01-02T06:00", "2024-01-02T12:00", "2024-01-02T18:00"], dtype="datetime64[ns]")
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float64)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float64)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    t = 250.0 + base / 100.0
    z = 50000.0 + base * 2.0
    t[1, 0, 2, 3] = np.nan
    z[2, 1, 3, 1] = np.nan
    ds = xr.Dataset(
        {
            "t": (("time", "pressure_level", "latitude", "longitude"), t, {"units": "K", "long_name": "Temperature"}),
            "z": (("time", "pressure_level", "latitude", "longitude"), z, {"units": "m**2 s**-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    path = cache / "era5_fixture.nc"
    ds.to_netcdf(path, engine="scipy")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "fixture-1", "relative_path": path.name, "source": "offline-test", "size_bytes": path.stat().st_size, "sha256": digest}],
    }
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return xr.open_dataset(path, decode_cf=True, mask_and_scale=True)


def test_contract_validation_rejects_invalid_selector(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache).close()
    bad = _contract(fields=[{"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "700", "unit": "hPa", "label": None}], "units": None, "description": None}])
    with pytest.raises(ValueError, match="selector value"):
        pipeline_impl.run_pipeline({"contract": bad}, _inventory(), str(cache), str(out))


def test_fixture_verification_prevents_corrupt_source(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache).close()
    fixture = cache / "era5_fixture.nc"
    fixture.write_bytes(fixture.read_bytes() + b"x")
    with pytest.raises(ValueError, match="verification"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(out))


def test_publication_semantics_layout_codec_and_rerun(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    source = _write_fixture(cache)
    result1 = pipeline_impl.run_pipeline({"selected_contract": _contract()}, _inventory(), str(cache), str(out))
    result2 = pipeline_impl.run_pipeline({"selected_contract": _contract()}, _inventory(), str(cache), str(out))
    assert result1 == result2
    assert result1["cache"]["hits"] == 1
    assert result1["cache"]["misses"] == 0
    assert result1["dataset_artifact"]["storage_format"] == "zarr"
    assert result1["dataset_artifact"]["store_path"] == "dataset.zarr"

    store = out / "dataset.zarr"
    root_meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    ds = xr.open_zarr(store, consolidated=True)
    try:
        channels = result1["dataset_artifact"]["channels"]
        assert [c["field_id"] for c in channels] == [
            'temperature[pressure_level="500"]',
            'temperature[pressure_level="850"]',
            'geopotential[pressure_level="500"]',
        ]
        assert set(c["array_path"] for c in channels) == set(ds.data_vars)
        assert np.array_equal(ds["time"].values, source["time"].values)
        assert np.array_equal(ds["latitude"].values, source["latitude"].values)
        assert np.array_equal(ds["longitude"].values, source["longitude"].values)

        for channel in channels:
            arr = ds[channel["array_path"]]
            selector_coord = channel["selector_coordinate_paths"]["pressure_level"]
            assert selector_coord in ds.coords
            assert selector_coord in arr.dims
            assert arr.sizes[selector_coord] == 1
            assert arr.dims == ("time", selector_coord, "latitude", "longitude")
            assert arr.chunks[0] == (1,) * arr.sizes["time"]
            assert arr.chunks[1] == (1,)
            assert arr.chunks[2] == (arr.sizes["latitude"],)
            assert arr.chunks[3] == (arr.sizes["longitude"],)
            level = int(ds[selector_coord].values[0])
            src_name = "t" if channel["field_id"].startswith("temperature") else "z"
            expected = source[src_name].sel(pressure_level=[level]).values
            np.testing.assert_array_equal(arr.values, expected)
            assert arr.attrs["field_id"] == channel["field_id"]
            assert "units" in arr.attrs

        # Lightweight full C,H,W-equivalent read across all declared channels.
        sample_planes = []
        for channel in channels:
            arr = ds[channel["array_path"]]
            sample_planes.append(arr.isel(time=0).values)
        tensor = np.concatenate(sample_planes, axis=0)
        assert tensor.shape == (3, source.sizes["latitude"], source.sizes["longitude"])

        # Inspect Zarr v3 array metadata for explicit footprint-oriented lossless codec.
        for channel in channels:
            meta = json.loads((store / channel["array_path"] / "zarr.json").read_text(encoding="utf-8"))
            assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 1, source.sizes["latitude"], source.sizes["longitude"]]
            codec_text = json.dumps(meta.get("codecs", [])).lower()
            assert "zstd" in codec_text
            assert "blosc" in codec_text
            assert "7" in codec_text
    finally:
        ds.close()
        source.close()


def test_inclusive_date_only_end_keeps_complete_final_day(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache).close()
    result = pipeline_impl.run_pipeline(_contract(end_date="2024-01-02", times=["18:00"]), _inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / result["dataset_artifact"]["store_path"], consolidated=True)
    try:
        assert ds.sizes["time"] == 2
        assert str(ds["time"].values[-1]).startswith("2024-01-02T18:00")
    finally:
        ds.close()


def test_supports_other_valid_lock_variable_subset(tmp_path: Path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache).close()
    fields = [{"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None}]
    result = pipeline_impl.run_pipeline({"lock": {"dataset_contract": _contract(fields=fields, times=["00:00"])}}, _inventory(), str(cache), str(out))
    assert len(result["dataset_artifact"]["channels"]) == 1
    ch = result["dataset_artifact"]["channels"][0]
    assert ch["field_id"] == 'geopotential[pressure_level="850"]'
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    try:
        assert set(ds.data_vars) == {ch["array_path"]}
        assert ds.sizes["time"] == 2
    finally:
        ds.close()
