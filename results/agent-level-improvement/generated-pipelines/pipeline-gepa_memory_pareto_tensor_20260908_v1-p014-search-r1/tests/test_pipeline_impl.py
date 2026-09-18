import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", ROOT / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "title": "ERA5 hourly data on pressure levels from 1940 to present",
        "provider": "ECMWF",
        "options": {
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "pressure_level": ["500", "850", "700"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "year": ["2024"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "defaults": {"area": [90, -180, -90, 180]},
        "option_metadata": {"variable": {"temperature": {"units": "K"}, "geopotential": {"units": "m2 s-2"}}},
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "fields": [
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa"}]},
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa"}]},
            {"name": "geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa"}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-02", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timezone": "UTC", "timestep": "6 hours"},
        },
        "advanced_options": {"dataset_id": "reanalysis-era5-pressure-levels", "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _lock(contract=None):
    return {
        "dataset_contract": contract or _contract(),
        "fixed_pipeline_policy": {
            "provider": "ECMWF",
            "dataset_id": "reanalysis-era5-pressure-levels",
            "acquisition_format": "grib",
            "publication_format": "zarr",
            "zarr": {"format_version": 3, "consolidated_metadata": True},
        },
    }


def _write_fixture(cache_dir: Path):
    times = np.array([
        "2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00", "2024-01-01T18:00",
        "2024-01-02T00:00", "2024-01-02T06:00", "2024-01-02T12:00", "2024-01-02T18:00",
    ], dtype="datetime64[ns]")
    levels = np.array([500, 700, 850], dtype=np.int32)
    lat = np.array([90.0, 0.0, -90.0], dtype=np.float32)
    lon = np.array([-180.0, 0.0, 180.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = base + 273.15
    geop = base * 10.0
    temp[1, 0, 1, 1] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    raw = cache_dir / "fixture.nc"
    ds.to_netcdf(raw, engine="h5netcdf")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "fixture-nc", "relative_path": "fixture.nc", "source": "offline-test", "size_bytes": raw.stat().st_size, "sha256": digest}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    return ds


def test_grouped_layout_publication_rerun_and_readback(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    src = _write_fixture(cache)
    monkeypatch.delenv("CDSAPI_KEY", raising=False)

    r1 = pipeline_impl.run_pipeline(_lock(), _inventory(), cache, out)
    r2 = pipeline_impl.run_pipeline(_lock(), _inventory(), cache, out)
    assert r1 == r2
    assert r1["cache"] == {"hits": 1, "misses": 0, "acquired": 0, "reused_keys": ["fixture-nc"], "acquired_keys": []}

    art = r1["dataset_artifact"]
    assert art["store_path"] == "dataset.zarr"
    assert art["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert [c["field_id"] for c in art["channels"]] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert art["channels"][0]["array_path"] == "temperature"
    assert art["channels"][1]["array_path"] == "temperature"
    assert art["channels"][0]["selector_coordinate_paths"] == {"pressure_level": "pressure_level__temperature"}

    store = out / "dataset.zarr"
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    zg = zarr.open_group(store, mode="r")
    assert set(["temperature", "geopotential", "pressure_level__temperature", "pressure_level__geopotential", "time", "latitude", "longitude"]).issubset(set(zg.array_keys()))
    assert tuple(zg["temperature"].chunks) == (1, 1, 3, 3)
    assert tuple(zg["geopotential"].chunks) == (1, 1, 3, 3)
    assert [str(int(x)) for x in zg["pressure_level__temperature"][:]] == ["500", "850"]
    assert [str(int(x)) for x in zg["pressure_level__geopotential"][:]] == ["500"]

    temp_meta = json.dumps(json.loads((store / "temperature" / "zarr.json").read_text())).lower()
    assert "zstd" in temp_meta and "bitshuffle" in temp_meta and "9" in temp_meta

    ds = xr.open_zarr(store, consolidated=True)
    try:
        assert list(ds["temperature"].dims) == ["time", "pressure_level__temperature", "latitude", "longitude"]
        assert list(ds["geopotential"].dims) == ["time", "pressure_level__geopotential", "latitude", "longitude"]
        np.testing.assert_array_equal(ds["time"].values.astype("datetime64[ns]"), src["time"].values)
        np.testing.assert_array_equal(ds["latitude"].values, src["latitude"].values)
        np.testing.assert_array_equal(ds["longitude"].values, src["longitude"].values)
        np.testing.assert_equal(ds["temperature"].sel(pressure_level__temperature=500).values, src["temperature"].sel(pressure_level=500).values)
        np.testing.assert_equal(ds["temperature"].sel(pressure_level__temperature=850).values, src["temperature"].sel(pressure_level=850).values)
        np.testing.assert_equal(ds["geopotential"].sel(pressure_level__geopotential=500).values, src["geopotential"].sel(pressure_level=500).values)
        assert np.isnan(ds["temperature"].sel(pressure_level__temperature=500).values[1, 1, 1])
        assert ds["temperature"].attrs["units"] == "K"
    finally:
        ds.close()


def test_invalid_selector_rejected_before_publication(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    c = _contract()
    c["fields"][0]["selectors"][0]["value"] = "999"
    with pytest.raises(Exception, match="Invalid pressure level"):
        pipeline_impl.run_pipeline(_lock(c), _inventory(), cache, out)
    assert not out.exists() or not (out / "dataset.zarr").exists()


def test_manifest_sha_mismatch_rejected_and_no_network_needed(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("CDSAPI_KEY", "SECRET_SHOULD_NOT_BE_USED")
    with pytest.raises(Exception, match="sha256 mismatch"):
        pipeline_impl.run_pipeline(_lock(), _inventory(), cache, out)


def test_missing_fixture_fails_safely(tmp_path):
    with pytest.raises(Exception, match="complete source_fixture_manifest"):
        pipeline_impl.run_pipeline(_lock(), _inventory(), tmp_path / "empty-cache", tmp_path / "out")


def test_inclusive_date_end_keeps_final_day_times(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    r = pipeline_impl.run_pipeline(_lock(), _inventory(), cache, out)
    ds = xr.open_zarr(out / r["dataset_artifact"]["store_path"], consolidated=True)
    try:
        assert ds.sizes["time"] == 8
        assert str(ds["time"].values[-1].astype("datetime64[m]")) == "2024-01-02T18:00"
    finally:
        ds.close()


def test_subset_filtering_and_no_unrequested_planes(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    src = _write_fixture(cache)
    c = _contract()
    c["fields"] = [c["fields"][0]]
    c["scope"]["time"]["selected_times"] = ["06:00"]
    c["scope"]["geography"]["cds_area"] = [90, -180, 0, 0]
    r = pipeline_impl.run_pipeline(_lock(c), _inventory(), cache, out)
    ds = xr.open_zarr(out / r["dataset_artifact"]["store_path"], consolidated=True)
    try:
        assert list(ds.data_vars) == ["temperature"]
        assert ds.sizes["time"] == 2
        assert ds.sizes["pressure_level__temperature"] == 1
        assert ds.sizes["latitude"] == 2
        assert ds.sizes["longitude"] == 2
        expected = src["temperature"].sel(time=src.time.dt.hour == 6).sel(pressure_level=500).isel(latitude=[0, 1], longitude=[0, 1]).values
        np.testing.assert_equal(ds["temperature"].isel(pressure_level__temperature=0).values, expected)
    finally:
        ds.close()
