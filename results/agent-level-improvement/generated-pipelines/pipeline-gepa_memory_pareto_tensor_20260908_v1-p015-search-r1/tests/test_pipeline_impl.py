from __future__ import annotations

import copy
import hashlib
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import pipeline_impl


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01", "02", "03", "04", "05", "06", "07", "08"],
            "time": [f"{h:02d}:00" for h in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K"},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2"},
            }
        },
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.invalid/no-runtime-use",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-07", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _lock(contract=None):
    return {
        "contract": contract or _contract(),
        "fixed_pipeline_policy": {
            "provider": "ECMWF",
            "dataset_id": "reanalysis-era5-pressure-levels",
            "acquisition_format": "grib",
            "publication_format": "zarr",
            "zarr": {"format_version": 3, "consolidated_metadata": True},
        },
    }


def _write_fixture(cache: Path):
    times = pd.date_range("2024-01-01T00:00", "2024-01-07T18:00", freq="6h")
    levels = np.array([500, 850], dtype="int32")
    lat = np.array([90.0, 89.75, 89.5], dtype="float32")
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype="float32")
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    temp = base + 273.15
    geop = base + 5000.0
    temp[2, 1, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2", "long_name": "Geopotential"}),
        },
        coords={
            "time": times.to_numpy(dtype="datetime64[ns]"),
            "pressure_level": ("pressure_level", levels, {"units": "hPa"}),
            "latitude": ("latitude", lat, {"units": "degrees_north"}),
            "longitude": ("longitude", lon, {"units": "degrees_east"}),
        },
    )
    path = cache / "raw.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    data = path.read_bytes()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "fixture-raw", "relative_path": "raw.nc", "source": "local-fixture", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def test_fixture_first_no_network_publication_and_semantics(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    source = _write_fixture(cache)

    def deny_network(*args, **kwargs):
        raise AssertionError("network was probed")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    result = pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))

    assert result["cache"] == {"hits": 1, "misses": 0, "acquired": 0, "reused_keys": ["fixture-raw"], "acquired_keys": []}
    artifact = result["dataset_artifact"]
    assert artifact["schema_version"] == "dataset_artifact_layout.v1"
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3
    assert {c["field_id"] for c in artifact["channels"]} == {
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    }

    store = out / "dataset.zarr"
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        assert list(reopened.coords) >= ["time", "latitude", "longitude"]
        np.testing.assert_array_equal(reopened["time"].values, source["time"].values)
        np.testing.assert_array_equal(reopened["latitude"].values, source["latitude"].values)
        np.testing.assert_array_equal(reopened["longitude"].values, source["longitude"].values)
        for ch in artifact["channels"]:
            arr_name = ch["array_path"]
            pcoord = ch["selector_coordinate_paths"]["pressure_level"]
            level = int(ch["selectors"]["pressure_level"])
            assert pcoord in reopened.coords
            assert reopened[pcoord].dims == (pcoord,)
            assert int(reopened[pcoord].values[0]) == level
            assert reopened[arr_name].dims == ("time", pcoord, "latitude", "longitude")
            src_name = ch["field_id"].split("[")[0]
            expected = source[src_name].sel(pressure_level=level).expand_dims({pcoord: [level]}).transpose("time", pcoord, "latitude", "longitude")
            np.testing.assert_equal(reopened[arr_name].values, expected.values)
            if src_name == "temperature":
                assert np.isnan(reopened[arr_name].values[2, 0, 1, 2])
            assert "units" in reopened[arr_name].attrs
    finally:
        reopened.close()


def test_chunks_are_sample_aligned_and_uncompressed(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    artifact = pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))["dataset_artifact"]
    for ch in artifact["channels"]:
        arr = zarr.open(str(out / "dataset.zarr" / ch["array_path"]), mode="r")
        assert tuple(arr.chunks) == (1, 1, 3, 4)
        codec_text = " ".join(type(c).__name__.lower() + repr(c).lower() for c in arr.metadata.codecs)
        for forbidden in ("blosc", "zstd", "lz4", "gzip", "zlib", "numcodecs", "shuffle", "bitshuffle"):
            assert forbidden not in codec_text


def test_invalid_selector_is_rejected_before_publication(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    c = _contract()
    c["fields"][0]["selectors"][0]["value"] = "777"
    with pytest.raises(ValueError, match="invalid pressure_level"):
        pipeline_impl.run_pipeline(_lock(c), _inventory(), str(cache), str(out))
    assert not (out / "dataset.zarr").exists()


def test_fixture_manifest_hash_is_verified(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))


def test_missing_fixture_fails_without_acquisition(tmp_path):
    with pytest.raises(RuntimeError, match="remote acquisition is disabled"):
        pipeline_impl.run_pipeline(_lock(), _inventory(), str(tmp_path / "cache"), str(tmp_path / "out"))


def test_inclusive_date_end_keeps_final_selected_times(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    result = pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / result["dataset_artifact"]["store_path"], consolidated=True, zarr_format=3)
    try:
        assert ds.sizes["time"] == 28
        assert pd.Timestamp(ds["time"].values[-1]) == pd.Timestamp("2024-01-07T18:00:00")
    finally:
        ds.close()


def test_rerun_is_equivalent_and_reuses_fixture(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    r1 = pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))
    ds1 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    r2 = pipeline_impl.run_pipeline(_lock(), _inventory(), str(cache), str(out))
    ds2 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    assert r1 == r2
    xr.testing.assert_identical(ds1, ds2)


def test_subregion_filtering_uses_contract_not_seed_only(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    c = _contract()
    c["scope"]["geography"] = {"area": "subregion", "cds_area": [90, -180, 89.75, -179.75], "cds_area_order": ["north", "west", "south", "east"]}
    pipeline_impl.run_pipeline(_lock(c), _inventory(), str(cache), str(out))
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3)
    try:
        assert ds.sizes["latitude"] == 2
        assert ds.sizes["longitude"] == 2
        np.testing.assert_array_equal(ds["latitude"].values, np.array([90.0, 89.75], dtype="float32"))
        np.testing.assert_array_equal(ds["longitude"].values, np.array([-180.0, -179.75], dtype="float32"))
    finally:
        ds.close()
