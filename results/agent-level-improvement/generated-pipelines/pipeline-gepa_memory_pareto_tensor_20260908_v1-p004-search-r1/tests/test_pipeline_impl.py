from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pipeline_impl


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "provider": "ECMWF",
        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850", "700"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "option_units": {"pressure_level": "hPa", "time": "UTC", "area": "north/west/south/east degrees"},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Air temperature."},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2", "description": "Geopotential."},
            }
        },
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test contract",
        "source_url": "https://example.invalid/era5",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
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


def _make_source_fixture(cache: Path):
    times = pd.date_range("2024-01-01T00:00", "2024-01-07T18:00", freq="6h")
    levels = np.asarray([500, 850], dtype=np.int32)
    lat = np.asarray([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float32)
    lon = np.asarray([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = 250.0 + base / 10.0
    geop = 5000.0 + base * 2.0
    temp[2, 1, 3, 4] = np.nan
    geop[4, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    raw = cache / "raw" / "era5_fixture.nc"
    raw.parent.mkdir(parents=True)
    # Packed CF encoding verifies mask/scale decode while keeping the test small.
    ds.to_netcdf(
        raw,
        engine="h5netcdf",
        encoding={
            "temperature": {"dtype": "int16", "scale_factor": 0.1, "add_offset": 250.0, "_FillValue": -32767},
            "geopotential": {"dtype": "int16", "scale_factor": 2.0, "add_offset": 5000.0, "_FillValue": -32767},
        },
    )
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [
            {"entry_id": "era5-test", "relative_path": "raw/era5_fixture.nc", "source": "local-test", "size_bytes": raw.stat().st_size, "sha256": sha}
        ],
    }
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw


def _hash_tree(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _array_metadata(store: Path, name: str):
    return json.loads((store / name / "zarr.json").read_text())


def test_multi_field_multi_pressure_publication_readback_chunks_and_compressor(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    _make_source_fixture(cache)
    result = pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), cache, out)

    assert result["cache"]["hits"] == 1
    assert result["cache"]["acquired"] == 0
    artifact = result["dataset_artifact"]
    assert artifact["schema_version"] == "dataset_artifact_layout.v1"
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "era5_pressure_levels.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert [c["field_id"] for c in artifact["channels"]] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]

    store = out / artifact["store_path"]
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert root_meta.get("consolidated_metadata")

    ds = xr.open_zarr(store, consolidated=True, zarr_format=3, chunks=None)
    try:
        assert set(ds.data_vars) == {"temperature", "geopotential"}
        assert ds["temperature"].dims == ("time", "pressure_level__temperature", "latitude", "longitude")
        assert ds["geopotential"].dims == ("time", "pressure_level__geopotential", "latitude", "longitude")
        assert ds.sizes["time"] == 28
        assert ds.sizes["pressure_level__temperature"] == 2
        assert ds.sizes["pressure_level__geopotential"] == 1
        np.testing.assert_array_equal(ds["pressure_level__temperature"].values, np.asarray([500, 850], dtype=np.int32))
        np.testing.assert_array_equal(ds["pressure_level__geopotential"].values, np.asarray([500], dtype=np.int32))
        assert np.isnan(ds["temperature"].isel(time=2, pressure_level__temperature=1, latitude=3, longitude=4).item())
        assert np.isnan(ds["geopotential"].isel(time=4, pressure_level__geopotential=0, latitude=0, longitude=0).item())
    finally:
        ds.close()

    temp_meta = _array_metadata(store, "temperature")
    geo_meta = _array_metadata(store, "geopotential")
    assert tuple(temp_meta["chunk_grid"]["configuration"]["chunk_shape"]) == (1, 1, 5, 5)
    assert tuple(geo_meta["chunk_grid"]["configuration"]["chunk_shape"]) == (1, 1, 5, 5)
    meta_text = json.dumps(temp_meta).lower()
    assert "blosc" in meta_text and "zstd" in meta_text and "bitshuffle" in meta_text
    assert "clevel" in meta_text and "3" in meta_text
    assert '"clevel": 7' not in meta_text


def test_invalid_inventory_option_is_rejected_before_publication(tmp_path):
    cache = tmp_path / "cache"
    _make_source_fixture(cache)
    bad = _contract()
    bad["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(pipeline_impl.PipelineError, match="invalid pressure_level"):
        pipeline_impl.run_pipeline(bad, _inventory(), cache, tmp_path / "out")


def test_manifest_hash_and_size_verification(tmp_path):
    cache = tmp_path / "cache"
    raw = _make_source_fixture(cache)
    raw.write_bytes(raw.read_bytes() + b"corrupt")
    with pytest.raises(pipeline_impl.PipelineError, match="size_bytes verification failed"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")


def test_read_only_fixture_cache_is_reused_without_writes(tmp_path):
    cache = tmp_path / "cache"
    _make_source_fixture(cache)
    before = _hash_tree(cache)
    for p in cache.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    result = pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")
    after = _hash_tree(cache)
    assert before == after
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired_keys"] == []


def test_rerun_is_deterministic_and_replaces_existing_store(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    _make_source_fixture(cache)
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    first = _hash_tree(out / "era5_pressure_levels.zarr")
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    second = _hash_tree(out / "era5_pressure_levels.zarr")
    assert first == second


def test_inclusive_date_only_end_includes_final_day_selected_times(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    _make_source_fixture(cache)
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    ds = xr.open_zarr(out / "era5_pressure_levels.zarr", consolidated=True, zarr_format=3, chunks=None)
    try:
        times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        assert pd.Timestamp("2024-01-07T18:00:00") in times
        assert pd.Timestamp("2024-01-08T00:00:00") not in times
    finally:
        ds.close()
