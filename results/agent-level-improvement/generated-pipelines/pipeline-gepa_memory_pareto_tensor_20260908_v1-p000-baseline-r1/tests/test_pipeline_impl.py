from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pipeline_impl


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory() -> dict:
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "provider": "ECMWF",
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["geopotential", "temperature", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "catalogue_metadata": {
            "extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2026-06-30T00:00:00+00:00"]]}},
        },
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Air temperature."},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2", "description": "Geopotential."},
            }
        },
    }


def _contract(times=None, fields=None, start="2024-01-01", end="2024-01-01") -> dict:
    if times is None:
        times = ["00:00", "06:00", "12:00", "18:00"]
    if fields is None:
        fields = [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
        ]
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.invalid",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "fields": fields,
        "scope": {
            "date_range": {"start_date": start, "end_date": end, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": times, "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _make_fixture(cache_dir: Path, *, packed: bool = False) -> Path:
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True)
    path = raw_dir / "era5_fixture.nc"
    times = pd.to_datetime(["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00"])
    levels = np.array([500, 850], dtype=np.int32)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([10.0, 11.0], dtype=np.float64)
    shape = (len(times), len(levels), len(lats), len(lons))
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    temperature = 250.0 + base / 10.0
    geopotential = 50000.0 + base
    temperature[1, 0, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temperature, {"units": "K"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geopotential, {"units": "m2 s-2"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lats, "longitude": lons},
    )
    encoding = {}
    if packed:
        encoding["temperature"] = {"dtype": "int16", "scale_factor": 0.1, "add_offset": 250.0, "_FillValue": -32768}
    ds.to_netcdf(path, engine="scipy", encoding=encoding)
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [
            {
                "entry_id": "fixture-era5-small",
                "relative_path": "raw/era5_fixture.nc",
                "source": "pytest-generated",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        ],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_contract_validation_rejects_unsupported_variable(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache)
    bad = _contract(fields=[{"name": "not_a_variable", "display_name": "bad", "selectors": [{"dimension": "pressure_level", "value": "500"}]}])
    with pytest.raises(pipeline_impl.PipelineValidationError, match="unsupported variable"):
        pipeline_impl.run_pipeline({"contract": bad}, _inventory(), cache, tmp_path / "out")


def test_source_fixture_reuse_and_no_cache_writes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache)
    before = sorted(str(p.relative_to(cache)) for p in cache.rglob("*"))
    result = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, tmp_path / "out")
    after = sorted(str(p.relative_to(cache)) for p in cache.rglob("*"))
    assert before == after
    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired"] == 0
    assert result["cache"]["reused_keys"][0].startswith("fixture:fixture-era5-small:sha256:")


def test_filtering_inclusive_date_end_and_pressure_dimension(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache)
    result = pipeline_impl.run_pipeline({"dataset_contract": _contract(times=["18:00"])}, _inventory(), cache, tmp_path / "out")
    store = tmp_path / "out" / result["dataset_artifact"]["store_path"]
    ds = xr.open_zarr(store, consolidated=True)
    assert list(pd.to_datetime(ds.time.values).strftime("%H:%M")) == ["18:00"]
    assert "pressure_level" in ds.dims
    assert ds.sizes["pressure_level"] == 2
    assert ds.sizes["time"] == 1
    ds.close()


def test_publication_readback_channels_and_decoded_missingness(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache, packed=True)
    result = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, tmp_path / "out")
    artifact = result["dataset_artifact"]
    assert artifact["storage_format"] == "zarr"
    assert artifact["dimensions"]["pressure_level"] == "pressure_level"
    assert (tmp_path / "out" / "dataset.zarr" / "zarr.json").is_file()
    ds = xr.open_zarr(tmp_path / "out" / "dataset.zarr", consolidated=True)
    var = 'temperature[pressure_level="500"]'
    assert var in ds.data_vars
    assert np.isnan(ds[var].sel(time=np.datetime64("2024-01-01T06:00:00"), pressure_level=500, latitude=1.0, longitude=10.0).item())
    assert "scale_factor" not in ds[var].encoding
    assert "add_offset" not in ds[var].encoding
    channel_ids = {c["field_id"] for c in artifact["channels"]}
    assert 'geopotential[pressure_level="500"]' in channel_ids
    ds.close()


def test_exact_rerun_behavior_same_values_and_manifest(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache)
    result1 = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, tmp_path / "out1")
    result2 = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, tmp_path / "out2")
    assert result1["cache"] == result2["cache"]
    assert result1["dataset_artifact"] == result2["dataset_artifact"]
    ds1 = xr.open_zarr(tmp_path / "out1" / "dataset.zarr", consolidated=True)
    ds2 = xr.open_zarr(tmp_path / "out2" / "dataset.zarr", consolidated=True)
    xr.testing.assert_identical(ds1, ds2)
    ds1.close(); ds2.close()


def test_manifest_hash_mismatch_fails_before_publication(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _make_fixture(cache)
    manifest_path = cache / "source_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(pipeline_impl.PipelineValidationError, match="sha256 mismatch"):
        pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, out)
    assert not (out / "dataset.zarr").exists()
