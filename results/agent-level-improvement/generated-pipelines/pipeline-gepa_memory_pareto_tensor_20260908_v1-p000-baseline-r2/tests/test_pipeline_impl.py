from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", ROOT / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024", "2025"],
            "month": ["01", "02"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850", "700"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_units": {"pressure_level": "hPa", "time": "UTC", "area": "north/west/south/east degrees"},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "air temperature"},
                "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>", "description": "geopotential"},
                "relative_humidity": {"label": "Relative humidity", "units": "%", "description": "relative humidity"},
            }
        },
    }


def _contract(start="2024-01-01", end="2024-01-02", fields=None, times=None, area=None):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.test",
        "intent": "test",
        "summary": "test",
        "provider": "ECMWF",
        "dataset_family": "reanalysis",
        "access_methods": [],
        "credential_requirements": [],
        "fields": fields or [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": start, "end_date": end, "inclusive": True},
            "geography": {"area": "global", "cds_area": area or [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": times or ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "defaults_used": [],
        "human_editable_fields": [],
        "assumptions": [],
        "risks_or_unknowns": [],
        "evidence": [],
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "pipeline_requirements": None,
        "human_confirmed": True,
        "recommended_next_step": None,
    }


def _write_fixture(cache_dir: Path, *, packed=False):
    raw = cache_dir / "raw"
    raw.mkdir(parents=True)
    path = raw / "fixture.nc"
    times = pd.to_datetime([
        "2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00", "2024-01-01T18:00:00",
        "2024-01-02T00:00:00", "2024-01-02T06:00:00", "2024-01-02T12:00:00", "2024-01-02T18:00:00",
    ])
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([1.0, 0.0, -1.0])
    lon = np.array([-1.0, 0.0, 1.0])
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    temp = 250.0 + base / 100.0
    geo = 50000.0 + base
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geo, {"units": "m2 s-2"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    if packed:
        # Force CF packing in the source file; xarray.open_dataset must decode it.
        encoding = {"temperature": {"dtype": "int16", "scale_factor": 0.01, "add_offset": 250.0, "_FillValue": -32768}}
        ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    else:
        ds.to_netcdf(path, engine="netcdf4")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "fixture", "relative_path": "raw/fixture.nc", "source": "unit-test", "size_bytes": path.stat().st_size, "sha256": sha}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _run(tmp_path, contract=None, inventory=None, packed=False):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache, packed=packed)
    result = pipeline_impl.run_pipeline({"envelope": {"selected_contract": contract or _contract()}}, inventory or _inventory(), cache, out)
    return result, out


def test_contract_validation_rejects_invalid_selector(tmp_path):
    c = _contract(fields=[{"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "999", "unit": "hPa", "label": None}]}])
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_fixture(cache)
    with pytest.raises(pipeline_impl.ContractError):
        pipeline_impl.run_pipeline(c, _inventory(), cache, tmp_path / "out")


def test_source_fixture_reuse_and_publication_readback(tmp_path):
    result, out = _run(tmp_path)
    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired"] == 0
    assert result["dataset_artifact"]["storage_format"] == "zarr"
    assert (out / "dataset.zarr" / "zarr.json").exists()
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3)
    try:
        assert "temperature_pressure_level_500" in ds
        assert "temperature_pressure_level_850" in ds
        assert "geopotential_pressure_level_500" in ds
        dim = "pressure_level__temperature_pressure_level_500"
        assert dim in ds["temperature_pressure_level_500"].dims
        assert ds.sizes[dim] == 1
        assert list(map(str, ds[dim].values.tolist())) == ["500"]
        assert len(ds.time) == 8
    finally:
        ds.close()


def test_filtering_area_time_and_inclusive_end_day(tmp_path):
    c = _contract(start="2024-01-01", end="2024-01-02", times=["18:00"], area=[0.5, -0.5, -0.5, 0.5])
    result, out = _run(tmp_path, contract=c)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3)
    try:
        assert pd.to_datetime(ds.time.values).strftime("%Y-%m-%dT%H:%M:%S").tolist() == ["2024-01-01T18:00:00", "2024-01-02T18:00:00"]
        assert ds.latitude.values.tolist() == [0.0]
        assert ds.longitude.values.tolist() == [0.0]
    finally:
        ds.close()
    assert result["dataset_artifact"]["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}


def test_cf_packing_is_decoded_before_zarr_publication(tmp_path):
    result, out = _run(tmp_path, packed=True)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3)
    try:
        value = float(ds["temperature_pressure_level_500"].isel(time=0, latitude=0, longitude=0).values[0])
        assert np.isclose(value, 250.0, atol=0.02)
        assert ds["temperature_pressure_level_500"].dtype.kind == "f"
    finally:
        ds.close()


def test_exact_rerun_result_and_store_are_stable(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    c = _contract()
    inv = _inventory()
    r1 = pipeline_impl.run_pipeline({"contract": c}, inv, cache, out)
    ds1 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    r2 = pipeline_impl.run_pipeline({"contract": c}, inv, cache, out)
    ds2 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    assert r1 == r2
    xr.testing.assert_identical(ds1, ds2)


def test_missing_or_tampered_fixture_fails_before_publication(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    path = _write_fixture(cache)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(pipeline_impl.SourceFixtureError):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")
    assert not (tmp_path / "out" / "dataset.zarr").exists()
