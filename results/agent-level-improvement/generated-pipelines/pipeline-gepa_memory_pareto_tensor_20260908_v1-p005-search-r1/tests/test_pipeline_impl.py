from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline_impl


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": {
            "product_type": ["reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K"},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2"},
                "relative_humidity": {"label": "Relative humidity", "units": "%"},
            }
        },
    }


def _contract(end_date="2024-01-02", fields=None):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "fields": fields or [
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": end_date, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timezone": "UTC", "timestep": "6 hours"},
        },
        "advanced_options": {"dataset_id": "reanalysis-era5-pressure-levels", "data_format": "grib", "download_format": "unarchived"},
        "human_confirmed": True,
    }


def _write_fixture(cache_dir: Path):
    times = pd.date_range("2024-01-01T00:00:00", "2024-01-02T18:00:00", freq="6h")
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 89.75, 89.5], dtype=np.float32)
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = 250.0 + base / 100.0
    geop = 50000.0 + base
    temp[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "t": (("time", "isobaricInhPa", "latitude", "longitude"), temp, {"units": "K"}),
            "z": (("time", "isobaricInhPa", "latitude", "longitude"), geop, {"units": "m2 s-2"}),
        },
        coords={"time": times, "isobaricInhPa": levels, "latitude": lat, "longitude": lon},
        attrs={"source": "unit-test-fixture"},
    )
    raw = cache_dir / "raw_fixture.nc"
    ds.to_netcdf(raw, engine="h5netcdf")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{
            "entry_id": "unit-test-raw",
            "relative_path": raw.name,
            "source": "local-test",
            "size_bytes": raw.stat().st_size,
            "sha256": digest,
        }],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds.rename({"isobaricInhPa": "pressure_level"})


def test_fixture_reuse_publication_semantics_chunks_codec_and_rerun(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    source = _write_fixture(cache)

    lock = {"selected_contract": _contract()}
    result1 = pipeline_impl.run_pipeline(lock, _inventory(), cache, out)
    result2 = pipeline_impl.run_pipeline(lock, _inventory(), cache, out)

    assert result1["cache"]["hits"] == 1
    assert result1["cache"]["misses"] == 0
    assert result1["cache"]["acquired"] == 0
    assert result1["cache"]["reused_keys"] == result2["cache"]["reused_keys"]
    artifact = result1["dataset_artifact"]
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert [c["field_id"] for c in artifact["channels"]] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert all(c["selector_coordinate_paths"] == {"pressure_level": "pressure_level"} for c in artifact["channels"])

    store = out / "dataset.zarr"
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    expected = xr.Dataset(
        {
            "temperature": source["t"].sel(time=reopened.time.values, pressure_level=[500, 850]),
            "geopotential": source["z"].sel(time=reopened.time.values, pressure_level=[500]),
        }
    )
    expected = expected.assign_coords(
        time=source.time.values,
        pressure_level=np.array([500, 850], dtype=np.int32),
        latitude=source.latitude.values,
        longitude=source.longitude.values,
    ).sel(time=reopened.time.values)
    xr.testing.assert_equal(reopened[["temperature", "geopotential"]], expected[["temperature", "geopotential"]])
    assert reopened["temperature"].dims == ("time", "pressure_level", "latitude", "longitude")
    assert reopened.sizes["pressure_level"] == 2
    np.testing.assert_array_equal(reopened["latitude"].values, source["latitude"].values)
    np.testing.assert_array_equal(reopened["longitude"].values, source["longitude"].values)
    assert np.isnan(reopened["temperature"].isel(time=1, pressure_level=0, latitude=1, longitude=2).item())

    import zarr
    group = zarr.open_group(store, mode="r")
    assert tuple(group["temperature"].chunks) == (1, 1, 3, 4)
    assert tuple(group["geopotential"].chunks) == (1, 1, 3, 4)
    codec_text = str(group["temperature"].metadata).lower() + str(getattr(group["temperature"], "compressors", "")).lower()
    assert "blosc" in codec_text
    assert "lz4" in codec_text or "zstd" in codec_text


def test_inclusive_date_only_end_keeps_final_day_times(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir(); out.mkdir()
    _write_fixture(cache)
    result = pipeline_impl.run_pipeline({"contract": _contract(end_date="2024-01-02")}, _inventory(), cache, out)
    reopened = xr.open_zarr(out / result["dataset_artifact"]["store_path"], consolidated=True, zarr_format=3)
    assert str(pd.Timestamp(reopened.time.values[-1])) == "2024-01-02 18:00:00"
    assert reopened.sizes["time"] == 8


def test_contract_validation_rejects_invalid_selector(tmp_path):
    bad = _contract(fields=[{"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "777", "unit": "hPa"}]}])
    with pytest.raises(ValueError, match="Unsupported pressure_level"):
        pipeline_impl.run_pipeline(bad, _inventory(), tmp_path / "cache", tmp_path / "out")


def test_manifest_checksum_is_verified_before_decode(tmp_path):
    cache = tmp_path / "cache"; out = tmp_path / "out"
    cache.mkdir(); out.mkdir()
    _write_fixture(cache)
    manifest_path = cache / "source_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)


def test_no_fixture_fails_closed_without_network_or_credentials(tmp_path):
    with pytest.raises(RuntimeError, match="network acquisition is intentionally disabled"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), tmp_path / "empty-cache", tmp_path / "out")
