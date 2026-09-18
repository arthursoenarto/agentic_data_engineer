from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import pipeline_impl


PIPE_INV = {
    "schema_version": "dataset_inventory.v1",
    "dataset_slug": "reanalysis_era5_pressure_levels",
    "dataset_id": "reanalysis-era5-pressure-levels",
    "provider": "ECMWF",
    "source_url": "https://example.invalid/era5",
    "catalogue_metadata": {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2026-06-30T00:00:00+00:00"]]}}},
    "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
    "options": {
        "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
        "variable": ["temperature", "geopotential"],
        "pressure_level": ["500", "850"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": ["grib", "netcdf"],
        "download_format": ["zip", "unarchived"],
    },
    "option_metadata": {"variable": {"temperature": {"units": "K"}, "geopotential": {"units": "m2 s-2"}}},
}


def seed_contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.invalid/era5",
        "provider": "Copernicus Climate Data Store",
        "fields": [
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
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


def make_fixture(cache_dir: Path, scalar_level: bool = False):
    times = np.array([np.datetime64(f"2024-01-{d:02d}T{h:02d}:00") for d in range(1, 8) for h in (0, 6, 12, 18)], dtype="datetime64[ns]")
    lat = np.array([90.0, 89.75, 89.5, 89.25], dtype="float32")
    lon = np.array([-180.0, -179.75, -179.5, -179.25, -179.0], dtype="float32")
    levels = np.array([500, 850], dtype="int32")
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    if scalar_level:
        ds = xr.Dataset(
            {
                "t": (("time", "latitude", "longitude"), base[:, 0]),
                "z": (("time", "latitude", "longitude"), base[:, 0] + 10000),
            },
            coords={"time": times, "latitude": lat, "longitude": lon, "pressure_level": np.array(500, dtype="int32")},
        )
    else:
        ds = xr.Dataset(
            {
                "t": (("time", "pressure_level", "latitude", "longitude"), base),
                "z": (("time", "pressure_level", "latitude", "longitude"), base + 10000),
            },
            coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
        )
    path = cache_dir / "raw.nc"
    ds.to_netcdf(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "raw", "relative_path": "raw.nc", "source": "unit-test", "size_bytes": path.stat().st_size, "sha256": digest}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_publication_artifact_mapping_rerun_and_readback(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache)

    result1 = pipeline_impl.run_pipeline({"selected_contract": seed_contract()}, PIPE_INV, str(cache), str(out))
    result2 = pipeline_impl.run_pipeline(seed_contract(), PIPE_INV, str(cache), str(out))
    assert result1 == result2
    assert result1["cache"]["hits"] == 1
    artifact = result1["dataset_artifact"]
    assert artifact["store_path"] == "era5_pressure_levels.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3

    store = out / artifact["store_path"]
    ds = xr.open_zarr(store, consolidated=True)
    try:
        assert ds.sizes["time"] == 28
        assert list(ds.coords)[:3] == ["time", "latitude", "longitude"]
        for ch in artifact["channels"]:
            arr = ds[ch["array_path"]]
            assert arr.dtype == np.float32
            assert arr.dims == ("time", "pressure_level", "latitude", "longitude")
            assert arr.sizes["pressure_level"] == 1
            for coord_path in ch["selector_coordinate_paths"].values():
                assert coord_path in ds.coords
            _ = arr.isel(time=0, pressure_level=0).values
    finally:
        ds.close()


def test_chunk_geometry(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache)
    artifact = pipeline_impl.run_pipeline(seed_contract(), PIPE_INV, str(cache), str(out))["dataset_artifact"]
    group = zarr.open_group(out / artifact["store_path"], mode="r")
    for ch in artifact["channels"]:
        chunks = group[ch["array_path"]].chunks
        assert chunks[0] == 1
        assert chunks[1] == 1
        assert chunks[-1] == 5
        assert chunks[-2] >= 1


def test_contract_validation_rejects_invalid_level(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache)
    c = seed_contract()
    c["fields"][0]["selectors"][0]["value"] = "700"
    with pytest.raises(ValueError):
        pipeline_impl.run_pipeline(c, PIPE_INV, str(cache), str(out))


def test_fixture_hash_is_authoritative(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache)
    (cache / "raw.nc").write_bytes((cache / "raw.nc").read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="size mismatch|sha256 mismatch"):
        pipeline_impl.run_pipeline(seed_contract(), PIPE_INV, str(cache), str(out))


def test_scalar_provider_level_reconstructed_as_length_one_selector_dimension(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache, scalar_level=True)
    c = seed_contract()
    c["fields"] = [c["fields"][0], c["fields"][2]]
    artifact = pipeline_impl.run_pipeline(c, PIPE_INV, str(cache), str(out))["dataset_artifact"]
    ds = xr.open_zarr(out / artifact["store_path"], consolidated=True)
    try:
        for ch in artifact["channels"]:
            arr = ds[ch["array_path"]]
            assert arr.dims == ("time", "pressure_level", "latitude", "longitude")
            assert arr.sizes["pressure_level"] == 1
            coord_path = ch["selector_coordinate_paths"]["pressure_level"]
            assert int(ds[coord_path].values[0]) == 500
    finally:
        ds.close()


def test_exact_time_filtering_date_only_end_includes_final_day(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    make_fixture(cache)
    c = seed_contract()
    c["scope"]["date_range"] = {"start_date": "2024-01-02", "end_date": "2024-01-03", "inclusive": True}
    c["scope"]["time"]["selected_times"] = ["06:00", "18:00"]
    artifact = pipeline_impl.run_pipeline(c, PIPE_INV, str(cache), str(out))["dataset_artifact"]
    ds = xr.open_zarr(out / artifact["store_path"], consolidated=True)
    try:
        assert [str(t) for t in ds.time.values.astype("datetime64[m]")] == [
            "2024-01-02T06:00",
            "2024-01-02T18:00",
            "2024-01-03T06:00",
            "2024-01-03T18:00",
        ]
    finally:
        ds.close()
