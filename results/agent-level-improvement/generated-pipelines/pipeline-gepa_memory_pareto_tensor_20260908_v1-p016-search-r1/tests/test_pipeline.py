from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pipeline_impl


PIPELINE_FIELDS = [
    {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
    {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
    {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
]


def inventory():
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
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": ["500", "700", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Air temperature"},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2", "description": "Geopotential"},
            }
        },
    }


def contract(fields=None, start="2024-01-01", end="2024-01-07", times=None):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.invalid",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "access_methods": [],
        "credential_requirements": [],
        "fields": fields or PIPELINE_FIELDS,
        "scope": {
            "date_range": {"start_date": start, "end_date": end, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": times or ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def make_fixture(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    times = pd.date_range("2024-01-01T00:00:00", "2024-01-07T18:00:00", freq="6h")
    levels = np.array([500, 700, 850], dtype=np.int32)
    lat = np.array([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float64)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float64)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temperature = base + 273.15
    geopotential = base * 10.0
    temperature[1, 0, 2, 3] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temperature, {"units": "K", "quality": "fixture-quality-metadata"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geopotential, {"units": "m2 s-2"}),
            "relative_humidity": (("time", "pressure_level", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
    )
    src = cache / "era5_fixture.nc"
    ds.to_netcdf(src, engine="h5netcdf")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "fixture", "relative_path": src.name, "source": "local-test", "size_bytes": src.stat().st_size, "sha256": sha}],
    }
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache, ds


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_seed(tmp_path: Path):
    cache, source = make_fixture(tmp_path)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline({"lock": {"contract": contract()}}, inventory(), cache, out)
    return result, out, source


def test_requires_verified_fixture_and_rejects_bad_contract(tmp_path):
    with pytest.raises(ValueError, match="source_fixture_manifest"):
        pipeline_impl.run_pipeline(contract(), inventory(), tmp_path / "empty", tmp_path / "out")
    cache, _ = make_fixture(tmp_path)
    bad = contract(fields=[{"name": "not_a_variable", "selectors": []}])
    with pytest.raises(ValueError, match="not in inventory"):
        pipeline_impl.run_pipeline(bad, inventory(), cache, tmp_path / "out2")


def test_publication_grouping_chunks_values_metadata_and_no_unrequested_content(tmp_path):
    result, out, source = run_seed(tmp_path)
    artifact = result["dataset_artifact"]
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "dataset.zarr"
    assert result["cache"]["hits"] == 1 and result["cache"]["misses"] == 0 and result["cache"]["acquired"] == 0

    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    try:
        assert set(ds.data_vars) == {"temperature", "geopotential"}
        assert "relative_humidity" not in ds
        assert tuple(ds["temperature"].dims) == ("time", "temperature_pressure_level", "latitude", "longitude")
        assert tuple(ds["geopotential"].dims) == ("time", "geopotential_pressure_level", "latitude", "longitude")
        assert ds.sizes["temperature_pressure_level"] == 2
        assert ds.sizes["geopotential_pressure_level"] == 1
        np.testing.assert_array_equal(ds["temperature_pressure_level"].values, np.array([500, 850], dtype=np.int32))
        np.testing.assert_array_equal(ds["geopotential_pressure_level"].values, np.array([500], dtype=np.int32))
        assert "700" not in [str(x) for x in ds["temperature_pressure_level"].values]
        assert len(ds.time) == 28
        assert str(ds.time.values[-1]).startswith("2024-01-07T18:00")
        assert ds["temperature"].attrs["quality"] == "fixture-quality-metadata"

        expected_temp = source["temperature"].sel(time=ds.time.values, pressure_level=[500, 850]).values
        expected_geo = source["geopotential"].sel(time=ds.time.values, pressure_level=[500]).values
        np.testing.assert_array_equal(ds["temperature"].values, expected_temp)
        np.testing.assert_array_equal(ds["geopotential"].values, expected_geo)
        assert np.isnan(ds["temperature"].isel(time=1, temperature_pressure_level=0, latitude=2, longitude=3).values)
    finally:
        ds.close()

    channels = artifact["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    temp_channels = [c for c in channels if c["array_path"] == "temperature"]
    assert len(temp_channels) == 2
    assert {c["selectors"]["pressure_level"] for c in temp_channels} == {"500", "850"}
    assert all(c["selector_coordinate_paths"]["pressure_level"] == "temperature_pressure_level" for c in temp_channels)


def test_zarr_v3_consolidated_chunk_topology_and_codec_configuration(tmp_path):
    result, out, _ = run_seed(tmp_path)
    root_meta = json.loads((out / "dataset.zarr" / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    import zarr
    group = zarr.open_group(out / "dataset.zarr", mode="r")
    assert tuple(group["temperature"].chunks) == (1, 1, 5, 5)
    assert tuple(group["geopotential"].chunks) == (1, 1, 5, 5)
    codec_text = json.dumps(pipeline_impl._jsonable(group["temperature"].metadata.codecs)).lower()
    if result["warnings"]:
        assert "lz4" in codec_text and "clevel" in codec_text
    else:
        assert "blosc" not in codec_text and "zstd" not in codec_text and "gzip" not in codec_text and "lz4" not in codec_text


def test_exact_rerun_behavior(tmp_path):
    cache, _ = make_fixture(tmp_path)
    out = tmp_path / "out"
    first = pipeline_impl.run_pipeline(contract(), inventory(), cache, out)
    first_hash = tree_hash(out / "dataset.zarr")
    second = pipeline_impl.run_pipeline(contract(), inventory(), cache, out)
    second_hash = tree_hash(out / "dataset.zarr")
    assert first == second
    assert first_hash == second_hash


def test_filtering_other_valid_lock_and_full_tensor_read_path(tmp_path):
    cache, source = make_fixture(tmp_path)
    fields = [
        {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
        {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
    ]
    c = contract(fields=fields, start="2024-01-02", end="2024-01-02", times=["06:00", "18:00"])
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline(c, inventory(), cache, out)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    try:
        assert len(ds.time) == 2
        assert str(ds.time.values[0]).startswith("2024-01-02T06:00")
        assert str(ds.time.values[1]).startswith("2024-01-02T18:00")
        assert list(ds["temperature_pressure_level"].values) == [850]
        planes = []
        for channel in result["dataset_artifact"]["channels"]:
            arr = ds[channel["array_path"]]
            indexers = {"time": 0}
            for sel_dim, sel_value in channel["selectors"].items():
                coord = channel["selector_coordinate_paths"][sel_dim]
                values = [str(int(v)) for v in ds[coord].values]
                indexers[coord] = values.index(sel_value)
            planes.append(arr.isel(indexers).values)
        tensor = np.stack(planes, axis=0)
        assert tensor.shape == (2, len(ds.latitude), len(ds.longitude))
        expected0 = source["temperature"].sel(time=np.datetime64("2024-01-02T06:00:00"), pressure_level=850).values
        np.testing.assert_array_equal(tensor[0], expected0)
    finally:
        ds.close()


def test_fixture_path_containment_and_hash_are_enforced(tmp_path):
    cache, _ = make_fixture(tmp_path)
    manifest_path = cache / "source_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline_impl.run_pipeline(contract(), inventory(), cache, tmp_path / "out")

    shutil.rmtree(cache)
    cache.mkdir()
    (cache / "source_fixture_manifest.json").write_text(json.dumps({
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "escape", "relative_path": "../x", "source": "x", "size_bytes": 1, "sha256": "0" * 64}],
    }))
    with pytest.raises(ValueError, match="escapes|missing"):
        pipeline_impl.run_pipeline(contract(), inventory(), cache, tmp_path / "out2")
