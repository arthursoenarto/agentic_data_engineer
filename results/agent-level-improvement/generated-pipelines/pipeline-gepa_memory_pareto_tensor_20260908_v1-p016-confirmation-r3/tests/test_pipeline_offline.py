from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

PIPELINE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", PIPELINE_DIR / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "provider": "ECMWF",
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["geopotential", "relative_humidity", "temperature"],
            "year": ["2024", "2025"],
            "month": ["01", "02"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "700", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_metadata": {"variable": {"temperature": {"label": "Temperature", "units": "K"}, "geopotential": {"label": "Geopotential", "units": "m2 s-2"}}},
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "human_confirmed": True,
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-02", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
    }


def _make_fixture(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    times = np.array([np.datetime64(f"2024-01-0{d}T{h:02d}:00:00", "ns") for d in [1, 2, 3] for h in [0, 6, 12, 18]])
    levels = np.array([500, 700, 850], dtype=np.int32)
    lat = np.array([90.0, 0.0, -90.0], dtype=np.float32)
    lon = np.array([-180.0, 0.0, 180.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = base + 273.15
    geop = base + 10000.0
    rh = base / 100.0
    temp[1, 2, 1, 1] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "quality_flag_meaning": "synthetic exact fixture"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2"}),
            "relative_humidity": (("time", "pressure_level", "latitude", "longitude"), rh, {"units": "%"}),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
        attrs={"source": "offline synthetic ERA5-like fixture"},
    )
    raw = cache / "era5_fixture.nc"
    ds.to_netcdf(raw, engine="h5netcdf")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {"schema_version": "source_fixture_manifest.v1", "entries": [{"entry_id": "synthetic-era5", "relative_path": raw.name, "source": "unit-test", "size_bytes": raw.stat().st_size, "sha256": digest}]}
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache, ds


def _run(tmp_path: Path):
    cache, source = _make_fixture(tmp_path)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), cache, out)
    return result, out / "era5_pressure_levels.zarr", source


def _codec_text(arr):
    meta = getattr(arr, "metadata", None)
    if meta is not None and hasattr(meta, "codecs"):
        return json.dumps([repr(c) for c in meta.codecs]).lower()
    return repr(arr).lower()


def test_contract_validation_rejects_bad_selector(tmp_path):
    cache, _ = _make_fixture(tmp_path)
    bad = _contract()
    bad["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(bad, _inventory(), cache, tmp_path / "out")


def test_requires_complete_verified_fixture_before_execution(tmp_path):
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")


def test_fixture_hash_and_path_containment_are_enforced(tmp_path):
    cache, _ = _make_fixture(tmp_path)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")


def test_publication_grouping_filtering_metadata_and_exact_values(tmp_path):
    result, store, source = _run(tmp_path)
    assert result["cache"]["hits"] == 1 and result["cache"]["misses"] == 0 and result["cache"]["acquired"] == 0
    assert result["dataset_artifact"]["storage_format"] == "zarr"
    assert (store / "zarr.json").exists()
    assert "consolidated_metadata" in (store / "zarr.json").read_text()

    temp = xr.open_zarr(store, group="temperature", consolidated=True, zarr_format=3)
    geop = xr.open_zarr(store, group="geopotential", consolidated=True, zarr_format=3)
    try:
        assert list(temp.data_vars) == ["temperature"]
        assert list(geop.data_vars) == ["geopotential"]
        assert temp["temperature"].dims == ("time", "pressure_level", "latitude", "longitude")
        assert geop["geopotential"].dims == ("time", "pressure_level", "latitude", "longitude")
        assert temp.sizes["time"] == 8
        assert geop.sizes["time"] == 8
        assert temp["pressure_level"].values.tolist() == [500, 850]
        assert geop["pressure_level"].values.tolist() == [500]
        assert np.isnan(temp["temperature"].sel(time=np.datetime64("2024-01-01T06:00:00"), pressure_level=850, latitude=0.0, longitude=0.0).item())
        expected_temp = source["temperature"].sel(time=temp["time"].values, pressure_level=[500, 850]).transpose("time", "pressure_level", "latitude", "longitude")
        expected_geop = source["geopotential"].sel(time=geop["time"].values, pressure_level=[500]).transpose("time", "pressure_level", "latitude", "longitude")
        xr.testing.assert_identical(temp["temperature"], expected_temp)
        xr.testing.assert_identical(geop["geopotential"], expected_geop)
        assert temp["temperature"].attrs["quality_flag_meaning"] == "synthetic exact fixture"
    finally:
        temp.close(); geop.close()


def test_artifact_channels_grouped_and_selector_paths_unambiguous(tmp_path):
    result, _, _ = _run(tmp_path)
    channels = result["dataset_artifact"]["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    temp_channels = [c for c in channels if c["array_path"] == "temperature/temperature"]
    assert len(temp_channels) == 2
    assert {c["selectors"]["pressure_level"] for c in temp_channels} == {"500", "850"}
    assert all(c["selector_coordinate_paths"] == {"pressure_level": "temperature/pressure_level"} for c in temp_channels)


def test_chunks_and_codec_policy(tmp_path):
    result, store, _ = _run(tmp_path)
    root = zarr.open_group(store, mode="r")
    assert tuple(root["temperature"]["temperature"].chunks) == (1, 1, 3, 3)
    assert tuple(root["geopotential"]["geopotential"].chunks) == (1, 1, 3, 3)
    temp_codec = _codec_text(root["temperature"]["temperature"])
    if not result["warnings"]:
        assert "blosc" not in temp_codec and "zstd" not in temp_codec and "lz4" not in temp_codec
    else:
        assert "lz4" in temp_codec and "clevel" in temp_codec


def test_no_unrequested_planes_variables_or_sidecar_tensor_store(tmp_path):
    _, store, _ = _run(tmp_path)
    root = zarr.open_group(store, mode="r")
    assert set(root.group_keys()) == {"temperature", "geopotential"}
    assert set(root["temperature"].array_keys()) == {"temperature", "time", "pressure_level", "latitude", "longitude"}
    assert set(root["geopotential"].array_keys()) == {"geopotential", "time", "pressure_level", "latitude", "longitude"}
    temp = xr.open_zarr(store, group="temperature", consolidated=True, zarr_format=3)
    geop = xr.open_zarr(store, group="geopotential", consolidated=True, zarr_format=3)
    try:
        assert 700 not in temp["pressure_level"].values.tolist()
        assert 850 not in geop["pressure_level"].values.tolist()
        assert np.datetime64("2024-01-03T00:00:00") not in temp["time"].values
        assert "relative_humidity" not in list(temp.data_vars) + list(geop.data_vars)
    finally:
        temp.close(); geop.close()


def test_rerun_is_exact_and_atomic_replaces_output(tmp_path):
    cache, _ = _make_fixture(tmp_path)
    out = tmp_path / "out"
    r1 = pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    first_meta = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    r2 = pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    second_meta = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert r1 == r2
    assert first_meta == second_meta
    assert not any(".tmp" in p for p in second_meta)


def test_full_chw_read_path_using_artifact_selectors(tmp_path):
    result, store, source = _run(tmp_path)
    arrays = {}
    for c in result["dataset_artifact"]["channels"]:
        group = c["array_path"].split("/")[0]
        if group not in arrays:
            arrays[group] = xr.open_zarr(store, group=group, consolidated=True, zarr_format=3)
    try:
        planes = []
        for c in result["dataset_artifact"]["channels"]:
            group, var = c["array_path"].split("/")
            level = int(c["selectors"]["pressure_level"])
            plane = arrays[group][var].sel(time=np.datetime64("2024-01-01T00:00:00"), pressure_level=level).values
            assert plane.shape == (3, 3)
            planes.append(plane)
        chw = np.stack(planes, axis=0)
        assert chw.shape == (3, 3, 3)
        expected = np.stack([
            source["temperature"].sel(time=np.datetime64("2024-01-01T00:00:00"), pressure_level=500).values,
            source["temperature"].sel(time=np.datetime64("2024-01-01T00:00:00"), pressure_level=850).values,
            source["geopotential"].sel(time=np.datetime64("2024-01-01T00:00:00"), pressure_level=500).values,
        ])
        np.testing.assert_array_equal(chw, expected)
    finally:
        for ds in arrays.values():
            ds.close()
