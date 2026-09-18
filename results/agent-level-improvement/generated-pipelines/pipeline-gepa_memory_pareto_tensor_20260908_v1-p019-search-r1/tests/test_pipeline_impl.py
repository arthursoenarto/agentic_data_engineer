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


def inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "options": {
            "product_type": ["reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01", "02", "03", "04", "05", "06", "07"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": ["500", "850", "700"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "option_metadata": {
            "variable": {
                "temperature": {"units": "K", "description": "Temperature"},
                "geopotential": {"units": "m2 s-2", "description": "Geopotential"},
                "relative_humidity": {"units": "%", "description": "Relative humidity"},
            }
        },
    }


def seed_contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
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


def other_contract():
    c = seed_contract()
    c = json.loads(json.dumps(c))
    c["fields"] = [
        {"name": "relative_humidity", "display_name": "Relative humidity", "selectors": [{"dimension": "pressure_level", "value": "700", "unit": "hPa", "label": None}]}
    ]
    c["scope"]["date_range"] = {"start_date": "2024-01-02", "end_date": "2024-01-03", "inclusive": True}
    c["scope"]["time"] = {"selected_times": ["06:00", "18:00"], "timestep": "12 hours", "timezone": "UTC"}
    c["scope"]["geography"] = {"area": "custom", "cds_area": [1.0, 10.0, -1.0, 12.0], "cds_area_order": ["north", "west", "south", "east"]}
    return c


def make_fixture(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    times = np.array([np.datetime64(f"2024-01-{d:02d}T{h:02d}:00:00") for d in range(1, 8) for h in (0, 6, 12, 18)], dtype="datetime64[ns]")
    levels = np.array([500, 700, 850], dtype="int32")
    lat = np.array([1.0, 0.0, -1.0], dtype="float32")
    lon = np.array([10.0, 11.0, 12.0, 13.0], dtype="float32")
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    temp = base + 273.15
    geop = base + 1000.0
    rh = base / 10.0
    temp[0, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "quality_flag_meaning": "synthetic"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2"}),
            "relative_humidity": (("time", "pressure_level", "latitude", "longitude"), rh, {"units": "%"}),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
    )
    raw = cache / "era5_fixture.nc"
    ds.to_netcdf(raw)
    data = raw.read_bytes()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "synthetic-era5", "relative_path": raw.name, "source": "offline-test", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache, ds


def test_seed_publication_per_channel_chunks_uncompressed_and_readback(tmp_path: Path):
    cache, source = make_fixture(tmp_path)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline({"contract": seed_contract()}, inventory(), str(cache), str(out))
    artifact = result["dataset_artifact"]
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3
    assert [c["field_id"] for c in artifact["channels"]] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    store = out / artifact["store_path"]
    assert (store / "zarr.json").exists()
    root_meta = json.loads((store / "zarr.json").read_text())
    assert "consolidated_metadata" in root_meta
    ds = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        assert set(ds.data_vars) == {c["array_path"] for c in artifact["channels"]}
        assert "pressure_level" not in ds.dims
        assert set(ds["time"].values) == set(source["time"].values)
        for ch in artifact["channels"]:
            arr = ds[ch["array_path"]]
            sel_coord = ch["selector_coordinate_paths"]["pressure_level"]
            assert arr.dims == ("time", sel_coord, "latitude", "longitude")
            assert arr.sizes[sel_coord] == 1
            meta = json.loads((store / ch["array_path"] / "zarr.json").read_text())
            assert tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]) == (1, 1, 3, 4)
            codecs = json.dumps(meta.get("codecs", [])).lower()
            if not result["warnings"]:
                assert "zstd" not in codecs and "blosc" not in codecs and "lz4" not in codecs
            else:
                assert "lz4" in codecs and "clevel" in codecs
            src_var = ch["array_path"].split("_pressure_level_")[0]
            level = int(ch["selectors"]["pressure_level"])
            expected = source[src_var].sel(pressure_level=[level]).values
            np.testing.assert_array_equal(arr.values, expected)
            np.testing.assert_array_equal(ds[sel_coord].values, np.array([level], dtype=source["pressure_level"].dtype))
        # C,H,W-equivalent read across declared paths.
        chw = np.stack([ds[c["array_path"]].isel(time=0).isel({c["selector_coordinate_paths"]["pressure_level"]: 0}).values for c in artifact["channels"]], axis=0)
        assert chw.shape == (3, 3, 4)
        assert np.isnan(chw[0, 1, 2])
    finally:
        ds.close()


def test_no_unrequested_levels_variables_times_or_sidecar_store(tmp_path: Path):
    cache, _source = make_fixture(tmp_path)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline(seed_contract(), inventory(), str(cache), str(out))
    store = out / result["dataset_artifact"]["store_path"]
    ds = xr.open_zarr(store, consolidated=True, zarr_format=3)
    try:
        assert "relative_humidity" not in " ".join(ds.data_vars)
        for ch in result["dataset_artifact"]["channels"]:
            assert list(ds[ch["selector_coordinate_paths"]["pressure_level"]].values) in ([500], [850])
        assert not any(p.name.endswith("tensor.zarr") for p in out.iterdir())
    finally:
        ds.close()


def test_exact_rerun_result_and_store_values(tmp_path: Path):
    cache, _ = make_fixture(tmp_path)
    out = tmp_path / "out"
    r1 = pipeline_impl.run_pipeline(seed_contract(), inventory(), str(cache), str(out))
    ds1 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    r2 = pipeline_impl.run_pipeline(seed_contract(), inventory(), str(cache), str(out))
    ds2 = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3).load()
    assert r1 == r2
    xr.testing.assert_identical(ds1, ds2)


def test_other_valid_lock_runtime_filtering_and_names(tmp_path: Path):
    cache, source = make_fixture(tmp_path)
    out = tmp_path / "out_other"
    result = pipeline_impl.run_pipeline({"selected_contract": other_contract()}, inventory(), str(cache), str(out))
    assert len(result["dataset_artifact"]["channels"]) == 1
    ch = result["dataset_artifact"]["channels"][0]
    assert ch["field_id"] == 'relative_humidity[pressure_level="700"]'
    assert ch["array_path"] == "relative_humidity_pressure_level_700"
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True, zarr_format=3)
    try:
        assert ds.sizes["time"] == 4
        assert ds.sizes["latitude"] == 3
        assert ds.sizes["longitude"] == 3
        expected_times = np.array([np.datetime64("2024-01-02T06:00"), np.datetime64("2024-01-02T18:00"), np.datetime64("2024-01-03T06:00"), np.datetime64("2024-01-03T18:00")], dtype="datetime64[ns]")
        np.testing.assert_array_equal(ds["time"].values, expected_times)
        expected = source["relative_humidity"].sel(time=expected_times, pressure_level=[700], longitude=[10.0, 11.0, 12.0]).values
        np.testing.assert_array_equal(ds[ch["array_path"]].values, expected)
    finally:
        ds.close()


def test_contract_validation_rejects_bad_selector_and_bad_fixture(tmp_path: Path):
    cache, _ = make_fixture(tmp_path)
    bad = seed_contract()
    bad = json.loads(json.dumps(bad))
    bad["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(bad, inventory(), str(cache), str(tmp_path / "out"))
    no_manifest = tmp_path / "empty_cache"
    no_manifest.mkdir()
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(seed_contract(), inventory(), str(no_manifest), str(tmp_path / "out2"))
