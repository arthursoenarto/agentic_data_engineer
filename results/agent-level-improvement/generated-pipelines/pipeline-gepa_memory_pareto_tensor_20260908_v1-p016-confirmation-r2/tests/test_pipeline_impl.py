from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

PIPELINE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", PIPELINE_DIR / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "provider": "ECMWF",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
    }


def _contract(days=2):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": f"2024-01-{days:02d}", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _write_fixture(cache_dir: Path):
    cache_dir.mkdir(parents=True)
    times = pd.date_range("2024-01-01", periods=8, freq="6h")
    levels = np.array([500, 850, 1000], dtype=np.int32)
    lat = np.array([1.0, 0.0], dtype=np.float32)
    lon = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = 250.0 + base / 2.0
    geo = 5000.0 + base * 2.0
    temp[1, 0, 0, 1] = np.nan
    ds = xr.Dataset(
        {
            "t": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "z": (("time", "pressure_level", "latitude", "longitude"), geo, {"units": "m**2 s**-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
    )
    raw = cache_dir / "era5_fixture.nc"
    enc = {
        "t": {"dtype": "int16", "scale_factor": 0.5, "add_offset": 250.0, "_FillValue": np.int16(-32768)},
        "z": {"dtype": "int16", "scale_factor": 2.0, "add_offset": 5000.0, "_FillValue": np.int16(-32768)},
    }
    ds.to_netcdf(raw, engine="scipy", encoding=enc)
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "era5-test", "relative_path": raw.name, "source": "unit-test", "size_bytes": raw.stat().st_size, "sha256": sha}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw, xr.open_dataset(raw, decode_cf=True, mask_and_scale=True).load()


def _digest_tree(path: Path):
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _run(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    raw, decoded = _write_fixture(cache)
    result = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))
    return cache, out, raw, decoded, result


def test_fixture_reuse_publication_readback_and_semantics(tmp_path):
    cache, out, raw, decoded, result = _run(tmp_path)
    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired"] == 0
    assert result["dataset_artifact"]["store_path"] == "dataset.zarr"

    store = out / "dataset.zarr"
    assert (store / "zarr.json").exists()
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta.get("zarr_format") == 3
    assert "consolidated_metadata" in root_meta

    ds = xr.open_zarr(store, consolidated=True)
    assert set(ds.data_vars) == {"temperature", "geopotential"}
    assert ds["temperature"].dims == ("time", "temperature_pressure_level", "latitude", "longitude")
    assert ds["geopotential"].dims == ("time", "geopotential_pressure_level", "latitude", "longitude")
    np.testing.assert_array_equal(ds["temperature_pressure_level"].values, np.array([500, 850], dtype=np.int32))
    np.testing.assert_array_equal(ds["geopotential_pressure_level"].values, np.array([500], dtype=np.int32))
    assert np.isnan(ds["temperature"].values[1, 0, 0, 1])

    expected_t = decoded["t"].sel(time=ds.time.values, pressure_level=[500, 850]).values
    expected_z = decoded["z"].sel(time=ds.time.values, pressure_level=[500]).values
    np.testing.assert_array_equal(ds["temperature"].values, expected_t)
    np.testing.assert_array_equal(ds["geopotential"].values, expected_z)
    np.testing.assert_array_equal(ds["latitude"].values, decoded["latitude"].values)
    np.testing.assert_array_equal(ds["longitude"].values, decoded["longitude"].values)


def test_grouped_channels_and_full_field_tensor_read_path(tmp_path):
    cache, out, raw, decoded, result = _run(tmp_path)
    channels = result["dataset_artifact"]["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert channels[0]["array_path"] == channels[1]["array_path"] == "temperature"
    assert channels[0]["selectors"] != channels[1]["selectors"]
    assert channels[0]["selector_coordinate_paths"] == {"pressure_level": "temperature_pressure_level"}
    assert channels[2]["selector_coordinate_paths"] == {"pressure_level": "geopotential_pressure_level"}

    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    planes = []
    for ch in channels:
        arr = ds[ch["array_path"]]
        for dim, val in ch["selectors"].items():
            coord_name = ch["selector_coordinate_paths"][dim]
            coord_vals = ds[coord_name].values
            idx = int(np.flatnonzero(coord_vals.astype(str) == val)[0])
            arr = arr.isel({coord_name: idx})
        planes.append(arr.isel(time=0).values)
    tensor = np.stack(planes, axis=0)
    assert tensor.shape == (3, 2, 3)
    np.testing.assert_array_equal(tensor[0], decoded["t"].sel(time=ds.time.values[0], pressure_level=500).values)
    np.testing.assert_array_equal(tensor[1], decoded["t"].sel(time=ds.time.values[0], pressure_level=850).values)
    np.testing.assert_array_equal(tensor[2], decoded["z"].sel(time=ds.time.values[0], pressure_level=500).values)


def test_chunks_and_no_compression_or_declared_lz4_fallback(tmp_path):
    cache, out, raw, decoded, result = _run(tmp_path)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    full_lat = ds.sizes["latitude"]
    full_lon = ds.sizes["longitude"]
    for var in ["temperature", "geopotential"]:
        meta = json.loads((out / "dataset.zarr" / var / "zarr.json").read_text())
        assert tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]) == (1, 1, full_lat, full_lon)
        codec_blob = json.dumps(meta.get("codecs", [])).lower()
        if any("fallback" in w.lower() or "lz4" in w.lower() for w in result["warnings"]):
            assert "blosc" in codec_blob and "lz4" in codec_blob
            assert "clevel" in codec_blob and "1" in codec_blob
        else:
            assert "blosc" not in codec_blob and "zstd" not in codec_blob and "gzip" not in codec_blob


def test_no_unrequested_payload_or_sidecar_store(tmp_path):
    cache, out, raw, decoded, result = _run(tmp_path)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    assert set(ds.data_vars) == {"temperature", "geopotential"}
    assert set(ds["temperature_pressure_level"].values.tolist()) == {500, 850}
    assert set(ds["geopotential_pressure_level"].values.tolist()) == {500}
    assert 1000 not in ds["temperature_pressure_level"].values.tolist()
    assert ds.sizes["time"] == 8
    assert not any(p.name.endswith("tensor") or "sidecar" in p.name for p in out.rglob("*"))


def test_rerun_is_exact_and_does_not_modify_cache(tmp_path):
    cache, out, raw, decoded, result1 = _run(tmp_path)
    cache_digest_before = _digest_tree(cache)
    out_digest_1 = _digest_tree(out)
    result2 = pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))
    assert _digest_tree(cache) == cache_digest_before
    assert _digest_tree(out) == out_digest_1
    assert result1["dataset_artifact"] == result2["dataset_artifact"]
    assert result2["cache"]["hits"] == 1 and result2["cache"]["acquired"] == 0


def test_contract_validation_rejects_invalid_selector_and_bad_manifest(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    _write_fixture(cache)
    bad = _contract()
    bad["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({"contract": bad}, _inventory(), str(cache), str(out))

    bad_manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    bad_manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(bad_manifest))
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))


def test_inclusive_end_date_applies_complete_final_day_then_selected_times(tmp_path):
    cache, out, raw, decoded, result = _run(tmp_path)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    got = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    assert got[0] == pd.Timestamp("2024-01-01T00:00:00")
    assert got[-1] == pd.Timestamp("2024-01-02T18:00:00")
    assert len(got) == 8
