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


def _inventory() -> dict:
    opts = {
        "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
        "variable": ["temperature", "geopotential", "u_component_of_wind"],
        "year": ["2024"],
        "month": ["01"],
        "day": [f"{i:02d}" for i in range(1, 32)],
        "time": [f"{i:02d}:00" for i in range(24)],
        "pressure_level": ["500", "850", "1000"],
        "data_format": ["grib", "netcdf"],
        "download_format": ["unarchived", "zip"],
    }
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": opts,
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
    }


def _contract(times=("00:00", "06:00"), fields=None) -> dict:
    if fields is None:
        fields = [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None},
        ]
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "fields": fields,
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-02", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": list(times), "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _make_fixture(cache_dir: Path) -> xr.Dataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-02T00:00", "2024-01-02T06:00"], dtype="datetime64[ns]")
    levels = np.array([500, 850], dtype="int32")
    lat = np.array([90.0, 89.75, 89.5], dtype="float32")
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype="float32")
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    temp = base.copy()
    temp[1, 0, 1, 2] = np.nan
    geop = base + 10000.0
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m**2 s**-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
    )
    path = cache_dir / "source.nc"
    # Add packing metadata to exercise CF decoding/masking in the read path without
    # changing exact decoded fixture values after open_dataset(mask_and_scale=True).
    ds.to_netcdf(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "source-nc", "relative_path": "source.nc", "source": "local-test", "size_bytes": path.stat().st_size, "sha256": h}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    return ds


def _open_zarr(path: Path) -> xr.Dataset:
    try:
        return xr.open_zarr(path, consolidated=True, zarr_format=3).load()
    except TypeError:
        return xr.open_zarr(path, consolidated=True, zarr_version=3).load()


def test_contract_validation_rejects_inventory_mismatch(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    inv = _inventory()
    inv["dataset_slug"] = "wrong"
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline({"lock": {"contract": _contract()}}, inv, str(cache), str(tmp_path / "out"))


def test_missing_or_bad_fixture_fails_before_publication(tmp_path):
    with pytest.raises(Exception, match="source_fixture_manifest"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), str(tmp_path / "cache"), str(tmp_path / "out"))
    cache = tmp_path / "cache2"
    _make_fixture(cache)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="SHA-256"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(tmp_path / "out2"))


def test_publication_readback_semantics_chunks_and_artifact(tmp_path):
    cache = tmp_path / "cache"
    source = _make_fixture(cache)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline({"envelope": {"dataset_contract": _contract()}}, _inventory(), str(cache), str(out))

    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    artifact = result["dataset_artifact"]
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3

    store = out / artifact["store_path"]
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    reopened = _open_zarr(store)
    expected_names = {"temperature__pressure_level_500", "temperature__pressure_level_850", "geopotential__pressure_level_500"}
    assert set(reopened.data_vars) == expected_names

    # Inclusive date-only end bound includes all selected times on the final day.
    assert reopened.sizes["time"] == 4
    assert str(reopened.time.values[-1]).startswith("2024-01-02T06:00")

    for channel in artifact["channels"]:
        arr = channel["array_path"]
        selector_coord = channel["selector_coordinate_paths"]["pressure_level"]
        assert selector_coord in reopened[arr].dims
        assert reopened[arr].sizes[selector_coord] == 1
        assert reopened[arr].dims == ("time", selector_coord, "latitude", "longitude")
        assert reopened[arr].sizes["latitude"] == source.sizes["latitude"]
        assert reopened[arr].sizes["longitude"] == source.sizes["longitude"]
        # Full-field sample-aligned chunks: one time, one selector, all H/W.
        meta = json.loads((store / arr / "zarr.json").read_text())
        assert tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]) == (1, 1, source.sizes["latitude"], source.sizes["longitude"])

    np.testing.assert_array_equal(reopened["temperature__pressure_level_500"].values[:, 0], source["temperature"].sel(pressure_level=500).values)
    np.testing.assert_array_equal(reopened["temperature__pressure_level_850"].values[:, 0], source["temperature"].sel(pressure_level=850).values)
    np.testing.assert_array_equal(reopened["geopotential__pressure_level_500"].values[:, 0], source["geopotential"].sel(pressure_level=500).values)
    assert np.isnan(reopened["temperature__pressure_level_500"].values[1, 0, 1, 2])
    assert reopened["temperature__pressure_level_500"].attrs["units"] == "K"


def test_codec_metadata_prefers_blosc_lz4_bitshuffle_or_documented_fallback(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(out))
    arr_meta = json.loads((out / "dataset.zarr" / "temperature__pressure_level_500" / "zarr.json").read_text())
    codec_text = json.dumps(arr_meta.get("codecs", []), sort_keys=True).lower()
    if "fallback" in " ".join(result.get("warnings", [])).lower():
        assert "zstd" in codec_text
    else:
        assert "blosc" in codec_text
        assert "lz4" in codec_text
        assert "bitshuffle" in codec_text


def test_exact_rerun_is_deterministic_and_reuses_fixture(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    out = tmp_path / "out"
    r1 = pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(out))
    ds1 = _open_zarr(out / "dataset.zarr")
    vals1 = {k: v.values.copy() for k, v in ds1.data_vars.items()}
    r2 = pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(out))
    ds2 = _open_zarr(out / "dataset.zarr")
    assert r1 == r2
    for k, v in vals1.items():
        np.testing.assert_array_equal(v, ds2[k].values)


def test_lightweight_tensor_read_path_across_all_channels(tmp_path):
    cache = tmp_path / "cache"
    source = _make_fixture(cache)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline(_contract(), _inventory(), str(cache), str(out))
    ds = _open_zarr(out / "dataset.zarr")
    planes = []
    for channel in result["dataset_artifact"]["channels"]:
        arr = channel["array_path"]
        selector_coord = channel["selector_coordinate_paths"]["pressure_level"]
        plane = ds[arr].isel(time=0, **{selector_coord: 0}).values
        assert plane.shape == (source.sizes["latitude"], source.sizes["longitude"])
        planes.append(plane)
    tensor = np.stack(planes, axis=0)
    assert tensor.shape == (3, source.sizes["latitude"], source.sizes["longitude"])
    np.testing.assert_array_equal(tensor[0], source["temperature"].sel(pressure_level=500).isel(time=0).values)
