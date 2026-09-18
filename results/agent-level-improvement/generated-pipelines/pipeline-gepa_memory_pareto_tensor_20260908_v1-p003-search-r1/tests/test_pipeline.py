from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

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
            "variable": ["temperature", "geopotential"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Temperature"},
                "geopotential": {"label": "Geopotential", "units": "m2 s-2", "description": "Geopotential"},
            }
        },
    }


def _contract(end_date="2024-01-02"):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": end_date, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _write_fixture(cache_dir: Path):
    times = pd.date_range("2024-01-01", periods=8, freq="6h")
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 89.75, 89.5], dtype=np.float32)
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = base + np.float32(250.0)
    geop = base + np.float32(5000.0)
    temp[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "t": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "long_name": "Temperature"}),
            "z": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2", "long_name": "Geopotential"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
        attrs={"source": "synthetic offline fixture"},
    )
    raw = cache_dir / "era5_fixture.nc"
    ds.to_netcdf(raw, engine="netcdf4")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {"schema_version": "source_fixture_manifest.v1", "entries": [{"entry_id": "synthetic-era5", "relative_path": raw.name, "source": "offline-test", "size_bytes": raw.stat().st_size, "sha256": digest}]}
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def _hash_tree(path: Path):
    out = []
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        out.append((str(p.relative_to(path)), hashlib.sha256(p.read_bytes()).hexdigest()))
    return out


def test_publication_semantics_chunks_codec_and_tensor_read(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    source = _write_fixture(cache)

    result = pipeline_impl.run_pipeline({"envelope": {"contract": _contract()}}, _inventory(), str(cache), str(out))
    assert result["cache"]["hits"] == 1
    assert result["cache"]["acquired"] == 0
    assert result["dataset_artifact"]["store_path"] == "dataset.zarr"
    channels = result["dataset_artifact"]["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]

    store = out / "dataset.zarr"
    reopened = xr.open_zarr(store, consolidated=True).load()
    assert reopened.sizes["time"] == 8  # inclusive date-only end includes the whole final UTC day with selected times
    assert reopened.sizes["latitude"] == 3
    assert reopened.sizes["longitude"] == 4

    for ch in channels:
        arr = reopened[ch["array_path"]]
        selector_dim = ch["selector_coordinate_paths"]["pressure_level"]
        assert selector_dim in arr.dims
        assert arr.sizes[selector_dim] == 1
        assert selector_dim in reopened.coords
        assert str(int(reopened[selector_dim].values[0])) == ch["selectors"]["pressure_level"]
        assert arr.dims == ("time", selector_dim, "latitude", "longitude")

    np.testing.assert_array_equal(reopened["temperature_pressure_level_500"].values[:, 0], source["t"].sel(pressure_level=500).values)
    np.testing.assert_array_equal(reopened["temperature_pressure_level_850"].values[:, 0], source["t"].sel(pressure_level=850).values)
    np.testing.assert_array_equal(reopened["geopotential_pressure_level_500"].values[:, 0], source["z"].sel(pressure_level=500).values)
    np.testing.assert_array_equal(reopened["time"].values, source["time"].values)
    np.testing.assert_array_equal(reopened["latitude"].values, source["latitude"].values)
    np.testing.assert_array_equal(reopened["longitude"].values, source["longitude"].values)
    assert np.isnan(reopened["temperature_pressure_level_500"].values[1, 0, 1, 2])
    assert reopened["temperature_pressure_level_500"].attrs["canonical_field_id"] == 'temperature[pressure_level="500"]'

    group = zarr.open_group(store, mode="r")
    for ch in channels:
        za = group[ch["array_path"]]
        assert tuple(za.chunks) == (1, 1, 3, 4)
        codecs_text = json.dumps(za.metadata.to_dict(), sort_keys=True).lower()
        assert "blosc" in codecs_text
        assert "lz4" in codecs_text

    # Lightweight C,H,W-equivalent read-path sanity: one complete H,W plane per declared channel at a sample time.
    planes = []
    for ch in channels:
        selector_dim = ch["selector_coordinate_paths"]["pressure_level"]
        plane = reopened[ch["array_path"]].isel(time=3, **{selector_dim: 0}).values
        assert plane.shape == (3, 4)
        planes.append(plane)
    chw = np.stack(planes, axis=0)
    assert chw.shape == (3, 3, 4)
    np.testing.assert_array_equal(chw[0], source["t"].sel(time=source.time.values[3], pressure_level=500).values)


def test_exact_rerun_and_read_only_fixture(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    _write_fixture(cache)
    pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))
    first = _hash_tree(out / "dataset.zarr")
    pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))
    second = _hash_tree(out / "dataset.zarr")
    assert first == second
    assert sorted(p.name for p in cache.iterdir()) == ["era5_fixture.nc", "source_fixture_manifest.json"]


def test_contract_validation_rejects_invalid_selector_and_missing_fixture(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    with pytest.raises(RuntimeError, match="source_fixture_manifest"):
        pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))

    _write_fixture(cache)
    bad = _contract()
    bad["fields"][0]["selectors"][0]["value"] = "700"
    with pytest.raises(ValueError, match="pressure_level"):
        pipeline_impl.run_pipeline({"contract": bad}, _inventory(), str(cache), str(out))


def test_source_fixture_manifest_hash_is_enforced(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    _write_fixture(cache)
    manifest_path = cache / "source_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        pipeline_impl.run_pipeline({"contract": _contract()}, _inventory(), str(cache), str(out))
