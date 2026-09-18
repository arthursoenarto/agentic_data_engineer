from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

PIPELINE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_impl", PIPELINE_DIR / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "provider": "ECMWF",
        "catalogue_metadata": {"extent": {"temporal": {"interval": [["1940-01-01T00:00:00+00:00", "2026-06-30T00:00:00+00:00"]]}}},
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "options": {
            "product_type": ["ensemble_mean", "ensemble_members", "ensemble_spread", "reanalysis"],
            "variable": ["temperature", "geopotential", "relative_humidity"],
            "year": [str(y) for y in range(1940, 2027)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["zip", "unarchived"],
        },
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "air temperature"},
                "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>", "description": "geopotential"},
            }
        },
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "title": "test",
        "source_url": "https://example.invalid",
        "provider": "Copernicus Climate Data Store",
        "dataset_family": "reanalysis",
        "access_methods": [],
        "credential_requirements": [],
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}], "units": None, "description": None},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}], "units": None, "description": None},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-02", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _write_fixture(cache_dir: Path):
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00", "2024-01-02T00:00", "2024-01-02T06:00"], dtype="datetime64[ns]")
    levels = np.array([500, 850], dtype="int32")
    lat = np.array([90.0, 89.75, 89.5], dtype="float64")
    lon = np.array([-180.0, -179.75, -179.5, -179.25], dtype="float64")
    base = np.arange(times.size * levels.size * lat.size * lon.size, dtype="float32").reshape(times.size, levels.size, lat.size, lon.size)
    temp = base + np.float32(250.0)
    geop = base + np.float32(50000.0)
    temp[1, 0, 1, 2] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K", "quality_flag_meaning": "synthetic_fixture"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2"}),
        },
        coords={"time": times, "pressure_level": ("pressure_level", levels, {"units": "hPa"}), "latitude": lat, "longitude": lon},
        attrs={"source_quality": "offline exact fixture"},
    )
    raw = cache_dir / "era5_fixture.nc"
    ds.to_netcdf(raw)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest = {"schema_version": "source_fixture_manifest.v1", "entries": [{"entry_id": "synthetic-era5", "relative_path": raw.name, "source": "pytest", "size_bytes": raw.stat().st_size, "sha256": digest}]}
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def test_contract_validation_rejects_bad_selector(tmp_path):
    c = _contract()
    c["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(ValueError):
        pipeline_impl.run_pipeline({"dataset_contract": c}, _inventory(), tmp_path / "cache", tmp_path / "out")


def test_complete_fixture_required_before_any_provider(tmp_path):
    (tmp_path / "cache").mkdir()
    with pytest.raises(FileNotFoundError):
        pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), tmp_path / "cache", tmp_path / "out")


def test_fixture_verification_rejects_hash_mismatch(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_fixture(cache)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), cache, tmp_path / "out")


def test_publication_grouped_arrays_exact_values_chunks_and_uncompressed(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    source = _write_fixture(cache)
    result = pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), cache, out)

    artifact = result["dataset_artifact"]
    assert artifact["schema_version"] == "dataset_artifact_layout.v1"
    assert artifact["storage_format"] == "zarr"
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3
    assert [c["array_path"] for c in artifact["channels"]] == ["temperature", "temperature", "geopotential"]
    assert artifact["channels"][0]["selector_coordinate_paths"] == {"pressure_level": "temperature_pressure_level"}
    assert artifact["channels"][2]["selector_coordinate_paths"] == {"pressure_level": "geopotential_pressure_level"}

    store = out / "dataset.zarr"
    root_meta = json.loads((store / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta

    reopened = xr.open_zarr(store, consolidated=True, zarr_format=3).load()
    assert set(reopened.data_vars) == {"temperature", "geopotential"}
    assert reopened["temperature"].dims == ("time", "temperature_pressure_level", "latitude", "longitude")
    assert reopened["geopotential"].dims == ("time", "geopotential_pressure_level", "latitude", "longitude")
    assert reopened.sizes["temperature_pressure_level"] == 2
    assert reopened.sizes["geopotential_pressure_level"] == 1
    np.testing.assert_array_equal(reopened["temperature_pressure_level"].values, np.array([500, 850], dtype="int32"))
    np.testing.assert_array_equal(reopened["geopotential_pressure_level"].values, np.array([500], dtype="int32"))
    np.testing.assert_array_equal(reopened["latitude"].values, source["latitude"].values)
    np.testing.assert_array_equal(reopened["longitude"].values, source["longitude"].values)

    requested_times = np.array(["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-02T00:00", "2024-01-02T06:00"], dtype="datetime64[ns]")
    expected_temp = source["temperature"].sel(time=requested_times, pressure_level=[500, 850]).values
    expected_geop = source["geopotential"].sel(time=requested_times, pressure_level=[500]).values
    np.testing.assert_array_equal(reopened["temperature"].values, expected_temp)
    np.testing.assert_array_equal(reopened["geopotential"].values, expected_geop)
    assert np.isnan(reopened["temperature"].values[1, 0, 1, 2])
    assert reopened["temperature"].attrs["units"] == "K"
    assert reopened["temperature"].attrs["quality_flag_meaning"] == "synthetic_fixture"

    t_meta = json.loads((store / "temperature" / "zarr.json").read_text())
    z_meta = json.loads((store / "geopotential" / "zarr.json").read_text())
    assert t_meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 1, 3, 4]
    assert z_meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 1, 3, 4]
    codecs_text = json.dumps(t_meta.get("codecs", [])).lower()
    if "blosc" in codecs_text:
        assert "lz4" in codecs_text and "clevel" in codecs_text
        assert "zstd" not in codecs_text
    else:
        assert "zstd" not in codecs_text and "gzip" not in codecs_text and "lz4" not in codecs_text

    # Lightweight consumer read-path sanity: assemble one C,H,W-equivalent sample
    # from declared channel paths and selector values.
    planes = []
    for ch in artifact["channels"]:
        da = reopened[ch["array_path"]]
        for dim, value in ch["selectors"].items():
            coord = ch["selector_coordinate_paths"][dim]
            da = da.sel({coord: int(value)})
        planes.append(da.isel(time=0).values)
    tensor = np.stack(planes, axis=0)
    assert tensor.shape == (3, 3, 4)
    np.testing.assert_array_equal(tensor[0], expected_temp[0, 0])
    np.testing.assert_array_equal(tensor[1], expected_temp[0, 1])
    np.testing.assert_array_equal(tensor[2], expected_geop[0, 0])


def test_exact_rerun_behavior_and_secret_safe_cache_evidence(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _write_fixture(cache)
    r1 = pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), cache, out)
    first_meta = (out / "dataset.zarr" / "zarr.json").read_bytes()
    r2 = pipeline_impl.run_pipeline({"dataset_contract": _contract()}, _inventory(), cache, out)
    second_meta = (out / "dataset.zarr" / "zarr.json").read_bytes()
    assert r1["dataset_artifact"] == r2["dataset_artifact"]
    assert first_meta == second_meta
    assert r2["cache"]["hits"] == 1
    assert r2["cache"]["misses"] == 0
    evidence = json.dumps(r2["cache"])
    assert "secret" not in evidence.lower()
    assert "token" not in evidence.lower()
    assert "password" not in evidence.lower()
    assert str(cache) not in evidence
