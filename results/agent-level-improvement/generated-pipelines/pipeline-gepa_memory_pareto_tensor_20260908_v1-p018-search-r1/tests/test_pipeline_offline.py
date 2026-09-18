import hashlib
import importlib.util
import json
import shutil
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
            "product_type": ["reanalysis"],
            "variable": ["temperature", "geopotential"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01", "02", "03", "04", "05", "06", "07"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": ["500", "850"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_units": {"pressure_level": "hPa"},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Air temperature"},
                "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>", "description": "Geopotential"},
            }
        },
    }


def _contract():
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "advanced_options": {"dataset_id": "reanalysis-era5-pressure-levels", "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "fields": [
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa"}]},
            {"name": "temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa"}]},
            {"name": "geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa"}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": "2024-01-07", "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "time": {"selected_times": ["00:00", "06:00", "12:00", "18:00"], "timestep": "6 hours", "timezone": "UTC"},
            "product_type": "reanalysis",
        },
    }


def _make_fixture(cache_dir: Path):
    times = pd.date_range("2024-01-01", "2024-01-07 18:00", freq="6h")
    levels = np.array([500, 850], dtype="int32")
    lat = np.array([1.0, 0.0, -1.0], dtype="float32")
    lon = np.array([-2.0, -1.0, 0.0, 1.0], dtype="float32")
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    temperature = base + 250.0
    geopotential = base + 50000.0
    temperature[2, 1, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temperature, {"units": "K", "quality_flag_meaning": "synthetic fixture"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geopotential, {"units": "m2 s-2"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    path = cache_dir / "era5_fixture.nc"
    ds.to_netcdf(path, engine="netcdf4", encoding={
        "temperature": {"zlib": True},
        "geopotential": {"zlib": True},
    })
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": "source_fixture_manifest.v1", "entries": [{"entry_id": "fixture", "relative_path": "era5_fixture.nc", "source": "unit-test", "size_bytes": path.stat().st_size, "sha256": sha}]}
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def _run(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    out.mkdir()
    source = _make_fixture(cache)
    result = pipeline_impl.run_pipeline({"selected_contract": _contract()}, _inventory(), str(cache), str(out))
    return source, result, out


def test_fixture_reuse_contract_validation_and_no_network(tmp_path):
    source, result, out = _run(tmp_path)
    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired"] == 0
    assert not result["cache"]["acquired_keys"]
    assert (out / "dataset.zarr" / "zarr.json").exists()
    bad = _contract()
    bad["fields"][0]["selectors"][0]["value"] = "925"
    cache = tmp_path / "badcache"
    cache.mkdir()
    _make_fixture(cache)
    with pytest.raises(Exception):
        pipeline_impl.run_pipeline(bad, _inventory(), str(cache), str(tmp_path / "badout"))


def test_publication_readback_grouped_layout_channels_and_exact_values(tmp_path):
    source, result, out = _run(tmp_path)
    artifact = result["dataset_artifact"]
    assert artifact["store_path"] == "dataset.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    channels = artifact["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert channels[0]["array_path"] == channels[1]["array_path"] == "temperature"
    assert channels[0]["selectors"] != channels[1]["selectors"]
    assert channels[2]["array_path"] == "geopotential"
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True).load()
    assert set(ds.data_vars) == {"temperature", "geopotential"}
    assert "pressure_level_temperature" in ds.dims
    assert "pressure_level_geopotential" in ds.dims
    assert ds.sizes["time"] == 28
    assert ds.sizes["pressure_level_temperature"] == 2
    assert ds.sizes["pressure_level_geopotential"] == 1
    assert np.array_equal(ds["pressure_level_temperature"].values.astype(int), np.array([500, 850]))
    assert np.array_equal(ds["pressure_level_geopotential"].values.astype(int), np.array([500]))
    assert np.array_equal(ds["latitude"].values, source["latitude"].values)
    assert np.array_equal(ds["longitude"].values, source["longitude"].values)
    assert np.array_equal(ds["temperature"].values, source["temperature"].values, equal_nan=True)
    assert np.array_equal(ds["geopotential"].values, source["geopotential"].sel(pressure_level=[500]).values, equal_nan=True)
    assert np.isnan(ds["temperature"].values[2, 1, 0, 0])
    assert ds["temperature"].attrs["quality_flag_meaning"] == "synthetic fixture"


def test_zarr_v3_consolidated_chunks_and_lossless_codec_metadata(tmp_path):
    source, result, out = _run(tmp_path)
    root_meta = json.loads((out / "dataset.zarr" / "zarr.json").read_text())
    assert root_meta["zarr_format"] == 3
    assert "consolidated_metadata" in root_meta
    for name in ["temperature", "geopotential"]:
        meta = json.loads((out / "dataset.zarr" / name / "zarr.json").read_text())
        chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape") or meta.get("chunks")
        assert chunk_shape == [1, 1, 3, 4]
        text = json.dumps(meta).lower()
        assert "zstd" in text
        assert "bitshuffle" in text
        assert "scale_factor" not in text
        assert "add_offset" not in text
        assert "quant" not in text
        assert "float16" not in text


def test_read_path_full_chw_samples_and_no_unrequested_content_or_sidecars(tmp_path):
    source, result, out = _run(tmp_path)
    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    planes = []
    for ch in result["dataset_artifact"]["channels"]:
        arr = ds[ch["array_path"]]
        coord = ch["selector_coordinate_paths"]["pressure_level"]
        level = int(ch["selectors"]["pressure_level"])
        idx = int(np.where(ds[coord].values.astype(int) == level)[0][0])
        selector_dim = coord
        plane = arr.isel(time=0, **{selector_dim: idx}).values
        planes.append(plane)
    chw = np.stack(planes, axis=0)
    assert chw.shape == (3, 3, 4)
    expected = np.stack([
        source["temperature"].sel(time=source.time.values[0], pressure_level=500).values,
        source["temperature"].sel(time=source.time.values[0], pressure_level=850).values,
        source["geopotential"].sel(time=source.time.values[0], pressure_level=500).values,
    ])
    assert np.array_equal(chw, expected, equal_nan=True)
    names = {p.name for p in (out / "dataset.zarr").iterdir()}
    assert "925" not in json.dumps(names)
    assert "sidecar" not in json.dumps(names).lower()
    assert "tensor" not in json.dumps(names).lower()
    assert set(ds.data_vars) == {"temperature", "geopotential"}


def test_exact_rerun_and_allowed_source_tree_files(tmp_path):
    source, result1, out = _run(tmp_path)
    digest1 = hashlib.sha256(json.dumps(result1, sort_keys=True).encode()).hexdigest()
    cache = tmp_path / "cache"
    result2 = pipeline_impl.run_pipeline({"selected_contract": _contract()}, _inventory(), str(cache), str(out))
    digest2 = hashlib.sha256(json.dumps(result2, sort_keys=True).encode()).hexdigest()
    assert digest1 == digest2
    forbidden = {"run_pipeline.py", "pipeline_contract.json", "manifest.json", "pipeline_run.json"}
    produced = {p.name for p in ROOT.rglob("*") if p.is_file()}
    assert not (produced & forbidden)
    assert ".pytest_cache" not in {p.name for p in ROOT.rglob("*")}
    assert "benchmark" not in " ".join(str(p) for p in ROOT.rglob("*")).lower()
