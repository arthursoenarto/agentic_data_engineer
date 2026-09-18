from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pipeline_impl", ROOT / "pipeline_impl.py")
pipeline_impl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline_impl)


def _inventory():
    return {
        "schema_version": "dataset_inventory.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "dataset_id": "reanalysis-era5-pressure-levels",
        "options": {
            "product_type": ["reanalysis", "ensemble_mean"],
            "variable": ["temperature", "geopotential", "u_component_of_wind"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{i:02d}" for i in range(1, 32)],
            "time": [f"{i:02d}:00" for i in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": ["grib", "netcdf"],
            "download_format": ["unarchived", "zip"],
        },
        "defaults": {"area": [90, -180, -90, 180], "data_format": "grib", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "option_metadata": {
            "variable": {
                "temperature": {"label": "Temperature", "units": "K", "description": "Air temperature"},
                "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>", "description": "Geopotential"},
            }
        },
    }


def _contract(end_date="2024-01-02"):
    return {
        "schema_version": "dataset_contract.v1",
        "dataset_slug": "reanalysis_era5_pressure_levels",
        "source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download",
        "fields": [
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
            {"name": "temperature", "display_name": "Temperature", "selectors": [{"dimension": "pressure_level", "value": "850", "unit": "hPa", "label": None}]},
            {"name": "geopotential", "display_name": "Geopotential", "selectors": [{"dimension": "pressure_level", "value": "500", "unit": "hPa", "label": None}]},
        ],
        "scope": {
            "date_range": {"start_date": "2024-01-01", "end_date": end_date, "inclusive": True},
            "geography": {"area": "global", "cds_area": [90, -180, -90, 180], "cds_area_order": ["north", "west", "south", "east"]},
            "product_type": "reanalysis",
            "time": {"selected_times": ["00:00", "06:00"], "timestep": "6 hours", "timezone": "UTC"},
        },
        "advanced_options": {"data_format": "grib", "dataset_id": "reanalysis-era5-pressure-levels", "download_format": "unarchived", "product_type": ["reanalysis"]},
        "human_confirmed": True,
    }


def _make_fixture(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    times = np.array(["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00", "2024-01-02T00:00", "2024-01-02T06:00"], dtype="datetime64[ns]")
    levels = np.array([500, 850], dtype=np.int32)
    lat = np.array([90.0, 0.0, -90.0])
    lon = np.array([-180.0, 0.0, 180.0])
    shape = (len(times), len(levels), len(lat), len(lon))
    temp = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10.0) + 250.0
    temp[1, 0, 1, 1] = np.nan
    geop = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 50000.0
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure_level", "latitude", "longitude"), temp, {"units": "K"}),
            "geopotential": (("time", "pressure_level", "latitude", "longitude"), geop, {"units": "m2 s-2"}),
        },
        coords={"time": times, "pressure_level": levels, "latitude": lat, "longitude": lon},
    )
    raw_path = cache_dir / "raw_fixture.nc"
    ds.to_netcdf(raw_path, engine="h5netcdf", encoding={"temperature": {"dtype": "int16", "scale_factor": 0.1, "add_offset": 250.0, "_FillValue": -32768}})
    data = raw_path.read_bytes()
    manifest = {
        "schema_version": "source_fixture_manifest.v1",
        "entries": [{"entry_id": "era5-small", "relative_path": "raw_fixture.nc", "source": "local-test", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    (cache_dir / "source_fixture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ds


def _hash_tree(path: Path):
    result = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        result.append((str(file.relative_to(path)), hashlib.sha256(file.read_bytes()).hexdigest()))
    return result


def test_contract_envelope_validation_fixture_reuse_publication_and_filtering(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    out = tmp_path / "out"
    result = pipeline_impl.run_pipeline({"runtime": {"contract": _contract()}}, _inventory(), cache, out)

    assert result["cache"]["hits"] == 1
    assert result["cache"]["misses"] == 0
    assert result["cache"]["acquired"] == 0
    assert result["dataset_artifact"]["store_path"] == "dataset.zarr"
    assert [c["field_id"] for c in result["dataset_artifact"]["channels"]] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]

    ds = xr.open_zarr(out / "dataset.zarr", consolidated=True)
    try:
        # Inclusive date-only end means Jan 2 selected times are retained.
        assert ds.sizes["time"] == 4
        assert [str(v) for v in ds["pressure_level"].values] == ["500", "850"]
        assert tuple(ds["temperature"].dims) == ("time", "pressure_level", "latitude", "longitude")
        assert np.isnan(ds["temperature"].sel(time=np.datetime64("2024-01-01T06:00"), pressure_level=500, latitude=0.0, longitude=0.0).values)
        assert ds["temperature"].attrs["units"] == "K"
    finally:
        ds.close()


def test_full_field_tensor_chunk_layout(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    out = tmp_path / "out"
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)

    group = zarr.open_group(out / "dataset.zarr", mode="r")
    assert group["temperature"].chunks == (1, 1, 3, 3)
    assert group["geopotential"].chunks == (1, 1, 3, 3)


def test_exact_rerun_is_deterministic(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    out = tmp_path / "out"
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    first = _hash_tree(out / "dataset.zarr")
    pipeline_impl.run_pipeline(_contract(), _inventory(), cache, out)
    second = _hash_tree(out / "dataset.zarr")
    assert first == second


def test_invalid_selector_rejected_against_inventory(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    contract = _contract()
    contract["fields"][0]["selectors"][0]["value"] = "999"
    with pytest.raises(Exception, match="invalid pressure_level"):
        pipeline_impl.run_pipeline(contract, _inventory(), cache, tmp_path / "out")


def test_fixture_manifest_is_verified_before_decoding(tmp_path):
    cache = tmp_path / "cache"
    _make_fixture(cache)
    manifest_path = cache / "source_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Exception, match="sha256 mismatch"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), cache, tmp_path / "out")
    assert not (tmp_path / "out" / "dataset.zarr").exists()


def test_missing_fixture_has_no_network_fallback(tmp_path):
    with pytest.raises(Exception, match="source fixture manifest is required"):
        pipeline_impl.run_pipeline(_contract(), _inventory(), tmp_path / "empty-cache", tmp_path / "out")
