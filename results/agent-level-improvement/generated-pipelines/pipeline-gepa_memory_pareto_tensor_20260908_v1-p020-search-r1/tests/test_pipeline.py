from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import pipeline_impl


INVENTORY = {
    "schema_version": "dataset_inventory.v1",
    "dataset_slug": "reanalysis_era5_pressure_levels",
    "dataset_id": "reanalysis-era5-pressure-levels",
    "options": {
        "variable": ["temperature", "geopotential"],
        "pressure_level": ["500", "850"],
        "year": ["2024"],
        "month": ["01"],
        "day": [f"{i:02d}" for i in range(1, 32)],
        "time": [f"{i:02d}:00" for i in range(24)],
        "data_format": ["grib", "netcdf"],
        "download_format": ["zip", "unarchived"],
        "product_type": ["reanalysis"],
    },
    "option_metadata": {
        "variable": {
            "temperature": {"label": "Temperature", "units": "K"},
            "geopotential": {"label": "Geopotential", "units": "m<sup>2</sup> s<sup>-2</sup>"},
        }
    },
}


CONTRACT = {
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


POLICY = {
    "provider": "ECMWF",
    "dataset_id": "reanalysis-era5-pressure-levels",
    "acquisition_format": "grib",
    "publication_format": "zarr",
    "zarr": {"format_version": 3, "consolidated_metadata": True},
}


def _make_fixture(cache: Path):
    times = pd.date_range("2024-01-01T00:00", "2024-01-07T18:00", freq="6h")
    levels = np.array([500, 850, 700], dtype=np.int32)
    lat = np.array([90.0, 45.0, 0.0, -45.0, -90.0], dtype=np.float32)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float32)
    shape = (len(times), len(levels), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    temp = base.copy()
    geop = base + 10000
    temp[1, 0, 2, 2] = np.nan
    ds = xr.Dataset(
        {
            "t": (("time", "isobaricInhPa", "latitude", "longitude"), temp, {"GRIB_shortName": "t", "units": "K"}),
            "z": (("time", "isobaricInhPa", "latitude", "longitude"), geop, {"GRIB_shortName": "z", "units": "m2 s-2"}),
        },
        coords={"time": times, "isobaricInhPa": levels, "latitude": lat, "longitude": lon},
    )
    raw = cache / "raw.zarr"
    ds.to_zarr(raw, mode="w", consolidated=False)
    entries = []
    for p in sorted(raw.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            entries.append({
                "entry_id": p.relative_to(cache).as_posix(),
                "relative_path": p.relative_to(cache).as_posix(),
                "source": "local-test-fixture",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    (cache / "source_fixture_manifest.json").write_text(json.dumps({"schema_version": "source_fixture_manifest.v1", "entries": entries}, sort_keys=True))
    return ds


def _run(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    src = _make_fixture(cache)
    lock = {"contract": CONTRACT, "fixed_output_policy": POLICY}
    result = pipeline_impl.run_pipeline(lock, INVENTORY, str(cache), str(out))
    return src, out, result


def test_publication_artifact_grouped_layout_chunking_codec_and_values(tmp_path):
    src, out, result = _run(tmp_path)
    artifact = result["dataset_artifact"]
    assert artifact["store_path"] == "era5_pressure_levels.zarr"
    assert artifact["dimensions"] == {"sample": "time", "y": "latitude", "x": "longitude"}
    assert len(artifact["channels"]) == 3

    channels = artifact["channels"]
    assert [c["field_id"] for c in channels] == [
        'temperature[pressure_level="500"]',
        'temperature[pressure_level="850"]',
        'geopotential[pressure_level="500"]',
    ]
    assert channels[0]["array_path"] == channels[1]["array_path"] == "temperature"
    assert channels[0]["selector_coordinate_paths"] == {"pressure_level": "temperature_pressure_level"}
    assert channels[2]["array_path"] == "geopotential"
    assert channels[2]["selector_coordinate_paths"] == {"pressure_level": "geopotential_pressure_level"}

    store = out / artifact["store_path"]
    root_meta = json.loads((store / "zarr.json").read_text())
    assert "consolidated_metadata" in root_meta
    ds = xr.open_zarr(store, consolidated=True).load()

    assert set(ds.data_vars) == {"temperature", "geopotential"}
    assert ds["temperature"].dims == ("time", "temperature_pressure_level", "latitude", "longitude")
    assert ds["geopotential"].dims == ("time", "geopotential_pressure_level", "latitude", "longitude")
    assert ds.sizes["temperature_pressure_level"] == 2
    assert ds.sizes["geopotential_pressure_level"] == 1
    assert ds["temperature_pressure_level"].values.tolist() == [500, 850]
    assert ds["geopotential_pressure_level"].values.tolist() == [500]
    assert "700" not in [str(x) for x in ds["temperature_pressure_level"].values]

    expected_t = src["t"].sel(isobaricInhPa=[500, 850]).rename({"isobaricInhPa": "temperature_pressure_level"})
    expected_z = src["z"].sel(isobaricInhPa=[500]).rename({"isobaricInhPa": "geopotential_pressure_level"})
    xr.testing.assert_equal(ds["temperature"], expected_t.astype(np.float32))
    xr.testing.assert_equal(ds["geopotential"], expected_z.astype(np.float32))
    assert np.isnan(ds["temperature"].isel(time=1, temperature_pressure_level=0, latitude=2, longitude=2).item())

    for name, selector_dim in [("temperature", "temperature_pressure_level"), ("geopotential", "geopotential_pressure_level")]:
        arr = zarr.open(str(store / name), mode="r")
        assert tuple(arr.chunks) == (1, 1, ds.sizes["latitude"], ds.sizes["longitude"])
        codec_text = repr(arr.metadata.codecs).lower()
        assert "blosc" in codec_text and "zstd" in codec_text and "clevel=9" in codec_text and "bitshuffle" in codec_text
        assert selector_dim in ds.dims


def test_inclusive_date_only_end_keeps_final_day_times(tmp_path):
    _, out, _ = _run(tmp_path)
    ds = xr.open_zarr(out / "era5_pressure_levels.zarr", consolidated=True)
    assert ds.sizes["time"] == 28
    assert str(pd.Timestamp(ds.time.values[-1])) == "2024-01-07 18:00:00"


def test_invalid_selector_rejected(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _make_fixture(cache)
    bad = copy.deepcopy(CONTRACT)
    bad["fields"][0]["selectors"][0]["value"] = "925"
    with pytest.raises(ValueError):
        pipeline_impl.run_pipeline({"contract": bad, "fixed_output_policy": POLICY}, INVENTORY, str(cache), str(out))


def test_fixture_manifest_verification_rejects_tamper_and_never_needs_credentials(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _make_fixture(cache)
    manifest = json.loads((cache / "source_fixture_manifest.json").read_text())
    manifest["entries"][0]["sha256"] = "0" * 64
    (cache / "source_fixture_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    with pytest.raises(ValueError, match="fixture entry failed verification"):
        pipeline_impl.run_pipeline({"contract": CONTRACT, "fixed_output_policy": POLICY}, INVENTORY, str(cache), str(out))


def test_missing_manifest_fails_safely_without_network(tmp_path):
    with pytest.raises(RuntimeError, match="network acquisition is not permitted"):
        pipeline_impl.run_pipeline({"contract": CONTRACT, "fixed_output_policy": POLICY}, INVENTORY, str(tmp_path / "cache"), str(tmp_path / "out"))


def test_rerun_equivalence_and_cache_evidence(tmp_path):
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    cache.mkdir()
    _make_fixture(cache)
    lock = {"contract": CONTRACT, "fixed_output_policy": POLICY}
    r1 = pipeline_impl.run_pipeline(lock, INVENTORY, str(cache), str(out))
    ds1 = xr.open_zarr(out / "era5_pressure_levels.zarr", consolidated=True).load()
    r2 = pipeline_impl.run_pipeline(lock, INVENTORY, str(cache), str(out))
    ds2 = xr.open_zarr(out / "era5_pressure_levels.zarr", consolidated=True).load()
    assert r1 == r2
    assert r1["cache"]["acquired"] == 0
    assert r1["cache"]["misses"] == 0
    xr.testing.assert_identical(ds1, ds2)
