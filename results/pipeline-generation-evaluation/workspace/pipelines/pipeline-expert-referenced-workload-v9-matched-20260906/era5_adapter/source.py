from __future__ import annotations

from pathlib import Path
from typing import Iterable

import xarray as xr


class SourceError(RuntimeError):
    """Raised when verified fixture files cannot be decoded."""


_RENAMES = {
    "valid_time": "time",
    "lat": "latitude",
    "lon": "longitude",
    "isobaricInhPa": "pressure_level",
    "level": "pressure_level",
}


def open_fixture_datasets(paths: Iterable[Path]) -> list[xr.Dataset]:
    datasets: list[xr.Dataset] = []
    errors: list[str] = []
    for path in paths:
        try:
            datasets.extend(_open_one(path))
        except Exception as exc:  # collect all paths for actionable diagnostics
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not datasets:
        raise SourceError("no fixture datasets could be opened; " + "; ".join(errors))
    return [_normalize_names(ds.load()) for ds in datasets]


def _open_one(path: Path) -> list[xr.Dataset]:
    suffix = path.suffix.lower()
    if path.is_dir() or suffix == ".zarr":
        return [xr.open_zarr(path, consolidated=None)]

    # NetCDF/HDF fixtures are useful in offline tests and are accepted as decoded
    # local source objects. Provider packing/missing-value conventions are decoded
    # by xarray before publication.
    if suffix in {".nc", ".nc4", ".cdf", ".h5", ".hdf5"}:
        return [xr.open_dataset(path, decode_cf=True)]

    # Native CDS acquisition format for this policy is GRIB. cfgrib may split a
    # heterogeneous GRIB into multiple hypercubes; all are normalized and merged
    # later by variable selection.
    try:
        import cfgrib  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional runtime system libs
        raise SourceError("GRIB fixture requires cfgrib/eccodes to be installed") from exc
    try:
        return list(xr.open_datasets(path, engine="cfgrib", backend_kwargs={"indexpath": ""}))
    except AttributeError:
        return [xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})]


def _normalize_names(ds: xr.Dataset) -> xr.Dataset:
    existing_names = set(ds.dims) | set(ds.coords) | set(ds.data_vars)
    renames = {
        old: new
        for old, new in _RENAMES.items()
        if (old in ds.dims or old in ds.coords) and new not in existing_names
    }
    if renames:
        ds = ds.rename(renames)
    if "time" not in ds.coords and "time" in ds.dims:
        ds = ds.assign_coords(time=ds["time"])
    return ds
