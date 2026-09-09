"""Deterministic helpers for evaluator-owned physical target mappings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from backend.evaluation.gridded_store import GriddedArray, GriddedStore


_CF_TIME_UNITS = re.compile(r"^[A-Za-z]+\s+since\s+.+$")


def resolve_cf_datetime_coordinate(
    store_path: Path,
    *,
    declared_path: str,
    sample_dimension: str,
) -> str:
    """Resolve a unique CF-time coordinate without changing candidate output.

    A dataset may use an integer dimension coordinate plus a one-dimensional
    auxiliary time coordinate. The evaluator-owned target mapping should point
    at the latter when it is uniquely identified by dimensions and CF metadata.
    """

    with GriddedStore(store_path, "zarr") as store:
        declared = store.array(declared_path)
        if declared.dimensions != (sample_dimension,):
            raise ValueError(
                f"Declared sample coordinate {declared_path!r} is not on "
                f"{sample_dimension!r}."
            )
        if _is_cf_datetime(declared):
            return declared_path

        candidates: list[str] = []
        for metadata in sorted(store_path.rglob("zarr.json")):
            relative = metadata.parent.relative_to(store_path).as_posix()
            if relative == "." or relative == declared_path:
                continue
            try:
                array = store.array(relative)
                dimensions = array.dimensions
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if dimensions == (sample_dimension,) and _is_cf_datetime(array):
                candidates.append(relative)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return declared_path
    raise ValueError(
        "Multiple CF-time arrays share the sample dimension; the suite must "
        f"map one explicitly: {candidates}"
    )


def _is_cf_datetime(array: GriddedArray) -> bool:
    if np.issubdtype(array.dtype, np.datetime64):
        return True
    attributes = array.attrs
    units = str(attributes.get("units", "")).strip()
    if _CF_TIME_UNITS.match(units):
        return True
    return array.dtype.kind in {"O", "S", "U"} and str(
        attributes.get("standard_name", "")
    ).lower() == "time"
