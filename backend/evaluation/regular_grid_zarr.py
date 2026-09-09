"""Deterministic class logic for regular rectilinear grids stored as Zarr."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr

from backend.contracts.output_policy import CoordinateMetadataRequirements
from backend.evaluation.constrained_schemas import (
    ArrayLocation,
    CoordinateLocation,
    ExpectedGridShape,
    GridSideSpec,
    LogicalChannelSpec,
    ValuePolicy,
)
from backend.evaluation.gridded_store import GriddedArray, GriddedStore
from backend.evaluation.schemas import (
    BenchmarkTarget,
    ChannelSpec,
    DimensionSpec,
)


class UnsupportedGridError(ValueError):
    """The artifact is outside the supported regular-grid dataset class."""


class OutputPolicyError(ValueError):
    """The artifact violates a generation-visible representation requirement."""


@dataclass(frozen=True)
class ResolvedChannel:
    """One native array resolved to logical sample/y/x roles."""

    field_id: str
    array: GriddedArray
    dimensions: tuple[str, ...]
    fixed_indices: dict[str, int]
    logical_dimensions: tuple[str, str, str]


@dataclass
class ResolvedGrid:
    """Opened grid whose stores must be closed by the caller."""

    side: GridSideSpec
    channels: dict[str, ResolvedChannel]
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    coordinate_orders: tuple[np.ndarray, np.ndarray, np.ndarray]
    stores: dict[Path, GriddedStore]
    store_paths: set[Path]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(len(values) for values in self.coordinates)  # type: ignore[return-value]

    def close(self) -> None:
        for store in self.stores.values():
            store.close()


def resolve_regular_grid(
    side: GridSideSpec,
    channel_specs: list[LogicalChannelSpec],
    *,
    candidate: bool,
    repository_root: Path,
    output_dir: Path | None,
    expected_shape: ExpectedGridShape,
    required_coordinate_metadata: CoordinateMetadataRequirements | None = None,
) -> ResolvedGrid:
    """Resolve one explicitly mapped Zarr side without inferring dimensions."""

    root = repository_root.resolve()
    stores: dict[Path, GriddedStore] = {}

    def store_for(raw_path: str) -> tuple[Path, GriddedStore]:
        path = resolve_store_path(
            raw_path,
            repository_root=root,
            output_dir=output_dir if candidate else None,
        )
        if path not in stores:
            stores[path] = GriddedStore(path, "zarr")
        return path, stores[path]

    try:
        coordinate_specs = (
            ("sample", side.dimensions.sample, side.sample_coordinate),
            ("y", side.dimensions.y, side.y_coordinate),
            ("x", side.dimensions.x, side.x_coordinate),
        )
        coordinates: list[np.ndarray] = []
        orders: list[np.ndarray] = []
        store_paths: set[Path] = set()
        for role, dimension, spec in coordinate_specs:
            path, store = store_for(spec.store_path)
            store_paths.add(path)
            array = store.array(spec.array_path)
            if array.dimensions != (dimension,):
                raise UnsupportedGridError(
                    f"{role} coordinate must be one-dimensional on {dimension!r}; "
                    f"observed {array.dimensions}"
                )
            values = array.read()
            normalized, order = _normalize_coordinate(values, spec, array)
            required_metadata = (
                required_coordinate_metadata.for_role(role)  # type: ignore[arg-type]
                if candidate and required_coordinate_metadata is not None
                else {}
            )
            _validate_coordinate(
                role,
                normalized,
                spec,
                array,
                required_metadata=required_metadata,
            )
            coordinates.append(normalized)
            orders.append(order)

        expected = (
            expected_shape.sample,
            expected_shape.y,
            expected_shape.x,
        )
        observed = tuple(len(values) for values in coordinates)
        if observed != expected:
            raise ValueError(
                f"Logical grid shape {observed} differs from frozen shape {expected}"
            )

        logical_dimensions = (
            side.dimensions.sample,
            side.dimensions.y,
            side.dimensions.x,
        )
        resolved: dict[str, ResolvedChannel] = {}
        for spec in channel_specs:
            location = spec.candidate if candidate else spec.reference
            path, store = store_for(location.store_path)
            store_paths.add(path)
            array = store.array(location.array_path)
            dimensions = array.dimensions
            missing = set(logical_dimensions) - set(dimensions)
            if missing:
                raise UnsupportedGridError(
                    f"Channel {spec.field_id!r} lacks logical dimensions {sorted(missing)}"
                )
            fixed = _resolve_fixed_indices(location, store, array, spec.field_id)
            fixed_logical = set(fixed) & set(logical_dimensions)
            if fixed_logical:
                raise ValueError(
                    f"Channel {spec.field_id!r} fixes logical dimensions "
                    f"{sorted(fixed_logical)}"
                )
            unresolved = [
                dimension
                for dimension in dimensions
                if dimension not in logical_dimensions and dimension not in fixed
            ]
            if unresolved:
                raise UnsupportedGridError(
                    f"Channel {spec.field_id!r} has unresolved dimensions {unresolved}"
                )
            logical_shape = tuple(
                array.shape[dimensions.index(name)] for name in logical_dimensions
            )
            if logical_shape != expected:
                raise ValueError(
                    f"Channel {spec.field_id!r} shape {logical_shape} differs from {expected}"
                )
            resolved[spec.field_id] = ResolvedChannel(
                field_id=spec.field_id,
                array=array,
                dimensions=dimensions,
                fixed_indices=fixed,
                logical_dimensions=logical_dimensions,
            )
        return ResolvedGrid(
            side=side,
            channels=resolved,
            coordinates=tuple(coordinates),  # type: ignore[arg-type]
            coordinate_orders=tuple(orders),  # type: ignore[arg-type]
            stores=stores,
            store_paths=store_paths,
        )
    except Exception:
        for store in stores.values():
            store.close()
        raise


def validate_zarr_integrity(paths: Iterable[Path], *, max_block_bytes: int) -> int:
    """Read every array through bounded blocks and return the array count."""

    array_count = 0
    for path in sorted(set(item.resolve() for item in paths)):
        group = zarr.open_group(path, mode="r")
        arrays = list(_iter_zarr_arrays(group))
        if not arrays:
            raise ValueError(f"Zarr group contains no arrays: {path}")
        for array in arrays:
            shape = tuple(int(size) for size in array.shape)
            if any(size == 0 for size in shape):
                raise ValueError(f"Zarr array is empty: {array.path}")
            itemsize = max(1, int(np.dtype(array.dtype).itemsize))
            for selection in iter_block_selections(shape, max_block_bytes // itemsize):
                np.asarray(array[selection])
            array_count += 1
    return array_count


def inspect_zarr_storage(
    paths: Iterable[Path],
    *,
    expected_format: int,
    require_consolidated: bool,
    open_latency_repetitions: int,
) -> dict[str, Any]:
    """Validate the fixed storage policy and report physical diagnostics."""

    stores = sorted({path.resolve() for path in paths})
    if not stores:
        raise ValueError("No mapped Zarr stores were supplied.")
    metadata_bytes = 0
    chunk_bytes = 0
    metadata_objects = 0
    chunk_objects = 0
    store_details: list[dict[str, Any]] = []
    for store in stores:
        root_path = store / "zarr.json"
        if not root_path.is_file():
            raise ValueError(f"Mapped store is not Zarr format 3: {store}")
        root = json.loads(root_path.read_text(encoding="utf-8"))
        if root.get("zarr_format") != expected_format or root.get("node_type") != "group":
            raise ValueError(
                f"Mapped store does not satisfy Zarr format {expected_format}: {store}"
            )
        consolidated = root.get("consolidated_metadata")
        consolidated_nodes = (
            consolidated.get("metadata") if isinstance(consolidated, dict) else None
        )
        if require_consolidated and not isinstance(consolidated_nodes, dict):
            raise ValueError(f"Mapped store lacks consolidated root metadata: {store}")

        files = sorted(path for path in store.rglob("*") if path.is_file())
        metadata_files = [path for path in files if path.name == "zarr.json"]
        chunk_files = [path for path in files if path.name != "zarr.json"]
        expected_nodes = {
            path.parent.relative_to(store).as_posix()
            for path in metadata_files
            if path != root_path
        }
        if require_consolidated and set(consolidated_nodes or {}) != expected_nodes:
            raise ValueError(
                "Consolidated metadata does not exactly cover every Zarr v3 node in "
                f"{store}"
            )
        current_metadata_bytes = sum(path.stat().st_size for path in metadata_files)
        current_chunk_bytes = sum(path.stat().st_size for path in chunk_files)
        metadata_bytes += current_metadata_bytes
        chunk_bytes += current_chunk_bytes
        metadata_objects += len(metadata_files)
        chunk_objects += len(chunk_files)
        store_details.append(
            {
                "path": str(store),
                "zarr_format": expected_format,
                "consolidated_metadata": isinstance(consolidated_nodes, dict),
                "metadata_bytes": current_metadata_bytes,
                "chunk_bytes": current_chunk_bytes,
                "object_count": len(files),
            }
        )

    open_latencies: list[float] = []
    for _ in range(open_latency_repetitions):
        started = time.perf_counter()
        for store in stores:
            group = zarr.open_group(
                store,
                mode="r",
                use_consolidated=True if require_consolidated else None,
            )
            list(_iter_zarr_arrays(group))
        open_latencies.append(time.perf_counter() - started)
    return {
        "zarr_format": expected_format,
        "consolidated_metadata": require_consolidated,
        "output_bytes": metadata_bytes + chunk_bytes,
        "chunk_bytes": chunk_bytes,
        "metadata_bytes": metadata_bytes,
        "object_count": metadata_objects + chunk_objects,
        "chunk_object_count": chunk_objects,
        "metadata_object_count": metadata_objects,
        "dataset_open_latency_seconds": {
            "repetitions": open_latency_repetitions,
            "values": open_latencies,
            "median": statistics.median(open_latencies),
            "minimum": min(open_latencies),
            "maximum": max(open_latencies),
        },
        "stores": store_details,
    }


def _iter_zarr_arrays(group: Any) -> Iterable[Any]:
    """Yield arrays recursively through APIs shared by Zarr 2 and Zarr 3."""

    for _, array in group.arrays():
        yield array
    for _, child in group.groups():
        yield from _iter_zarr_arrays(child)


def compare_coordinates(
    candidate: ResolvedGrid,
    reference: ResolvedGrid,
    *,
    atol: float,
) -> dict[str, Any]:
    """Compare all logical coordinates after only declared normalization."""

    roles = ("sample", "y", "x")
    mismatches: list[dict[str, Any]] = []
    for role, left, right in zip(
        roles, candidate.coordinates, reference.coordinates, strict=True
    ):
        if left.shape != right.shape:
            mismatches.append(
                {
                    "role": role,
                    "candidate_shape": list(left.shape),
                    "reference_shape": list(right.shape),
                }
            )
            continue
        equal = _coordinate_equal(left, right, atol=atol)
        if not bool(np.all(equal)):
            index = int(np.flatnonzero(~equal)[0])
            mismatches.append(
                {
                    "role": role,
                    "index": index,
                    "candidate": _json_scalar(left[index]),
                    "reference": _json_scalar(right[index]),
                }
            )
    return {"matched": not mismatches, "mismatches": mismatches}


def validate_metadata(
    candidate: ResolvedGrid,
    channels: list[LogicalChannelSpec],
) -> list[dict[str, Any]]:
    """Apply only suite-declared dtype and metadata assertions."""

    mismatches: list[dict[str, Any]] = []
    for spec in channels:
        array = candidate.channels[spec.field_id].array
        if spec.expected_dtype is not None:
            expected_dtype = np.dtype(spec.expected_dtype)
            if array.dtype != expected_dtype:
                mismatches.append(
                    {
                        "field_id": spec.field_id,
                        "property": "dtype",
                        "expected": str(expected_dtype),
                        "observed": str(array.dtype),
                    }
                )
        if spec.expected_fill_value is not None and not _asserted_value_equal(
            array.fill_value, spec.expected_fill_value
        ):
            mismatches.append(
                {
                    "field_id": spec.field_id,
                    "property": "encoding.fill_value",
                    "expected": spec.expected_fill_value,
                    "observed": _json_scalar(array.fill_value),
                }
            )
        attrs = array.attrs
        for name, expected in spec.metadata.items():
            observed = attrs.get(name)
            if observed != expected:
                mismatches.append(
                    {
                        "field_id": spec.field_id,
                        "property": f"attrs.{name}",
                        "expected": expected,
                        "observed": observed,
                    }
                )
    return mismatches


def compare_values(
    candidate: ResolvedGrid,
    reference: ResolvedGrid,
    channels: list[LogicalChannelSpec],
    *,
    default_policy: ValuePolicy,
    max_block_bytes: int,
    mismatch_limit: int,
) -> dict[str, Any]:
    """Compare every native value and missingness marker through bounded reads."""

    total = 0
    mismatched = 0
    missingness_mismatched = 0
    max_absolute: float | None = None
    max_relative: float | None = None
    infinite_relative_errors = 0
    examples: list[dict[str, Any]] = []
    affected_channels: set[str] = set()
    for spec in channels:
        left_channel = candidate.channels[spec.field_id]
        right_channel = reference.channels[spec.field_id]
        if not (
            np.issubdtype(left_channel.array.dtype, np.number)
            and np.issubdtype(right_channel.array.dtype, np.number)
        ):
            raise UnsupportedGridError(
                f"Channel {spec.field_id!r} is not numeric on both sides"
            )
        policy = spec.value_policy or default_policy
        itemsize = max(
            left_channel.array.dtype.itemsize, right_channel.array.dtype.itemsize, 1
        )
        for logical_slice in iter_block_selections(
            candidate.shape, max_block_bytes // itemsize
        ):
            left = read_logical_block(candidate, left_channel, logical_slice)
            right = read_logical_block(reference, right_channel, logical_slice)
            left_missing = _missing_mask(left, policy)
            right_missing = _missing_mask(right, policy)
            missing_delta = left_missing != right_missing
            comparable = ~(left_missing | right_missing)
            equal_values = np.ones(left.shape, dtype=bool)
            if not policy.equal_nan:
                equal_values[_nan_mask(left) & _nan_mask(right)] = False
            equal_values[comparable] = np.isclose(
                left[comparable],
                right[comparable],
                atol=policy.atol,
                rtol=policy.rtol,
                equal_nan=policy.equal_nan,
            )
            unequal = missing_delta | ~equal_values
            count = int(np.count_nonzero(unequal))
            total += int(unequal.size)
            mismatched += count
            missingness_mismatched += int(np.count_nonzero(missing_delta))
            if count:
                affected_channels.add(spec.field_id)
                _append_mismatch_examples(
                    examples,
                    unequal,
                    left,
                    right,
                    logical_slice,
                    spec.field_id,
                    mismatch_limit,
                )
            if bool(np.any(comparable)):
                left_float = left[comparable].astype(np.float64)
                right_float = right[comparable].astype(np.float64)
                absolute = np.abs(left_float - right_float)
                relative = np.divide(
                    absolute,
                    np.abs(right_float),
                    out=np.full_like(absolute, np.inf),
                    where=right_float != 0,
                )
                relative[(absolute == 0) & (right_float == 0)] = 0
                max_absolute = _max_finite(max_absolute, absolute)
                max_relative = _max_finite(max_relative, relative)
                infinite_relative_errors += int(np.count_nonzero(np.isinf(relative)))
    return {
        "matched": mismatched == 0,
        "values_compared": total,
        "mismatch_count": mismatched,
        "missingness_mismatch_count": missingness_mismatched,
        "max_finite_absolute_error": max_absolute,
        "max_finite_relative_error": max_relative,
        "infinite_relative_error_count": infinite_relative_errors,
        "affected_channels": sorted(affected_channels),
        "examples": examples,
    }


def logical_fingerprint(
    grid: ResolvedGrid,
    channels: list[LogicalChannelSpec],
    *,
    default_policy: ValuePolicy,
    max_block_bytes: int,
) -> str:
    """Hash normalized logical coordinates, native dtypes, values, and missingness."""

    digest = hashlib.sha256()
    digest.update(b"regular_grid_zarr.logical.v1\0")
    for role, coordinate in zip(("sample", "y", "x"), grid.coordinates, strict=True):
        digest.update(role.encode("utf-8") + b"\0")
        _hash_array(digest, coordinate, np.zeros(coordinate.shape, dtype=bool))
    for spec in channels:
        channel = grid.channels[spec.field_id]
        policy = spec.value_policy or default_policy
        digest.update(spec.field_id.encode("utf-8") + b"\0")
        digest.update(channel.array.dtype.str.encode("ascii") + b"\0")
        itemsize = max(1, channel.array.dtype.itemsize)
        for logical_slice in iter_block_selections(
            grid.shape, max_block_bytes // itemsize
        ):
            values = read_logical_block(grid, channel, logical_slice)
            _hash_array(digest, values, _missing_mask(values, policy))
    return digest.hexdigest()


def logical_uncompressed_bytes(
    grid: ResolvedGrid, channels: list[LogicalChannelSpec]
) -> int:
    """Return cheap logical bytes for diagnostics, not optimization."""

    coordinate_bytes = sum(values.nbytes for values in grid.coordinates)
    value_count = math.prod(grid.shape)
    channel_bytes = sum(
        value_count * grid.channels[spec.field_id].array.dtype.itemsize
        for spec in channels
    )
    return int(coordinate_bytes + channel_bytes)


def benchmark_target(
    grid: ResolvedGrid,
    channels: list[LogicalChannelSpec],
    *,
    output_dir: Path,
    repository_root: Path,
) -> BenchmarkTarget:
    """Expose the validated candidate through the evaluator-owned PyTorch adapter."""

    channel_mappings = []
    for spec in channels:
        location = spec.candidate
        channel_mappings.append(
            ChannelSpec(
                name=spec.field_id,
                store_path=str(
                    resolve_store_path(
                        location.store_path,
                        repository_root=repository_root,
                        output_dir=output_dir,
                    )
                ),
                array_path=location.array_path,
                storage_format="zarr",
                selectors=location.selectors,
                indices=location.indices,
                selector_coordinate_paths=location.selector_coordinate_paths,
            )
        )
    return BenchmarkTarget(
        name="constrained_candidate",
        dimensions=DimensionSpec(
            time=grid.side.dimensions.sample,
            y=grid.side.dimensions.y,
            x=grid.side.dimensions.x,
        ),
        channels=channel_mappings,
    )


def resolve_store_path(
    raw_path: str,
    *,
    repository_root: Path,
    output_dir: Path | None,
) -> Path:
    """Resolve a suite path and confine candidate artifacts to the output root."""

    if output_dir is not None:
        replaced = raw_path.replace("{output_dir}", str(output_dir.resolve()))
        if "{" in replaced or "}" in replaced:
            raise ValueError(f"Unresolved store placeholder in {raw_path!r}")
        path = Path(replaced).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Candidate store escapes output directory: {raw_path}"
            ) from error
        return path
    if "{" in raw_path or "}" in raw_path:
        raise ValueError(f"Reference paths cannot contain placeholders: {raw_path}")
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def iter_block_selections(
    shape: tuple[int, ...], max_elements: int
) -> Iterable[tuple[slice, ...]]:
    """Yield complete non-overlapping N-D selections bounded by element count."""

    if not shape:
        yield ()
        return
    if max_elements < 1:
        max_elements = 1
    block = list(shape)
    while math.prod(block) > max_elements:
        axis = max(range(len(block)), key=block.__getitem__)
        block[axis] = max(1, (block[axis] + 1) // 2)
    starts = [range(0, size, width) for size, width in zip(shape, block, strict=True)]
    for offsets in itertools.product(*starts):
        yield tuple(
            slice(start, min(size, start + width))
            for start, size, width in zip(offsets, shape, block, strict=True)
        )


def _resolve_fixed_indices(
    location: ArrayLocation,
    store: GriddedStore,
    array: GriddedArray,
    field_id: str,
) -> dict[str, int]:
    fixed = dict(location.indices)
    for dimension, selected in location.selectors.items():
        if (
            dimension not in location.selector_coordinate_paths
            and dimension not in array.dimensions
        ):
            # The selector is materialized into this logical channel and no
            # longer exists as a physical array axis. Semantic comparison still
            # validates the channel values against the selected-field oracle.
            continue
        coordinate_path = location.selector_coordinate_paths.get(dimension, dimension)
        coordinate_array = store.array(coordinate_path)
        coordinate = coordinate_array.read()
        if coordinate.ndim == 0:
            if not _selector_values_equal(coordinate.reshape(()).item(), selected):
                raise ValueError(
                    f"Scalar selector {dimension}={selected!r} for {field_id!r} "
                    f"does not match {coordinate.reshape(()).item()!r}"
                )
            continue
        if coordinate.ndim != 1:
            raise UnsupportedGridError(
                f"Selector coordinate {coordinate_path!r} is neither scalar nor one-dimensional"
            )
        if np.issubdtype(coordinate.dtype, np.number):
            try:
                selected_number = float(selected)
            except (TypeError, ValueError):
                matches = np.array([], dtype=np.int64)
            else:
                matches = np.flatnonzero(
                    np.isclose(coordinate.astype(np.float64), selected_number)
                )
        else:
            matches = np.flatnonzero(coordinate.astype(str) == str(selected))
        if len(matches) != 1:
            raise ValueError(
                f"Selector {dimension}={selected!r} for {field_id!r} matched "
                f"{len(matches)} coordinates"
            )
        coordinate_dimensions = coordinate_array.dimensions
        if len(coordinate_dimensions) != 1:
            raise UnsupportedGridError(
                f"Selector coordinate {coordinate_path!r} has no single physical dimension"
            )
        physical_dimension = coordinate_dimensions[0]
        if physical_dimension not in array.dimensions:
            raise ValueError(
                f"Selector coordinate {coordinate_path!r} uses dimension "
                f"{physical_dimension!r}, which is absent from channel {field_id!r}"
            )
        fixed[physical_dimension] = int(matches[0])
    for dimension, index in fixed.items():
        if dimension not in array.dimensions:
            raise ValueError(f"Channel {field_id!r} has no dimension {dimension!r}")
        size = array.shape[array.dimensions.index(dimension)]
        if index >= size:
            raise ValueError(
                f"Channel {field_id!r} index {index} exceeds {dimension!r} size {size}"
            )
    return fixed


def _selector_values_equal(observed: Any, selected: Any) -> bool:
    observed_array = np.asarray(observed)
    if np.issubdtype(observed_array.dtype, np.number):
        try:
            return bool(np.isclose(float(observed), float(selected)))
        except (TypeError, ValueError):
            return False
    return str(observed) == str(selected)


def _normalize_coordinate(
    values: np.ndarray,
    spec: CoordinateLocation,
    array: GriddedArray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    order = np.arange(values.size)
    if spec.normalization == "none":
        return values, order
    if spec.normalization in {"cf_datetime", "cf_datetime_ascending"}:
        values = _decode_cf_datetime(values, array)
        if spec.normalization == "cf_datetime":
            return values, order
    if spec.normalization == "longitude_modulo_360":
        if not np.issubdtype(values.dtype, np.number):
            raise UnsupportedGridError(
                "Longitude normalization requires numeric values"
            )
        values = np.mod(values, 360)
    order = np.argsort(values, kind="stable")
    return values[order], order


def _decode_cf_datetime(values: np.ndarray, array: GriddedArray) -> np.ndarray:
    """Normalize native/ISO datetimes or bounded CF numeric coordinates."""

    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    if values.dtype.kind in {"O", "S", "U"}:
        try:
            normalized = values.astype(str).astype("datetime64[ns]")
        except (TypeError, ValueError) as error:
            raise UnsupportedGridError(
                "Datetime normalization requires ISO-8601 string values"
            ) from error
        if np.isnat(normalized).any():
            raise UnsupportedGridError("Datetime normalization produced NaT values")
        return normalized
    if not np.issubdtype(values.dtype, np.number):
        raise UnsupportedGridError(
            "Datetime normalization requires datetime, ISO-8601 string, or numeric CF values"
        )
    calendar = str(array.attrs.get("calendar", "standard"))
    if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise UnsupportedGridError(
            f"CF calendar {calendar!r} requires a dataset-specific time adapter"
        )
    units = str(array.attrs.get("units", ""))
    match = re.fullmatch(
        r"\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*",
        units,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise UnsupportedGridError(
            "CF datetime normalization requires '<unit> since <origin>' units"
        )
    nanoseconds = {
        "seconds": 1_000_000_000,
        "minutes": 60 * 1_000_000_000,
        "hours": 3_600 * 1_000_000_000,
        "days": 86_400 * 1_000_000_000,
    }[match.group(1).lower()]
    try:
        origin = np.datetime64(match.group(2).replace(" ", "T"), "ns")
    except ValueError as error:
        raise UnsupportedGridError(
            f"CF datetime origin is not supported: {match.group(2)!r}"
        ) from error
    offsets = np.rint(values.astype(np.float64) * nanoseconds).astype(
        "timedelta64[ns]"
    )
    return origin + offsets


def _validate_coordinate(
    role: str,
    values: np.ndarray,
    spec: CoordinateLocation,
    array: GriddedArray,
    *,
    required_metadata: dict[str, Any],
) -> None:
    if values.ndim != 1 or values.size == 0:
        raise UnsupportedGridError(f"{role} coordinate is not a non-empty 1-D array")
    supported = (
        np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.datetime64)
        or np.issubdtype(values.dtype, np.timedelta64)
    )
    if not supported:
        raise UnsupportedGridError(
            f"{role} coordinate dtype {values.dtype} is outside the supported class"
        )
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{role} coordinate contains non-finite values")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{role} coordinate contains duplicates")
    if spec.expected_dtype is not None and array.dtype != np.dtype(spec.expected_dtype):
        raise ValueError(
            f"{role} coordinate dtype {array.dtype} differs from {spec.expected_dtype}"
        )
    if spec.expected_fill_value is not None and not _asserted_value_equal(
        array.fill_value, spec.expected_fill_value
    ):
        raise ValueError(
            f"{role} coordinate fill value {array.fill_value!r} differs from "
            f"{spec.expected_fill_value!r}"
        )
    declared_metadata = dict(spec.metadata)
    for name, expected in required_metadata.items():
        if name in declared_metadata and declared_metadata[name] != expected:
            raise OutputPolicyError(
                f"{role} coordinate metadata {name!r} conflicts with the public output policy"
            )
        declared_metadata[name] = expected
    for name, expected in declared_metadata.items():
        if array.attrs.get(name) != expected:
            error_type = OutputPolicyError if name in required_metadata else ValueError
            raise error_type(
                f"{role} coordinate metadata {name!r} differs from {expected!r}"
            )
    differences = np.diff(values)
    if spec.monotonic is not None:
        zero: Any = (
            np.timedelta64(0, "ns")
            if np.issubdtype(differences.dtype, np.timedelta64)
            else 0
        )
        monotonic = (
            bool(np.all(differences > zero))
            if spec.monotonic == "increasing"
            else bool(np.all(differences < zero))
        )
        if not monotonic:
            raise ValueError(
                f"{role} coordinate is not strictly {spec.monotonic} as declared"
            )
    if spec.expected_step is not None:
        if not np.issubdtype(differences.dtype, np.number):
            raise UnsupportedGridError(
                f"{role} expected_step currently requires numeric coordinates"
            )
        if not bool(
            np.allclose(
                np.abs(differences.astype(np.float64)),
                spec.expected_step,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise ValueError(
                f"{role} coordinate step differs from {spec.expected_step}"
            )
    if spec.expected_step_seconds is not None:
        if not (
            np.issubdtype(values.dtype, np.datetime64)
            or np.issubdtype(values.dtype, np.timedelta64)
        ):
            raise UnsupportedGridError(
                f"{role} expected_step_seconds requires datetime coordinates"
            )
        seconds = np.abs(
            differences.astype("timedelta64[ns]").astype(np.int64) / 1_000_000_000
        )
        if not bool(
            np.allclose(
                seconds,
                spec.expected_step_seconds,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise ValueError(
                f"{role} coordinate step differs from "
                f"{spec.expected_step_seconds} seconds"
            )


def read_logical_block(
    grid: ResolvedGrid,
    channel: ResolvedChannel,
    logical_slice: tuple[slice, slice, slice],
) -> np.ndarray:
    selection: list[Any] = []
    remaining: list[str] = []
    order_by_dimension = dict(
        zip(channel.logical_dimensions, grid.coordinate_orders, strict=True)
    )
    slice_by_dimension = dict(
        zip(channel.logical_dimensions, logical_slice, strict=True)
    )
    for dimension in channel.dimensions:
        if dimension in channel.fixed_indices:
            selection.append(channel.fixed_indices[dimension])
        else:
            normalized_slice = slice_by_dimension[dimension]
            indices = order_by_dimension[dimension][normalized_slice]
            selection.append(indices)
            remaining.append(dimension)
    values = channel.array.read_orthogonal(tuple(selection))
    axes = tuple(remaining.index(name) for name in channel.logical_dimensions)
    return np.transpose(values, axes)


def _coordinate_equal(
    left: np.ndarray, right: np.ndarray, *, atol: float
) -> np.ndarray:
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        return np.isclose(left, right, atol=atol, rtol=0.0, equal_nan=False)
    return left == right


def _missing_mask(values: np.ndarray, policy: ValuePolicy) -> np.ndarray:
    missing = _nan_mask(values)
    for marker in policy.missing_values:
        missing |= values == marker
    return missing


def _nan_mask(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        return np.isnan(values)
    return np.zeros(values.shape, dtype=bool)


def _append_mismatch_examples(
    examples: list[dict[str, Any]],
    unequal: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    logical_slice: tuple[slice, slice, slice],
    field_id: str,
    limit: int,
) -> None:
    remaining = limit - len(examples)
    if remaining <= 0:
        return
    starts = tuple(item.start or 0 for item in logical_slice)
    for local in np.argwhere(unequal)[:remaining]:
        index = tuple(starts[axis] + int(value) for axis, value in enumerate(local))
        examples.append(
            {
                "field_id": field_id,
                "index": {"sample": index[0], "y": index[1], "x": index[2]},
                "candidate": _json_scalar(candidate[tuple(local)]),
                "reference": _json_scalar(reference[tuple(local)]),
            }
        )


def _max_finite(current: float | None, values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return current
    observed = float(np.max(finite))
    return observed if current is None else max(current, observed)


def _hash_array(digest: Any, values: np.ndarray, missing: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    normalized = contiguous.copy()
    normalized[missing] = 0
    digest.update(str(normalized.shape).encode("ascii") + b"\0")
    digest.update(normalized.dtype.str.encode("ascii") + b"\0")
    digest.update(np.ascontiguousarray(missing).tobytes())
    digest.update(normalized.tobytes())


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.datetime64) or np.issubdtype(
            value.dtype, np.timedelta64
        ):
            return str(value)
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _asserted_value_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, str) and expected.lower() == "nan":
        try:
            return bool(np.isnan(observed))
        except TypeError:
            return False
    return bool(observed == expected)
