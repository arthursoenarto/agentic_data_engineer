"""Evaluator-owned PyTorch adapters over registered gridded outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from backend.evaluation.gridded_store import GriddedStore
from backend.evaluation.schemas import BenchmarkTarget, ChannelSpec


@dataclass(frozen=True)
class PatchRequest:
    """A deterministic spatial access policy applied before materialization."""

    size: int
    mode: Literal["random_misaligned", "chunk_aligned"]
    chunk_size: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class _ResolvedChannel:
    name: str
    store_path: str
    storage_format: str
    array_path: str
    dimensions: tuple[str, ...]
    fixed_indices: dict[str, int]


class GriddedTensorDataset(Dataset[Tensor]):
    """Present explicit gridded field mappings as ``[C, H, W]`` tensors."""

    def __init__(
        self,
        target: BenchmarkTarget,
        *,
        channel_count: int | None = None,
        sample_limit: int | None = None,
        patch: PatchRequest | None = None,
    ) -> None:
        selected_count = channel_count or len(target.channels)
        if not 1 <= selected_count <= len(target.channels):
            raise ValueError(
                f"channel_count must be between 1 and {len(target.channels)}"
            )

        self.target_name = target.name
        self.dimension_spec = target.dimensions
        self.patch = patch
        self._channel_specs = target.channels[:selected_count]
        self._stores: dict[tuple[str, str], GriddedStore] | None = None
        self._resolved, shape = self._resolve_channels(self._channel_specs)
        self._time_count, self._height, self._width = shape
        self._sample_count = min(self._time_count, sample_limit or self._time_count)

        if patch is not None:
            if patch.size > self._height or patch.size > self._width:
                raise ValueError(
                    f"Patch {patch.size} exceeds spatial shape {(self._height, self._width)}"
                )
            if patch.mode == "chunk_aligned" and patch.chunk_size != patch.size:
                raise ValueError(
                    "chunk-aligned reads require patch size to equal chunk size"
                )

    @property
    def channel_names(self) -> list[str]:
        return [channel.name for channel in self._resolved]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self._height, self._width

    @property
    def source_paths(self) -> list[Path]:
        return [
            Path(path)
            for path, _ in dict.fromkeys(
                (item.store_path, item.storage_format) for item in self._resolved
            )
        ]

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: int) -> Tensor:
        return torch.from_numpy(self.read_numpy(index))

    def read_numpy(self, index: int) -> np.ndarray:
        """Read one full field or patch before converting it to a tensor."""

        if not 0 <= index < self._sample_count:
            raise IndexError(index)
        y_slice, x_slice = self._spatial_slices(index)
        arrays = [
            self._read_channel(channel, index, y_slice, x_slice)
            for channel in self._resolved
        ]
        return np.stack(arrays).astype(np.float32, copy=False)

    def _resolve_channels(
        self, channel_specs: list[ChannelSpec]
    ) -> tuple[list[_ResolvedChannel], tuple[int, int, int]]:
        resolved: list[_ResolvedChannel] = []
        expected_shape: tuple[int, int, int] | None = None
        stores: dict[tuple[str, str], GriddedStore] = {}

        try:
            for spec in channel_specs:
                key = (spec.store_path, spec.storage_format)
                if key not in stores:
                    stores[key] = GriddedStore(spec.store_path, spec.storage_format)
                store = stores[key]
                array = store.array(spec.array_path)
                dimensions = array.dimensions
                required = {
                    self.dimension_spec.time,
                    self.dimension_spec.y,
                    self.dimension_spec.x,
                }
                missing = required - set(dimensions)
                if missing:
                    raise ValueError(
                        f"Channel {spec.name!r} is missing dimensions: {sorted(missing)}"
                    )

                fixed_indices = dict(spec.indices)
                for dimension, value in spec.selectors.items():
                    if (
                        dimension not in spec.selector_coordinate_paths
                        and dimension not in dimensions
                    ):
                        # The selector was materialized into this channel, so it
                        # is logical identity rather than a remaining array axis.
                        continue
                    physical_dimension, position = _coordinate_index(
                        store,
                        dimension,
                        value,
                        array_path=spec.selector_coordinate_paths.get(
                            dimension, dimension
                        ),
                    )
                    if physical_dimension is not None and position is not None:
                        if physical_dimension not in dimensions:
                            raise ValueError(
                                f"Selector coordinate for {dimension!r} uses dimension "
                                f"{physical_dimension!r}, which is absent from "
                                f"channel {spec.name!r}"
                            )
                        fixed_indices[physical_dimension] = position

                for dimension, size in zip(dimensions, array.shape, strict=True):
                    if dimension in required or dimension in fixed_indices:
                        continue
                    if size == 1:
                        fixed_indices[dimension] = 0
                        continue
                    raise ValueError(
                        f"Channel {spec.name!r} has unresolved dimension "
                        f"{dimension!r} of size {size}"
                    )

                for dimension, position in fixed_indices.items():
                    if dimension not in dimensions:
                        raise ValueError(
                            f"Channel {spec.name!r} has no dimension {dimension!r}"
                        )
                    size = array.shape[dimensions.index(dimension)]
                    if not 0 <= position < size:
                        raise ValueError(
                            f"Index {position} is outside dimension "
                            f"{dimension!r} of size {size}"
                        )

                shape = (
                    array.shape[dimensions.index(self.dimension_spec.time)],
                    array.shape[dimensions.index(self.dimension_spec.y)],
                    array.shape[dimensions.index(self.dimension_spec.x)],
                )
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise ValueError(
                        f"Channel {spec.name!r} shape {shape} does not match "
                        f"{expected_shape}"
                    )

                resolved.append(
                    _ResolvedChannel(
                        name=spec.name,
                        store_path=spec.store_path,
                        storage_format=spec.storage_format,
                        array_path=spec.array_path,
                        dimensions=dimensions,
                        fixed_indices=fixed_indices,
                    )
                )
        finally:
            for store in stores.values():
                store.close()

        assert expected_shape is not None
        return resolved, expected_shape

    def _read_channel(
        self,
        channel: _ResolvedChannel,
        time_index: int,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        store = self._open_stores()[(channel.store_path, channel.storage_format)]
        array = store.array(channel.array_path)
        selection: list[int | slice] = []
        remaining_dimensions: list[str] = []

        for dimension in channel.dimensions:
            if dimension == self.dimension_spec.time:
                selection.append(time_index)
            elif dimension == self.dimension_spec.y:
                selection.append(y_slice)
                remaining_dimensions.append(dimension)
            elif dimension == self.dimension_spec.x:
                selection.append(x_slice)
                remaining_dimensions.append(dimension)
            else:
                selection.append(channel.fixed_indices[dimension])

        result = np.asarray(array.read(tuple(selection)), dtype=np.float32)
        expected_dimensions = [self.dimension_spec.y, self.dimension_spec.x]
        if remaining_dimensions == list(reversed(expected_dimensions)):
            result = result.T
        elif remaining_dimensions != expected_dimensions:
            raise ValueError(
                f"Channel {channel.name!r} produced unsupported spatial order "
                f"{remaining_dimensions}"
            )
        return result

    def _open_stores(self) -> dict[tuple[str, str], GriddedStore]:
        if self._stores is None:
            self._stores = {
                (channel.store_path, channel.storage_format): GriddedStore(
                    channel.store_path, channel.storage_format
                )
                for channel in self._resolved
            }
        return self._stores

    def _spatial_slices(self, index: int) -> tuple[slice, slice]:
        if self.patch is None:
            return slice(None), slice(None)

        size = self.patch.size
        rng = np.random.default_rng(self.patch.seed + index)
        if self.patch.mode == "chunk_aligned":
            assert self.patch.chunk_size is not None
            row_count = ((self._height - size) // self.patch.chunk_size) + 1
            col_count = ((self._width - size) // self.patch.chunk_size) + 1
            top = int(rng.integers(0, row_count)) * self.patch.chunk_size
            left = int(rng.integers(0, col_count)) * self.patch.chunk_size
        else:
            top = int(rng.integers(0, self._height - size + 1))
            left = int(rng.integers(0, self._width - size + 1))
            chunk_size = self.patch.chunk_size
            if chunk_size and top % chunk_size == 0 and left % chunk_size == 0:
                if left < self._width - size:
                    left += 1
                elif top < self._height - size:
                    top += 1

        return slice(top, top + size), slice(left, left + size)

    def close(self) -> None:
        if self._stores is not None:
            for store in self._stores.values():
                store.close()
            self._stores = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_stores"] = None
        return state

    def __del__(self) -> None:
        self.close()


def _coordinate_index(
    store: GriddedStore,
    dimension: str,
    value: str | int | float,
    *,
    array_path: str | None = None,
) -> tuple[str | None, int | None]:
    coordinate_path = array_path or dimension
    if not store.has_array(coordinate_path):
        raise ValueError(
            f"Selector dimension {dimension!r} has no coordinate array "
            f"{coordinate_path!r}"
        )
    values = store.array(coordinate_path).read()
    if values.ndim == 0:
        if not _selector_values_equal(values.reshape(()).item(), value):
            raise ValueError(
                f"Scalar selector {dimension}={value!r} does not match "
                f"{values.reshape(()).item()!r}"
            )
        return None, None
    if values.ndim != 1:
        raise ValueError(
            f"Selector coordinate {coordinate_path!r} must be scalar or one-dimensional"
        )
    if np.issubdtype(values.dtype, np.number):
        try:
            selected_number = float(value)
        except (TypeError, ValueError):
            matches = np.array([], dtype=np.int64)
        else:
            matches = np.flatnonzero(
                np.isclose(values.astype(float), selected_number)
            )
    else:
        matches = np.flatnonzero(values.astype(str) == str(value))
    if len(matches) != 1:
        raise ValueError(
            f"Selector {dimension}={value!r} matched {len(matches)} coordinate values"
        )
    coordinate_dimensions = store.array(coordinate_path).dimensions
    if len(coordinate_dimensions) != 1:
        raise ValueError(
            f"Selector coordinate {coordinate_path!r} has no single physical dimension"
        )
    return coordinate_dimensions[0], int(matches[0])


def _selector_values_equal(observed: Any, selected: Any) -> bool:
    observed_array = np.asarray(observed)
    if np.issubdtype(observed_array.dtype, np.number):
        try:
            return bool(np.isclose(float(observed), float(selected)))
        except (TypeError, ValueError):
            return False
    return str(observed) == str(selected)
