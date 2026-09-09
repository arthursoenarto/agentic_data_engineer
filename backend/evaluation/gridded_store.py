"""Format-neutral read access for evaluator-registered gridded artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from backend.evaluation.schemas import ChannelSpec


class GriddedArray:
    """Small common interface over a Zarr or NetCDF variable."""

    def __init__(self, array: Any, *, storage_format: str, array_path: str) -> None:
        self._array = array
        self.storage_format = storage_format
        self.array_path = array_path

    @property
    def dimensions(self) -> tuple[str, ...]:
        if self.storage_format == "netcdf":
            return tuple(str(value) for value in self._array.dimensions)
        metadata = getattr(self._array, "metadata", None)
        names = getattr(metadata, "dimension_names", None)
        if names is None:
            names = self._array.attrs.get("_ARRAY_DIMENSIONS")
        if not names:
            raise ValueError(
                f"Zarr array {self.array_path!r} has no dimension-name metadata"
            )
        return tuple(str(value) for value in names)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._array.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self._array.dtype)

    @property
    def chunks(self) -> tuple[int, ...] | None:
        if self.storage_format == "zarr":
            return tuple(int(value) for value in self._array.chunks)
        chunking = self._array.chunking()
        if chunking == "contiguous":
            return None
        return tuple(int(value) for value in chunking)

    @property
    def shards(self) -> tuple[int, ...] | None:
        shards = getattr(self._array, "shards", None)
        if self.storage_format != "zarr" or shards is None:
            return None
        return tuple(int(value) for value in shards)

    @property
    def attrs(self) -> dict[str, Any]:
        if self.storage_format == "zarr":
            return dict(self._array.attrs)
        return {name: self._array.getncattr(name) for name in self._array.ncattrs()}

    @property
    def fill_value(self) -> Any:
        if self.storage_format == "zarr":
            return self._array.fill_value
        return getattr(self._array, "_FillValue", None)

    def read(self, selection: tuple[int | slice, ...] | None = None) -> np.ndarray:
        if selection is None:
            selection = tuple(slice(None) for _ in self.shape)
        return np.asarray(self._array[selection])

    def read_orthogonal(self, selection: tuple[Any, ...]) -> np.ndarray:
        """Read independent indices along multiple axes without loading full axes."""

        if self.storage_format != "zarr":
            raise ValueError("Orthogonal indexed reads are implemented only for Zarr")
        return np.asarray(self._array.oindex[selection])


class GriddedStore:
    """Open one explicitly typed gridded store without inferring its format."""

    def __init__(self, path: Path | str, storage_format: str) -> None:
        self.path = Path(path)
        self.storage_format = storage_format
        self._handle: Any
        if storage_format == "zarr":
            self._handle = zarr.open_group(self.path, mode="r")
        elif storage_format == "netcdf":
            try:
                import netCDF4
            except ImportError as error:
                raise RuntimeError(
                    "NetCDF evaluation requires the benchmark extra 'netCDF4'."
                ) from error
            self._handle = netCDF4.Dataset(self.path, mode="r")
            self._handle.set_auto_mask(False)
        else:
            raise ValueError(f"Unsupported gridded storage format: {storage_format}")

    def has_array(self, array_path: str) -> bool:
        try:
            self.array(array_path)
        except (IndexError, KeyError):
            return False
        return True

    def array(self, array_path: str) -> GriddedArray:
        if self.storage_format == "zarr":
            return GriddedArray(
                self._handle[array_path],
                storage_format=self.storage_format,
                array_path=array_path,
            )

        parts = [part for part in array_path.strip("/").split("/") if part]
        if not parts:
            raise KeyError(array_path)
        group = self._handle
        for part in parts[:-1]:
            group = group.groups[part]
        return GriddedArray(
            group.variables[parts[-1]],
            storage_format=self.storage_format,
            array_path=array_path,
        )

    def close(self) -> None:
        if self.storage_format == "netcdf" and self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "GriddedStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_channel_store(channel: ChannelSpec) -> GriddedStore:
    """Open the store declared by a channel mapping."""

    return GriddedStore(channel.store_path, channel.storage_format)
