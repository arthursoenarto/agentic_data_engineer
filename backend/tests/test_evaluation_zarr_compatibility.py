from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

from backend.evaluation.constrained_schemas import ArrayLocation, CoordinateLocation
from backend.evaluation.gridded_dataset import GriddedTensorDataset
from backend.evaluation.gridded_store import GriddedStore
from backend.evaluation.regular_grid_zarr import (
    UnsupportedGridError,
    _decode_cf_datetime,
    _normalize_coordinate,
    _resolve_fixed_indices,
    inspect_zarr_storage,
    validate_zarr_integrity,
)
from backend.evaluation.schemas import BenchmarkTarget, ChannelSpec, DimensionSpec
from backend.evaluation.target_mapping import resolve_cf_datetime_coordinate


def _create_array(group: object, name: str, data: np.ndarray) -> object:
    create = getattr(group, "create_array", None)
    if create is not None:
        return create(name, data=data, overwrite=True)
    return group.create_dataset(  # type: ignore[attr-defined,no-any-return]
        name,
        data=data,
        shape=data.shape,
        dtype=data.dtype,
        overwrite=True,
    )


class EvaluationZarrCompatibilityTests(unittest.TestCase):
    def test_cf_datetime_ascending_returns_canonical_values_and_source_order(self) -> None:
        values = np.array([86_400.0, 0.0], dtype=np.float64)
        array = SimpleNamespace(
            attrs={"units": "seconds since 1970-01-01"},
        )

        normalized, order = _normalize_coordinate(
            values,
            CoordinateLocation(
                store_path="unused.zarr",
                array_path="valid_time",
                normalization="cf_datetime_ascending",
            ),
            array,
        )

        np.testing.assert_array_equal(order, np.array([1, 0]))
        np.testing.assert_array_equal(
            normalized,
            np.array(
                ["1970-01-01T00:00:00", "1970-01-02T00:00:00"],
                dtype="datetime64[ns]",
            ),
        )

    def test_target_mapping_uses_unique_cf_time_auxiliary_coordinate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store_path = Path(temporary) / "dataset.zarr"
            group = zarr.open_group(store_path, mode="w", zarr_format=3)
            sample = _create_array(group, "sample", np.arange(2, dtype=np.int64))
            sample.attrs["_ARRAY_DIMENSIONS"] = ["sample"]
            valid_time = _create_array(
                group,
                "valid_time",
                np.array([0.0, 86_400.0], dtype=np.float64),
            )
            valid_time.attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["sample"],
                    "standard_name": "time",
                    "units": "seconds since 1970-01-01",
                }
            )
            _create_array(
                group,
                "unrelated_scalar",
                np.array(500, dtype=np.int32),
            )
            zarr.consolidate_metadata(store_path)

            resolved = resolve_cf_datetime_coordinate(
                store_path,
                declared_path="sample",
                sample_dimension="sample",
            )

            self.assertEqual(resolved, "valid_time")

    def test_zarr3_consolidated_storage_diagnostics_partition_footprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store_path = Path(temporary) / "dataset.zarr"
            group = zarr.open_group(store_path, mode="w", zarr_format=3)
            values = _create_array(
                group,
                "temperature",
                np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            )
            values.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
            zarr.consolidate_metadata(store_path)

            diagnostics = inspect_zarr_storage(
                [store_path],
                expected_format=3,
                require_consolidated=True,
                open_latency_repetitions=2,
            )

            self.assertEqual(diagnostics["zarr_format"], 3)
            self.assertTrue(diagnostics["consolidated_metadata"])
            self.assertGreater(diagnostics["metadata_bytes"], 0)
            self.assertGreater(diagnostics["chunk_bytes"], 0)
            self.assertEqual(
                diagnostics["output_bytes"],
                diagnostics["metadata_bytes"] + diagnostics["chunk_bytes"],
            )
            self.assertEqual(
                diagnostics["object_count"],
                diagnostics["metadata_object_count"]
                + diagnostics["chunk_object_count"],
            )
            self.assertEqual(
                diagnostics["dataset_open_latency_seconds"]["repetitions"],
                2,
            )

    def test_datetime_normalization_supports_native_and_iso_string_arrays(self) -> None:
        array = SimpleNamespace(attrs={})
        expected = np.array(
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            dtype="datetime64[ns]",
        )

        native = _decode_cf_datetime(expected.astype("datetime64[s]"), array)
        strings = _decode_cf_datetime(expected.astype(str), array)

        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(strings, expected)

    def test_datetime_normalization_rejects_non_iso_strings(self) -> None:
        with self.assertRaises(UnsupportedGridError):
            _decode_cf_datetime(
                np.array(["not-a-date"]),
                SimpleNamespace(attrs={}),
            )

    def test_zarr2_and_zarr3_common_reader_supports_scalar_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store_path = Path(temporary) / "dataset.zarr"
            group = zarr.open_group(store_path, mode="w")
            nested = group.require_group("observations")
            values = _create_array(
                nested,
                "temperature",
                np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            )
            values.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
            selector = _create_array(
                group,
                "pressure_selector",
                np.array(500, dtype=np.int32),
            )
            selector.attrs["_ARRAY_DIMENSIONS"] = []
            for name, data, dimension in (
                ("time", np.array([0, 1], dtype=np.int64), "time"),
                ("latitude", np.array([52.0, 51.0]), "latitude"),
                ("longitude", np.array([-1.0, 0.0]), "longitude"),
            ):
                coordinate = _create_array(group, name, data)
                coordinate.attrs["_ARRAY_DIMENSIONS"] = [dimension]

            with GriddedStore(store_path, "zarr") as store:
                array = store.array("observations/temperature")
                self.assertEqual(array.dimensions, ("time", "latitude", "longitude"))
                fixed = _resolve_fixed_indices(
                    ArrayLocation(
                        store_path=str(store_path),
                        array_path="observations/temperature",
                        selectors={"pressure_level": "500"},
                        selector_coordinate_paths={
                            "pressure_level": "pressure_selector"
                        },
                    ),
                    store,
                    array,
                    'temperature[pressure_level="500"]',
                )
                self.assertEqual(fixed, {})
                materialized = _resolve_fixed_indices(
                    ArrayLocation(
                        store_path=str(store_path),
                        array_path="observations/temperature",
                        selectors={"pressure_level": "500"},
                    ),
                    store,
                    array,
                    'temperature[pressure_level="500"]',
                )
                self.assertEqual(materialized, {})

            self.assertEqual(
                validate_zarr_integrity([store_path], max_block_bytes=1024),
                5,
            )

            dataset = GriddedTensorDataset(
                BenchmarkTarget(
                    name="scalar-selector",
                    dimensions=DimensionSpec(
                        time="time",
                        y="latitude",
                        x="longitude",
                    ),
                    channels=[
                        ChannelSpec(
                            name='temperature[pressure_level="500"]',
                            store_path=str(store_path),
                            array_path="observations/temperature",
                            selectors={"pressure_level": "500"},
                            selector_coordinate_paths={
                                "pressure_level": "pressure_selector"
                            },
                        )
                    ],
                )
            )
            try:
                self.assertEqual(tuple(dataset[0].shape), (1, 2, 2))
            finally:
                dataset.close()

            materialized_dataset = GriddedTensorDataset(
                BenchmarkTarget(
                    name="materialized-selector",
                    dimensions=DimensionSpec(
                        time="time",
                        y="latitude",
                        x="longitude",
                    ),
                    channels=[
                        ChannelSpec(
                            name='temperature[pressure_level="500"]',
                            store_path=str(store_path),
                            array_path="observations/temperature",
                            selectors={"pressure_level": "500"},
                        )
                    ],
                )
            )
            try:
                self.assertEqual(tuple(materialized_dataset[0].shape), (1, 2, 2))
            finally:
                materialized_dataset.close()

    def test_selector_coordinate_can_use_a_channel_specific_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store_path = Path(temporary) / "dataset.zarr"
            group = zarr.open_group(store_path, mode="w")
            values = _create_array(
                group,
                "temperature_500",
                np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2),
            )
            values.attrs["_ARRAY_DIMENSIONS"] = [
                "time",
                "temperature_500__pressure_level",
                "latitude",
                "longitude",
            ]
            selector = _create_array(
                group,
                "temperature_500__pressure_level",
                np.array([500], dtype=np.int32),
            )
            selector.attrs["_ARRAY_DIMENSIONS"] = [
                "temperature_500__pressure_level"
            ]
            for name, data in (
                ("time", np.array([0, 1], dtype=np.int64)),
                ("latitude", np.array([52.0, 51.0])),
                ("longitude", np.array([-1.0, 0.0])),
            ):
                coordinate = _create_array(group, name, data)
                coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]

            location = ArrayLocation(
                store_path=str(store_path),
                array_path="temperature_500",
                selectors={"pressure_level": "500"},
                selector_coordinate_paths={
                    "pressure_level": "temperature_500__pressure_level"
                },
            )
            with GriddedStore(store_path, "zarr") as store:
                fixed = _resolve_fixed_indices(
                    location,
                    store,
                    store.array("temperature_500"),
                    'temperature[pressure_level="500"]',
                )
            self.assertEqual(fixed, {"temperature_500__pressure_level": 0})

            dataset = GriddedTensorDataset(
                BenchmarkTarget(
                    name="renamed-selector-dimension",
                    dimensions=DimensionSpec(
                        time="time",
                        y="latitude",
                        x="longitude",
                    ),
                    channels=[
                        ChannelSpec(
                            name='temperature[pressure_level="500"]',
                            store_path=str(store_path),
                            array_path="temperature_500",
                            selectors={"pressure_level": "500"},
                            selector_coordinate_paths={
                                "pressure_level": "temperature_500__pressure_level"
                            },
                        )
                    ],
                )
            )
            try:
                self.assertEqual(tuple(dataset[0].shape), (1, 2, 2))
            finally:
                dataset.close()


if __name__ == "__main__":
    unittest.main()
