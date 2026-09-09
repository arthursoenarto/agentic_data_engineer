from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend.contracts.output_policy import StationTimeSeriesParquetOutputPolicy
from backend.evaluation.station_parquet import (
    _compare_tables,
    _measure_filtered_station_scan,
    _validate_parquet_file,
)
from backend.evaluation.station_parquet_schemas import StationWorkloadSpec


class StationParquetEvaluationTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            output_policy=StationTimeSeriesParquetOutputPolicy(),
            table=SimpleNamespace(value_atol=0.0, value_rtol=0.0),
        )

    def test_logical_comparison_normalizes_string_and_timestamp_units(self) -> None:
        candidate = pa.table(
            {
                "station_id": pa.array(["station"], type=pa.large_string()),
                "timestamp": pa.array(
                    [0], type=pa.timestamp("ms", tz="UTC")
                ),
                "field_id": pa.array(["temperature"], type=pa.large_string()),
                "value": pa.array([12.5], type=pa.float64()),
            }
        )
        reference = pa.table(
            {
                "station_id": pa.array(["station"], type=pa.string()),
                "timestamp": pa.array(
                    [0], type=pa.timestamp("us", tz="UTC")
                ),
                "field_id": pa.array(["temperature"], type=pa.string()),
                "value": pa.array([12.5], type=pa.float64()),
            }
        )

        evidence = _compare_tables(candidate, reference, config=self._config())

        self.assertEqual(evidence["rows_compared"], 1)
        self.assertEqual(evidence["value_mismatches"], 0)

    def test_missingness_mismatch_is_rejected(self) -> None:
        base = {
            "station_id": ["station"],
            "timestamp": pa.array([0], type=pa.timestamp("us", tz="UTC")),
            "field_id": ["temperature"],
        }
        candidate = pa.table({**base, "value": [np.nan]})
        reference = pa.table({**base, "value": [12.5]})

        with self.assertRaisesRegex(ValueError, "Missingness differs"):
            _compare_tables(candidate, reference, config=self._config())

    def test_physical_validation_accepts_large_string_columns(self) -> None:
        table = pa.table(
            {
                "station_id": pa.array(["station"], type=pa.large_string()),
                "timestamp": pa.array(
                    [0], type=pa.timestamp("us", tz="UTC")
                ),
                "field_id": pa.array(["temperature"], type=pa.large_string()),
                "value": pa.array([12.5], type=pa.float64()),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.parquet"
            pq.write_table(table, path, compression="zstd")
            _validate_parquet_file(path, config=self._config())

    def test_v2_workload_pushes_station_and_field_filters_into_parquet(self) -> None:
        table = pa.table(
            {
                "station_id": ["a", "a", "b"],
                "timestamp": pa.array([0, 1, 2], type=pa.timestamp("us", tz="UTC")),
                "field_id": ["pm25", "pm25", "pm25"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        config = SimpleNamespace(
            output_policy=StationTimeSeriesParquetOutputPolicy(),
            table=SimpleNamespace(
                expected_station_ids=["a", "b"],
                expected_field_ids=["pm25"],
            ),
            workload=StationWorkloadSpec(
                protocol="filtered_station_scan.v2",
                repetitions=1,
                warmup_passes=0,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.parquet"
            pq.write_table(table, path, compression="zstd", row_group_size=1)

            evidence = _measure_filtered_station_scan(path, config=config)

        self.assertEqual(evidence["protocol"], "filtered_station_scan.v2")
        self.assertEqual(evidence["matched_rows"], 2)
        self.assertEqual(evidence["repetitions"], 1)


if __name__ == "__main__":
    unittest.main()
