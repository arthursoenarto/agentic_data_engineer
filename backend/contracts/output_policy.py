"""Shared public output policies for generation and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CoordinateMetadataRequirements(BaseModel):
    """Coordinate attributes that are part of the public artifact contract."""

    model_config = ConfigDict(extra="forbid")

    sample: dict[str, Any] = Field(default_factory=dict)
    y: dict[str, Any] = Field(default_factory=dict)
    x: dict[str, Any] = Field(default_factory=dict)

    def for_role(self, role: Literal["sample", "y", "x"]) -> dict[str, Any]:
        """Return a copy of the required attributes for one logical role."""

        return dict(getattr(self, role))


class RegularGridZarrOutputPolicy(BaseModel):
    """Public representation contract for regular-grid Zarr artifacts.

    Logical dimensions, coordinates, requested channels, and values are checked
    independently. Only attributes listed in ``coordinate_metadata`` are hard
    representation requirements; other descriptive attributes remain optional.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["regular_grid_zarr_output_policy.v1"] = (
        "regular_grid_zarr_output_policy.v1"
    )
    dataset_class: Literal["regular_rectilinear_grid"] = "regular_rectilinear_grid"
    format_version: Literal[3] = 3
    consolidated_metadata: Literal[True] = True
    coordinate_metadata: CoordinateMetadataRequirements = Field(
        default_factory=CoordinateMetadataRequirements
    )


class StationTimeSeriesParquetOutputPolicy(BaseModel):
    """Canonical long-form Parquet contract for station observations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["station_time_series_parquet_output_policy.v1"] = (
        "station_time_series_parquet_output_policy.v1"
    )
    dataset_class: Literal["station_time_series"] = "station_time_series"
    format_version: Literal[1] = 1
    compression: Literal["zstd"] = "zstd"
    required_columns: list[str] = Field(
        default_factory=lambda: ["station_id", "timestamp", "field_id", "value"]
    )
    primary_key: list[str] = Field(
        default_factory=lambda: ["station_id", "timestamp", "field_id"]
    )
    timestamp_timezone: Literal["UTC"] = "UTC"
