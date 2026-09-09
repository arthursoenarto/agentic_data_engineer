"""Small shared types used by constrained evaluation adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


Scalar = str | int | float


class ChannelSpec(BaseModel):
    """One logical field-selector combination in a physical array."""

    name: str
    store_path: str
    array_path: str
    storage_format: Literal["zarr", "netcdf"] = "zarr"
    selectors: dict[str, Scalar] = Field(default_factory=dict)
    indices: dict[str, int] = Field(default_factory=dict)
    selector_coordinate_paths: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def selector_dimensions_must_be_unique(self) -> "ChannelSpec":
        overlap = set(self.selectors) & set(self.indices)
        if overlap:
            raise ValueError(
                f"Dimensions cannot use both selectors and indices: {sorted(overlap)}"
            )
        unknown = set(self.selector_coordinate_paths) - set(self.selectors)
        if unknown:
            raise ValueError(
                "selector_coordinate_paths has no matching selector for "
                f"{sorted(unknown)}"
            )
        return self


class DimensionSpec(BaseModel):
    """Physical dimension names for sample and spatial roles."""

    time: str
    y: str
    x: str

    @model_validator(mode="after")
    def dimensions_must_be_distinct(self) -> "DimensionSpec":
        if len({self.time, self.y, self.x}) != 3:
            raise ValueError("time, y, and x dimensions must be distinct")
        return self


class BenchmarkTarget(BaseModel):
    """A regular-grid store plus its explicit channel mapping."""

    name: str
    description: str | None = None
    dimensions: DimensionSpec
    channels: list[ChannelSpec] = Field(min_length=1)


class CheckStatus(StrEnum):
    """Machine-readable outcome for one evaluation check."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_ASSESSED = "not_assessed"


class CheckConfidence(StrEnum):
    """Strength of evidence supporting an evaluation check."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
