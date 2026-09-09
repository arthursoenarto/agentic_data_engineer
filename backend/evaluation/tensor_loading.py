"""Opt-in, versioned CPU tensor delivery benchmark for eager grid readers.

This module does not alter the historical objective-v3 consumer. Scientific
equivalence must be established independently before publishing measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch


class SampleReader(Protocol):
    def sample(self, index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class TensorLoadingProtocol:
    sample_count: int
    sample_shape: tuple[int, int, int]
    seed: int = 20260907
    warmup_passes: int = 1
    timed_passes: int = 5

    def __post_init__(self) -> None:
        if (
            self.sample_count < 1
            or len(self.sample_shape) != 3
            or min(self.sample_shape) < 1
        ):
            raise ValueError("A positive sample count and [C,H,W] shape are required")
        if self.warmup_passes < 1 or self.timed_passes < 1:
            raise ValueError("At least one warm-up and one timed pass are required")

    def identity(self) -> dict[str, Any]:
        definition = {
            **asdict(self),
            "sample_shape": list(self.sample_shape),
            "schema_version": "tensor_loading.v1",
            "dtype": "float32",
            "device": "cpu",
            "layout": "C-contiguous",
            "cache": "warm; OS page cache not flushed",
            "consumer_workers": 0,
            "reduction": "one scalar marker; no full-field sum or model compute",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        digest = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()
        return {**definition, "protocol_sha256": digest}


def deliver_tensor(
    reader: SampleReader, index: int, shape: tuple[int, int, int]
) -> torch.Tensor:
    """Include native read, decoding, channel/coordinate mapping and packing."""
    array = reader.sample(index)
    if not isinstance(array, np.ndarray):
        raise ValueError("Reader must return an eager NumPy array")
    if array.shape != shape or array.dtype != np.dtype("float32"):
        raise ValueError(
            f"Wrong logical sample: {array.shape}/{array.dtype}; expected {shape}/float32"
        )
    # A read-only buffer cannot safely serve as a mutable PyTorch tensor.
    contiguous = np.ascontiguousarray(array)
    if not contiguous.flags.writeable:
        contiguous = contiguous.copy()
    return torch.from_numpy(contiguous)


def measure_tensor_loading(
    reader: SampleReader, protocol: TensorLoadingProtocol
) -> dict[str, Any]:
    """One full warm-up and fixed full passes; no adaptive winner selection."""
    order = (
        np.random.default_rng(protocol.seed).permutation(protocol.sample_count).tolist()
    )

    def full_pass() -> dict[str, float]:
        marker = 0.0
        started = time.perf_counter()
        for index in order:
            tensor = deliver_tensor(reader, index, protocol.sample_shape)
            marker += float(tensor[0, 0, 0])
            # Release each delivered sample within the timer. Do not cache
            # decoded samples in the harness or retain a complete epoch.
            del tensor
        seconds = time.perf_counter() - started
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid timer duration")
        return {
            "seconds": seconds,
            "samples_per_second": protocol.sample_count / seconds,
            "marker": marker,
        }

    warmup = [full_pass() for _ in range(protocol.warmup_passes)]
    passes = [full_pass() for _ in range(protocol.timed_passes)]
    values = [item["samples_per_second"] for item in passes]
    q1, q3 = np.percentile(values, [25, 75], method="linear").tolist()
    return {
        "protocol": protocol.identity(),
        "sample_order": order,
        "warmup": warmup,
        "timed_passes": passes,
        "summary": {
            "median_samples_per_second": statistics.median(values),
            "minimum_samples_per_second": min(values),
            "maximum_samples_per_second": max(values),
            "q1_samples_per_second": q1,
            "q3_samples_per_second": q3,
            "iqr_samples_per_second": q3 - q1,
            "timing_repetitions": len(values),
            "independent_generations": None,
        },
    }


def validate_matched_loading(records: list[dict[str, Any]]) -> None:
    """Reject a combined table containing mixed protocols or reader runtimes."""
    if not records:
        raise ValueError("No eligible measurements")
    first = records[0]
    for record in records:
        consumer = record["consumer"]
        if consumer["protocol"] != first["consumer"]["protocol"]:
            raise ValueError("Mixed tensor-loading protocols")
        if record["reader_runtime"] != first["reader_runtime"]:
            raise ValueError("Mixed reader runtimes or concurrency settings")
        if consumer["sample_order"] != first["consumer"]["sample_order"]:
            raise ValueError("Mixed sample orders")
        protocol = consumer["protocol"]
        if sorted(consumer["sample_order"]) != list(range(protocol["sample_count"])):
            raise ValueError("Consumer did not cover the full sample population")
        if (
            len(consumer["timed_passes"]) != protocol["timed_passes"]
            or len(consumer["warmup"]) != protocol["warmup_passes"]
        ):
            raise ValueError("Incomplete warm-up or measurement repetitions")


def capture_zarr_metadata(stores: list[Path], *, output_root: Path) -> dict[str, Any]:
    """Retain native codecs, dtypes, chunks and shards before payload cleanup."""
    metadata: dict[str, Any] = {}
    root = output_root.resolve()
    for store in stores:
        store.resolve().relative_to(root)
        for path in sorted(store.rglob("*")):
            if path.name in {
                "zarr.json",
                ".zarray",
                ".zattrs",
                ".zgroup",
                ".zmetadata",
            }:
                path.resolve().relative_to(root)
                metadata[str(path.relative_to(output_root))] = json.loads(
                    path.read_text()
                )
    return metadata
