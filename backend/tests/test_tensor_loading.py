"""Contract tests for the additive thesis tensor-delivery measurement."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from backend.evaluation.tensor_loading import (
    TensorLoadingProtocol,
    capture_zarr_metadata,
    deliver_tensor,
    measure_tensor_loading,
    validate_matched_loading,
)


class Reader:
    def __init__(self, array):
        self.array, self.calls = array, []

    def sample(self, index):
        self.calls.append(index)
        return self.array


def test_delivered_tensor_preserves_values_and_packs_negative_strides():
    array = np.arange(24, dtype=np.float32).reshape(3, 2, 4)[:, ::-1, ::-1]
    tensor = deliver_tensor(Reader(array), 0, (3, 2, 4))
    assert tensor.is_contiguous() and tensor.device.type == "cpu"
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_read_only_buffers_are_copied_and_writable_buffers_share_memory():
    array = np.zeros((1, 2, 3), dtype=np.float32)
    assert deliver_tensor(Reader(array), 0, array.shape).data_ptr() == array.ctypes.data
    array.flags.writeable = False
    tensor = deliver_tensor(Reader(array), 0, array.shape)
    tensor[0, 0, 0] = 7
    assert array[0, 0, 0] == 0


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((1, 2, 3), dtype=np.float64),
        np.zeros((1, 2, 4), dtype=np.float32),
        [[1, 2, 3]],
    ],
)
def test_incompatible_samples_are_rejected_instead_of_silently_cast(array):
    with pytest.raises(ValueError):
        deliver_tensor(Reader(array), 0, (1, 2, 3))


def test_complete_fixed_passes_exclude_warmup_and_full_tensor_reductions(monkeypatch):
    protocol = TensorLoadingProtocol(4, (1, 2, 3))
    reader = Reader(np.ones(protocol.sample_shape, dtype=np.float32))
    # Warmup is 100 s; timed passes are 1,2,3,4,5 s. Verify it cannot
    # enter the median and verify no full reduction is accidentally timed.
    ticks = iter([0, 100, 100, 101, 101, 103, 103, 106, 106, 110, 110, 115])
    monkeypatch.setattr(
        "backend.evaluation.tensor_loading.time.perf_counter", lambda: next(ticks)
    )
    monkeypatch.setattr(
        torch.Tensor, "sum", lambda *a, **kw: pytest.fail("full reduction")
    )
    result = measure_tensor_loading(reader, protocol)
    assert reader.calls == result["sample_order"] * 6
    assert sorted(result["sample_order"]) == list(range(4))
    assert result["summary"]["median_samples_per_second"] == 4 / 3
    assert result["summary"]["minimum_samples_per_second"] == 4 / 5
    assert result["summary"]["maximum_samples_per_second"] == 4
    assert result["summary"]["timing_repetitions"] == 5


@pytest.mark.parametrize("change", ["runtime", "order", "protocol", "passes"])
def test_combining_unmatched_measurements_fails(change):
    consumer = measure_tensor_loading(
        Reader(np.zeros((1, 2, 3), dtype=np.float32)),
        TensorLoadingProtocol(4, (1, 2, 3)),
    )
    a = {"consumer": consumer, "reader_runtime": {"freeze": "abc", "threads": 1}}
    b = deepcopy(a)
    if change == "runtime":
        b["reader_runtime"]["threads"] = 2
    if change == "order":
        b["consumer"]["sample_order"].reverse()
    if change == "protocol":
        b["consumer"]["protocol"]["seed"] += 1
    if change == "passes":
        b["consumer"]["timed_passes"].pop()
    with pytest.raises(ValueError):
        validate_matched_loading([a, b])
    validate_matched_loading([a, deepcopy(a)])


def test_metadata_preserves_codec_and_shard_details_without_chunk_payloads(tmp_path):
    store = tmp_path / "native.zarr"
    store.mkdir()
    metadata = {"zarr_format": 3, "codecs": [{"name": "sharding_indexed"}]}
    (store / "zarr.json").write_text(json.dumps(metadata))
    (store / "c").write_bytes(b"payload")
    assert capture_zarr_metadata([store], output_root=tmp_path) == {
        "native.zarr/zarr.json": metadata
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_count": 0},
        {"warmup_passes": 0},
        {"timed_passes": 0},
        {"sample_shape": (1, 0, 3)},
    ],
)
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        TensorLoadingProtocol(
            **{"sample_count": 4, "sample_shape": (1, 2, 3), **kwargs}
        )
