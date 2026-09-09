"""Evidence and retention invariants for expensive matched experiments."""

import json

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.experiments.generation_matrix import (
    OfflinePreparationAgent,
    RecordingLLMClient,
    remove_owned_payloads,
    sha256,
    verify_freeze,
)


def test_freeze_rejects_changed_source_and_escaping_path(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("original")
    manifest = {"files": [{"path": "input.json", "sha256": sha256(source)}]}
    verify_freeze(manifest, tmp_path)
    source.write_text("edited")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_freeze(manifest, tmp_path)
    with pytest.raises(ValueError):
        verify_freeze(
            {"files": [{"path": "../input.json", "sha256": "0" * 64}]}, tmp_path
        )


def test_cleanup_confines_deletions_and_retains_evidence(tmp_path):
    experiment = tmp_path / "experiment"
    payload = experiment / "candidate/output"
    payload.mkdir(parents=True)
    (payload / "array.bin").write_bytes(b"0123")
    source = experiment / "candidate/pipeline_impl.py"
    source.write_text("pass")
    outside = tmp_path / "trusted_fixture"
    outside.mkdir()
    (outside / "raw").write_bytes(b"source")
    link = experiment / "link"
    link.symlink_to(outside, target_is_directory=True)
    for forbidden in (outside, experiment, link):
        with pytest.raises(ValueError):
            remove_owned_payloads(
                [forbidden], [experiment], experiment / "cleanup.json"
            )
    payload.chmod(0o500)
    remove_owned_payloads([payload], [experiment], experiment / "cleanup.json")
    assert not payload.exists()
    assert source.read_text() == "pass"
    assert (outside / "raw").read_bytes() == b"source"


def test_only_pipeline_preparation_has_network_denied():
    agent = OfflinePreparationAgent({})
    with (
        patch("backend.experiments.generation_matrix.Path.is_file", return_value=True),
        patch(
            "backend.experiments.generation_matrix.PipelineGenerationAgent.repair"
        ) as repair,
    ):
        command = ["python", "run_pipeline.py", "--output-dir", "result"]
        agent.repair(command=command, pipeline_dir=Path("candidate"))
        actual = repair.call_args.kwargs["command"]
        assert actual[:3] == [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1) (allow default) (deny network*)",
        ]
        assert actual[3:] == command
        install = ["python", "-m", "pip", "install", "-r", "requirements.txt"]
        agent.repair(command=install, pipeline_dir=Path("candidate"))
        assert repair.call_args.kwargs["command"] == install


def test_recording_client_can_raise_a_frozen_output_token_allowance(tmp_path):
    client = RecordingLLMClient(
        calls_dir=tmp_path / "calls",
        api_key="test-key",
        model="gpt-test",
        reasoning_effort="xhigh",
        max_output_tokens_override=32768,
    )
    with patch(
        "backend.experiments.generation_matrix.LLMClient.complete_text",
        return_value="complete response",
    ) as complete:
        assert (
            client.complete_text("prompt", max_output_tokens=8192)
            == "complete response"
        )
    assert complete.call_args.kwargs["max_output_tokens"] == 32768
    receipt = json.loads((tmp_path / "calls/call-001.json").read_text())
    assert receipt["request_settings"]["model"] == "gpt-test"
    assert receipt["request_settings"]["reasoning_effort"] == "xhigh"
