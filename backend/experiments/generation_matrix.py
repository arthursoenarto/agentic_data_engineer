"""Serial, resumable execution of a frozen matrix through public Stage A APIs.

Dataset facts live in the experiment's config. This wrapper preserves terminal
failures, captures raw generation/repair responses even before schema parsing,
and removes only explicitly owned regenerable payloads after evidence capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.agents.etl_pipeline import (
    DirectLLMStrategy, PipelineGenerationAgent, PipelineVariant, TerraioDirectStrategy,
)
from backend.agents.pipeline_repair import PipelineRepairAgent
from backend.llm import LLMClient, LLMUsageTracker
from backend.file_copy import copytree_isolated
from backend.orchestration.stage_a import StageAWorkflowConfig, run_stage_a_workflow

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_in_root(raw: str, root: Path = ROOT) -> Path:
    path = (root / raw).resolve()
    path.relative_to(root.resolve())
    return path


def verify_freeze(manifest: dict[str, Any], root: Path = ROOT) -> None:
    for record in manifest["files"]:
        path = path_in_root(record["path"], root)
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen input changed or missing: {record['path']}")


class _TeeUsage(LLMUsageTracker):
    def __init__(self, upstream: LLMUsageTracker | None) -> None:
        super().__init__()
        self.upstream = upstream
        self.last_response: dict[str, Any] | None = None

    def record_response(self, response: dict[str, Any], *, requested_model: str) -> None:
        self.last_response = response
        super().record_response(response, requested_model=requested_model)
        if self.upstream is not None:
            self.upstream.record_response(response, requested_model=requested_model)


class RecordingLLMClient(LLMClient):
    """Observe the existing client without changing its request or retry policy."""

    def __init__(
        self,
        *,
        calls_dir: Path,
        max_output_tokens_override: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.calls_dir = calls_dir
        self.max_output_tokens_override = max_output_tokens_override

    def complete_text(self, prompt: str, **kwargs: Any) -> str:
        if self.max_output_tokens_override is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens_override
        tracker = _TeeUsage(kwargs.pop("usage_tracker", None))
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        call_id = len(list(self.calls_dir.glob("call-*.json"))) + 1
        target = self.calls_dir / f"call-{call_id:03d}.json"
        started = time.perf_counter()
        record = {"started_at": datetime.now(UTC).isoformat(), "prompt": prompt,
                  "request_settings": {
                      **kwargs,
                      "model": kwargs.get("model") or self.model,
                      "reasoning_effort": (
                          kwargs.get("reasoning_effort") or self.reasoning_effort
                      ),
                  }, "status": "requesting"}
        write_json(target, record)
        try:
            answer = super().complete_text(prompt, usage_tracker=tracker, **kwargs)
            record.update(status="response_received", response_text=answer)
            if tracker.last_response is not None:
                record["response_model"] = tracker.last_response.get("model")
                record["provider_metadata"] = tracker.last_response.get(
                    "provider_metadata"
                )
            return answer
        except Exception as error:
            # Deliberately exclude exception bodies, which can echo remote payloads.
            record.update(status="request_failed", error_type=type(error).__name__)
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            try:
                record["usage"] = tracker.summary().model_dump(mode="json")
            except RuntimeError:
                record["usage"] = None
            write_json(target, record)


class OfflinePreparationAgent(PipelineGenerationAgent):
    """Deny provider networking for the generated command, preserving pip access."""

    def repair(self, *, command: Any, **kwargs: Any) -> Any:
        command = list(command)
        if "run_pipeline.py" in [Path(value).name for value in command]:
            sandbox = Path("/usr/bin/sandbox-exec")
            if not sandbox.is_file():
                raise RuntimeError("This frozen experiment requires the macOS network-denial wrapper")
            command = [str(sandbox), "-p", "(version 1) (allow default) (deny network*)", *command]
        return super().repair(command=command, **kwargs)


def generation_agent(llm: LLMClient) -> PipelineGenerationAgent:
    return OfflinePreparationAgent(
        {PipelineVariant.DIRECT_LLM: DirectLLMStrategy(llm),
         PipelineVariant.TERRAIO_DIRECT: TerraioDirectStrategy(llm)},
        repair_agent=PipelineRepairAgent(llm),
    )


def trial_record(item: dict[str, Any], config: StageAWorkflowConfig, report_path: Path,
                 calls_dir: Path) -> dict[str, Any]:
    report = read_json(report_path)
    if report["status"] == "running" or report.get("completed_at") is None:
        raise RuntimeError(f"Trial has no terminal report: {report_path}")
    record = {"trial_id": item["trial_id"], "dataset": item["dataset"],
              "condition": config.condition.value, "repetition": item["repetition"],
              "status": report["status"], "optimization_ready": report["optimization_ready"],
              "repairs": report["repair_attempts_used"], "stage_a_report": str(report_path.relative_to(ROOT)),
              "evaluation_report": report.get("evaluation_report"), "error": report.get("error"),
              "lineage_unchanged": report.get("lineage_unchanged"),
              "framework_source_sha256": report["framework_source_sha256"],
              "pipeline_id": config.pipeline_id, "runtime_freeze": report.get("runtime_freeze")}
    for key in (
        "provider",
        "display_name",
        "configuration_id",
        "model",
        "reasoning_effort",
        "max_output_tokens",
        "release_month",
        "family_release_month",
        "attempt",
    ):
        if key in item:
            record[key] = item[key]
    calls = [read_json(p) for p in sorted(calls_dir.glob("call-*.json"))]
    generation_cost = repair_cost = 0.0
    for index, call in enumerate(calls):
        cost = float((call.get("usage") or {}).get("estimated_cost_usd", 0))
        if index == 0:
            generation_cost += cost
        else:
            repair_cost += cost
    record["generation_repair_calls"] = len(calls)
    record["calls_without_usage"] = sum(call.get("usage") is None for call in calls)
    judge_cost = 0.0
    if report.get("evaluation_report"):
        evaluation = read_json(path_in_root(report["evaluation_report"]))
        summary = evaluation.get("summary", {})
        record["evaluation_summary"] = summary
        record["objectives"] = {
            **summary.get("operational_objectives", {}),
            **summary.get("engineering_objective", {}),
        }
        judge = evaluation.get("engineering_quality", {}) or {}
        judge_cost = float((judge.get("usage") or {}).get("estimated_cost_usd", 0))
    record["llm_cost_usd"] = {"generation": generation_cost, "repair": repair_cost,
                              "judge": judge_cost, "total_measured": generation_cost + repair_cost + judge_cost}
    record["success_without_repair"] = record["optimization_ready"] and record["repairs"] == 0
    start = datetime.fromisoformat(report["started_at"])
    stop = datetime.fromisoformat(report["completed_at"])
    record["wall_seconds"] = (stop - start).total_seconds()
    return record


def remove_owned_payloads(paths: list[Path], allowed_roots: list[Path], log_path: Path) -> None:
    """Delete an explicit list only after confinement and symlink validation."""
    entries = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        resolved = path.resolve()
        if path.is_symlink() or not any(resolved.is_relative_to(root.resolve()) and
                                       resolved != root.resolve() for root in allowed_roots):
            raise ValueError(f"Cleanup path escapes owned subtree: {path}")
        entries.append({"path": str(path), "logical_bytes": sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()),
            "reason": "Regenerable payload; compact trial/source/receipt/measurement evidence retained"})
    manifest = {"created_at": datetime.now(UTC).isoformat(), "status": "planned", "entries": entries}
    write_json(log_path, manifest)
    for entry in entries:
        for directory, _, _ in os.walk(entry["path"], followlinks=False):
            if not Path(directory).is_symlink():
                os.chmod(directory, Path(directory).stat().st_mode | 0o700)
        shutil.rmtree(entry["path"])
    manifest["status"] = "completed"
    manifest["deleted_logical_bytes"] = sum(entry["logical_bytes"] for entry in entries)
    write_json(log_path, manifest)


def cleanup_trial(config: StageAWorkflowConfig, experiment: Path, log_path: Path) -> None:
    workspace = config.dataset_dir.resolve()
    workspace.relative_to(experiment.resolve())
    pipeline = workspace / "pipelines" / config.pipeline_id
    stage = workspace / "runs/stage_a" / config.workflow_id
    runtime_parent = ROOT / "tmp/venvs/stage_a"
    runtime = runtime_parent / config.workflow_id
    paths = [runtime, *pipeline.glob("acceptance/*/frozen_cache")]
    if config.candidate_cache is not None:
        config.candidate_cache.resolve().relative_to(experiment.resolve())
        paths.append(config.candidate_cache.resolve())
    for owned in (pipeline / "runs", stage, config.evaluation_output_dir.resolve()):
        if owned.exists():
            paths.extend(p for p in owned.rglob("output") if p.is_dir())
    for materializations in config.evaluation_output_dir.resolve().rglob("materializations"):
        evidence = log_path.parent / "receipts" / config.pipeline_id
        for receipt in materializations.rglob("pipeline_run.json"):
            if "output" not in receipt.relative_to(materializations).parts:
                target = evidence / receipt.relative_to(materializations)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(receipt, target)
        paths.append(materializations)
    # Only maximal directories are removed; preserve all receipts and logs beside them.
    paths = [p for p in paths if not any(p != q and p.is_relative_to(q) for q in paths)]
    remove_owned_payloads(paths, [experiment, runtime_parent], log_path)


def run_matrix(config_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    matrix = read_json(config_path)
    experiment = config_path.resolve().parent
    freeze = read_json(experiment / "freeze.json")
    verify_freeze(freeze)
    import fcntl
    with (experiment / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        executed = 0
        for item in matrix["trials"]:
            terminal = experiment / "trials" / f"{item['trial_id']}.json"
            config = StageAWorkflowConfig.model_validate_json(path_in_root(item["config"]).read_text())
            calls_dir = experiment / "calls" / item["trial_id"]
            if terminal.is_file():
                records.append(read_json(terminal))
                continue
            prerequisite = item.get("run_if_trial_failed")
            if prerequisite is not None:
                prerequisite_path = experiment / "trials" / f"{prerequisite}.json"
                if not prerequisite_path.is_file():
                    raise RuntimeError(
                        f"Conditional trial prerequisite is not terminal: {prerequisite}"
                    )
                prerequisite_record = read_json(prerequisite_path)
                if prerequisite_record.get("evaluation_summary", {}).get("feasible") is True:
                    record = {
                        "trial_id": item["trial_id"],
                        "dataset": item["dataset"],
                        "condition": config.condition.value,
                        "repetition": item["repetition"],
                        "status": "skipped",
                        "optimization_ready": False,
                        "repairs": 0,
                        "calls_without_usage": 0,
                        "llm_cost_usd": {
                            "generation": 0.0,
                            "repair": 0.0,
                            "judge": 0.0,
                            "total_measured": 0.0,
                        },
                        "success_without_repair": False,
                        "skip_reason": "first_attempt_passed_hard_gates",
                        "selected_trial_id": prerequisite,
                        **{
                            key: item[key]
                            for key in (
                                "provider", "display_name", "configuration_id",
                                "model", "reasoning_effort", "max_output_tokens",
                                "release_month", "family_release_month", "attempt",
                            )
                            if key in item
                        },
                    }
                    write_json(terminal, record)
                    records.append(record)
                    write_json(
                        experiment / "results.json",
                        {
                            "complete": len(records) == len(matrix["trials"]),
                            "expected_trials": len(matrix["trials"]),
                            "trials": records,
                        },
                    )
                    print(
                        f"SKIP {item['trial_id']} first attempt passed hard gates",
                        flush=True,
                    )
                    continue
            if limit is not None and executed >= limit:
                break
            if sum(r["llm_cost_usd"]["total_measured"] for r in records) >= matrix["stop_before_trial_cost_usd"]:
                raise RuntimeError("Frozen measured-cost stopping threshold reached")
            if shutil.disk_usage(experiment).free < 10 * 1024**3:
                raise RuntimeError("Less than 10 GiB free before trial")
            verify_freeze(freeze)
            report_path = config.dataset_dir.resolve() / "runs/stage_a" / config.workflow_id / "stage_a_run.json"
            if report_path.exists():
                if read_json(report_path)["status"] == "running":
                    raise RuntimeError(f"Interrupted or live trial requires inspection: {item['trial_id']}")
            else:
                print(f"START {item['trial_id']} ({len(records)+1}/{len(matrix['trials'])})", flush=True)
                assert config.candidate_cache is not None
                copytree_isolated(path_in_root(matrix["datasets"][item["dataset"]]["fixture"]),
                                  config.candidate_cache.resolve())
                llm = RecordingLLMClient(
                    calls_dir=calls_dir,
                    model=item.get("model", matrix["model"]),
                    reasoning_effort=item.get("reasoning_effort"),
                    max_output_tokens_override=item.get("max_output_tokens"),
                    timeout_seconds=config.llm_timeout_seconds,
                )
                run_stage_a_workflow(config, generation_agent=generation_agent(llm))
                executed += 1
            verify_freeze(freeze)
            record = trial_record(item, config, report_path, calls_dir)
            write_json(terminal, record)
            records.append(record)
            write_json(experiment / "results.json", {"complete": len(records) == len(matrix["trials"]),
                       "expected_trials": len(matrix["trials"]), "trials": records})
            cleanup_trial(config, experiment, experiment / "cleanup" / f"{item['trial_id']}.json")
            print(f"DONE {item['trial_id']} status={record['status']} ready={record['optimization_ready']} "
                  f"repairs={record['repairs']} cost=${record['llm_cost_usd']['total_measured']:.3f}", flush=True)
            if record["status"] == "blocked" or (record["calls_without_usage"] and not record["optimization_ready"]):
                raise RuntimeError("Infrastructure or unbilled API failure: inspect terminal evidence before continuing")
        result = {"complete": len(records) == len(matrix["trials"]),
                  "expected_trials": len(matrix["trials"]), "trials": records}
        write_json(experiment / "results.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run_matrix(args.config, limit=args.limit)
    print(json.dumps({"complete": result["complete"], "terminal_trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
