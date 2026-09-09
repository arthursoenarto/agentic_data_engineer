"""Run a generated pipeline with at most three LLM-guided repair attempts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.pipeline_repair import PipelineRepairAgent  # noqa: E402
from backend.env import ENV_FILE  # noqa: E402
from backend.llm import LLMClient  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute and minimally repair a generated pipeline.",
    )
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--prompt-name", default="default")
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--llm-timeout-seconds", type=int, default=300)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    parser.add_argument(
        "--repair-root",
        type=Path,
        help="Optional immutable run-local directory for repair logs.",
    )
    parser.add_argument(
        "--protect-family-interface",
        action="store_true",
        help="Prevent repairs to framework-owned family runner and provenance files.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Direct command to execute; place it after --.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("A pipeline command is required after --.")
    return args


if __name__ == "__main__":
    arguments = _args()
    repair_log, repair_log_path = PipelineRepairAgent(
        LLMClient(timeout_seconds=arguments.llm_timeout_seconds)
    ).repair_pipeline(
        pipeline_dir=arguments.pipeline_dir,
        command=arguments.command,
        prompt_name=arguments.prompt_name,
        max_attempts=arguments.max_attempts,
        timeout_seconds=arguments.timeout_seconds,
        env_path=arguments.env_file,
        repair_root=arguments.repair_root,
        protected_paths=(
            {
                "manifest.json",
                "pipeline_contract.json",
                "pipeline_run.json",
                "run_pipeline.py",
            }
            if arguments.protect_family_interface
            else None
        ),
    )
    usage = repair_log.aggregate_llm_usage
    print(f"Repair status: {repair_log.final_status}")
    print(f"Repair log: {repair_log_path}")
    if usage:
        print(
            "Repair LLM usage: "
            f"{usage.total_tokens} total tokens across {usage.call_count} call(s), "
            f"estimated ${usage.estimated_cost_usd:.6f}"
        )
    raise SystemExit(
        0
        if repair_log.final_status in {"already_succeeded", "repaired"}
        else (2 if repair_log.final_status == "non_candidate_failure" else 1)
    )
