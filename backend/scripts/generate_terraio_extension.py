"""Generate one PR-style TerraIO extension proposal in an isolated clone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import (  # noqa: E402
    ExtensionContract,
    RepositoryExtensionAgent,
    TerraioExtensionStrategy,
)
from backend.llm import LLMClient  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and persist a typed TerraIO repository-extension proposal "
            "without modifying the authoritative checkout."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    contract = ExtensionContract.model_validate_json(
        arguments.extension_contract.read_text(encoding="utf-8")
    )
    build = RepositoryExtensionAgent(
        TerraioExtensionStrategy(LLMClient(timeout_seconds=arguments.timeout_seconds))
    ).build(
        contract=contract,
        dataset_dir=arguments.dataset_dir,
    )
    usage = build.manifest.llm_usage
    print(f"Extension ID: {build.extension_id}")
    print(f"Extension directory: {build.extension_dir}")
    print(f"Patch SHA-256: {build.manifest.patch_sha256}")
    print(
        f"Generation: {usage.total_tokens} tokens across {usage.call_count} call(s), "
        f"estimated ${usage.estimated_cost_usd:.6f}"
    )
