"""Draft a user-provided dataset candidate into a project DatasetContract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.contract_drafting.workflow import draft_candidate_file


DEFAULT_INPUT = ROOT / "project/datasets/openaq/candidate.json"
DEFAULT_DATASETS_DIR = ROOT / "project/datasets"


def draft_contract(
    *,
    input_path: Path = DEFAULT_INPUT,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    prompt_name: str = "default",
    allow_llm_inventory_fallback: bool = False,
) -> Path:
    run = draft_candidate_file(
        input_path,
        datasets_dir=datasets_dir,
        prompt_name=prompt_name,
        allow_llm_inventory_fallback=allow_llm_inventory_fallback,
        root=ROOT,
    )
    output_path = ROOT / run.contract_file
    print(f"Saved dataset contract to {output_path.relative_to(ROOT)}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft a dataset contract into /project.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a DatasetCandidate JSON file.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR, help="Project datasets directory.")
    parser.add_argument("--prompt-name", default="default", help="Prompt variant name under the agent prompts folder.")
    parser.add_argument(
        "--llm-inventory-fallback",
        action="store_true",
        help="Use the generic web inventory agent when no deterministic extractor can handle the candidate.",
    )
    args = parser.parse_args()
    draft_contract(
        input_path=args.input,
        datasets_dir=args.datasets_dir,
        prompt_name=args.prompt_name,
        allow_llm_inventory_fallback=args.llm_inventory_fallback,
    )
