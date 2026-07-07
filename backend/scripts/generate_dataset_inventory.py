"""Generate project/datasets/{slug}/dataset_inventory.json from a DatasetCandidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.dataset_inventory import build_candidate_file_inventory


DEFAULT_INPUT = ROOT / "project/datasets/reanalysis_era5_pressure_levels/candidate.json"
DEFAULT_DATASETS_DIR = ROOT / "project/datasets"


def _display_path(path: Path) -> Path:
    absolute = path if path.is_absolute() else ROOT / path
    try:
        return absolute.relative_to(ROOT)
    except ValueError:
        return path


def generate_inventory(
    *,
    input_path: Path = DEFAULT_INPUT,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    allow_llm_fallback: bool = False,
) -> Path:
    _inventory, output_path = build_candidate_file_inventory(
        input_path,
        datasets_dir=datasets_dir,
        root=ROOT,
        allow_llm_fallback=allow_llm_fallback,
    )
    print(f"Saved dataset inventory to {_display_path(output_path)}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset inventory into /project.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a DatasetCandidate JSON file.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR, help="Project datasets directory.")
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        help="Use the generic web inventory agent when no deterministic extractor can handle the candidate.",
    )
    args = parser.parse_args()
    generate_inventory(
        input_path=args.input,
        datasets_dir=args.datasets_dir,
        allow_llm_fallback=args.llm_fallback,
    )
