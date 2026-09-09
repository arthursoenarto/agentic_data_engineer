"""Validate an editable dataset contract and create an immutable execution lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.contracts import (  # noqa: E402
    load_dataset_contract,
    lock_dataset_contract,
    write_editable_contract,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or reuse a canonical contract_vN.lock.json.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        help="Editable YAML path; defaults to DATASET_DIR/contracts/contract.yaml.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Inventory path; defaults to DATASET_DIR/dataset_inventory.json.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Create the editable YAML from a historical contract JSON before locking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _args()
    dataset_dir = arguments.dataset_dir.resolve()
    editable = (arguments.contract or dataset_dir / "contracts" / "contract.yaml").resolve()
    inventory = (arguments.inventory or dataset_dir / "dataset_inventory.json").resolve()
    if arguments.initialize_from:
        if editable.exists():
            raise FileExistsError(f"Editable contract already exists: {editable}")
        write_editable_contract(load_dataset_contract(arguments.initialize_from), editable)
    result = lock_dataset_contract(
        editable_contract_path=editable,
        inventory_path=inventory,
    )
    action = "Created" if result.created else "Reused"
    print(f"{action}: {result.path}")
    print(f"Lock SHA-256: {result.sha256}")
