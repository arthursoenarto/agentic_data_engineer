"""Freeze prompt-template hashes for a matched generation experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agents.etl_pipeline import (  # noqa: E402
    create_pipeline_prompt_lock,
    write_pipeline_prompt_lock,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    lock = create_pipeline_prompt_lock(lock_id=args.lock_id)
    write_pipeline_prompt_lock(output.resolve(), lock)
    print(f"Saved prompt lock to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
