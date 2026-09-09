"""Build one frozen regular-grid suite from generation acceptance evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.suite_builder import (  # noqa: E402
    build_accepted_regular_grid_suite,
    load_accepted_regular_grid_suite_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--accepted-candidate", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable")
    args = parser.parse_args()
    if not args.accepted_candidate:
        parser.error("At least one --accepted-candidate is required")

    spec_path = args.spec if args.spec.is_absolute() else ROOT / args.spec
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    spec = load_accepted_regular_grid_suite_spec(spec_path.resolve())
    for candidate_id in args.accepted_candidate:
        destination = build_accepted_regular_grid_suite(
            spec=spec,
            candidate_id=candidate_id,
            repository_root=ROOT,
            output_dir=output_dir.resolve(),
            python_executable=args.python_executable,
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
