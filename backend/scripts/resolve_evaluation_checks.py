"""Resolve a human-editable evaluation profile against the trusted check catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.evaluation.check_library import (  # noqa: E402
    compile_evaluation_check_plan,
    load_evaluation_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--disable-engineering", action="store_true")
    parser.add_argument("--extensibility-probe", action="store_true")
    args = parser.parse_args()
    profile_path = (
        args.profile.resolve()
        if args.profile.is_absolute()
        else (ROOT / args.profile).resolve()
    )
    plan = compile_evaluation_check_plan(
        load_evaluation_profile(profile_path),
        engineering_enabled=not args.disable_engineering,
        extensibility_probe_enabled=args.extensibility_probe,
    )
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
