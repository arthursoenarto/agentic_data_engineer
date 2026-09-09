"""Fetch or verify one immutable dataset-local source fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.env import load_env  # noqa: E402
from backend.source_fixtures import (  # noqa: E402
    fetch_source_fixture,
    load_source_fixture_request,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    arguments = parser.parse_args()
    load_env()
    manifest, path, created = fetch_source_fixture(
        load_source_fixture_request(arguments.request.resolve()),
        dataset_dir=arguments.dataset_dir.resolve(),
        repository_root=ROOT,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(f"{'Created' if created else 'Verified'}: {path}")
    print(f"Entries: {len(manifest.entries)}")
    print(f"Source bytes: {manifest.total_size_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
