"""Exclusive atomic persistence for immutable evaluation artifacts."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def write_new_atomic(path: Path, content: str) -> None:
    """Create a new text artifact by fsyncing then atomically renaming it."""

    if path.exists():
        raise FileExistsError(f"Evaluation artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:8]}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Evaluation artifact already exists: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
