"""Isolated file copies with copy-on-write support when available."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def copy_file_isolated(source: str | Path, destination: str | Path) -> str:
    """Copy one file without sharing mutable contents with the source."""

    src = Path(source)
    dst = Path(destination)
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["cp", "-c", str(src), str(dst)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode == 0:
            shutil.copystat(src, dst)
            return str(dst)
    return str(shutil.copy2(src, dst))


def copytree_isolated(source: str | Path, destination: str | Path) -> Path:
    """Copy a directory tree using isolated copy-on-write files when possible."""

    return Path(
        shutil.copytree(
            source,
            destination,
            copy_function=copy_file_isolated,
        )
    )
