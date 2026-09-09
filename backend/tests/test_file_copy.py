from __future__ import annotations

from pathlib import Path

from backend.file_copy import copy_file_isolated, copytree_isolated


def test_copy_file_isolated_does_not_share_mutations(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"original")

    copy_file_isolated(source, destination)
    destination.write_bytes(b"changed")

    assert source.read_bytes() == b"original"
    assert destination.read_bytes() == b"changed"


def test_copytree_isolated_preserves_tree_and_isolation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "value.txt").write_text("original", encoding="utf-8")

    copytree_isolated(source, destination)
    copied = destination / "nested" / "value.txt"
    copied.write_text("changed", encoding="utf-8")

    assert (source / "nested" / "value.txt").read_text(encoding="utf-8") == "original"
    assert copied.read_text(encoding="utf-8") == "changed"
