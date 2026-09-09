from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class GenerationEvaluationBoundaryTests(unittest.TestCase):
    def test_generation_does_not_import_evaluation(self) -> None:
        source_roots = [
            ROOT / "backend" / "agents" / "etl_pipeline",
            ROOT / "backend" / "contracts",
        ]
        violations: list[str] = []
        for source_root in source_roots:
            for path in source_root.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    else:
                        continue
                    if any(name.startswith("backend.evaluation") for name in names):
                        violations.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
