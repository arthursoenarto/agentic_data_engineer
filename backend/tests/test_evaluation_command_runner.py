from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from backend.evaluation.command_runner import run_observed_command


class EvaluationCommandRunnerTests(unittest.TestCase):
    def test_resource_sampling_keeps_cpu_and_memory_when_io_is_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observation = run_observed_command(
                [
                    "python3",
                    "-c",
                    "import time; values = bytearray(4_000_000); time.sleep(0.2); print(len(values))",
                ],
                cwd=root,
                log_dir=root / "logs",
                label="resource-test",
                repository_root=root,
            )

            self.assertEqual(observation.exit_code, 0)
            self.assertIsNotNone(observation.resources.peak_rss_bytes)
            self.assertGreater(observation.resources.peak_rss_bytes or 0, 0)
            self.assertIsNotNone(observation.resources.user_cpu_seconds)
            self.assertIn("4000000", observation.stdout_tail)

    def test_timeout_terminates_descendant_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "child-survived.txt"
            child = (
                "import time; from pathlib import Path; time.sleep(0.8); "
                f"Path({str(marker)!r}).write_text('alive')"
            )
            parent = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                "time.sleep(10)"
            )
            observation = run_observed_command(
                ["python3", "-c", parent],
                cwd=root,
                log_dir=root / "logs",
                label="timeout-test",
                repository_root=root,
                timeout_seconds=0.2,
            )

            self.assertTrue(observation.timed_out)
            self.assertNotEqual(observation.exit_code, 0)
            time.sleep(0.9)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
