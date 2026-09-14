from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from oscar_ascend.environment import command, source_state


class EnvironmentTests(unittest.TestCase):
    def test_no_upward_git_lookup_for_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("oscar_ascend.environment.command") as run:
                state = source_state(Path(directory))
                self.assertEqual(state["status"], "snapshot_without_git")
                run.assert_not_called()

    def test_commands_are_bounded(self):
        result = command([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05)
        self.assertIsNone(result["returncode"])
        self.assertIn("timed out", result["error"])

    def test_package_import_does_not_load_platform(self):
        result = command([sys.executable, "-c", "import oscar_ascend, sys; assert not any(m in sys.modules for m in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'))"])
        self.assertEqual(result["returncode"], 0, result)


if __name__ == "__main__":
    unittest.main()
