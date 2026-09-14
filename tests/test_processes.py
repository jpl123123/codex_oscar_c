from pathlib import Path
import os
import sys
import tempfile
import unittest

from oscar_ascend.processes import OwnedProcess, live_group_members, run_phase


@unittest.skipUnless(os.environ.get("OSCAR_RUN_PROCESS_TESTS") == "1",
                     "requires ps/process-group access; run on target with OSCAR_RUN_PROCESS_TESTS=1")
class ProcessTests(unittest.TestCase):
    def test_failure_exits_with_log_and_cleanup_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failure.log"
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                run_phase([sys.executable, "-c", "print('diagnostic', flush=True); raise SystemExit(7)"],
                          log, timeout=5)
            self.assertIn("diagnostic", log.read_text())
            self.assertTrue(log.with_suffix(".process.json").is_file())

    def test_cleanup_reaps_owned_descendant_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
            process = OwnedProcess([sys.executable, "-c", code], Path(directory) / "tree.log").start()
            try:
                self.assertEqual(process.wait(5), 0)
            finally:
                result = process.cleanup(grace=1)
            self.assertEqual(result["status"], "released")
            self.assertEqual(live_group_members(process.process.pid), [])

    def test_timeout_still_cleans_group(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                run_phase([sys.executable, "-c", "import time; time.sleep(60)"],
                          Path(directory) / "timeout.log", timeout=0.1)


if __name__ == "__main__":
    unittest.main()
