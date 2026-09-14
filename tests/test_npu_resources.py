from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from oscar_ascend.npu_resources import parse_process_table, verify_release

TABLE = '| NPU Chip | Process id | Process name | Process memory(MB) |\n| 4 0 | 1234 | python | 100 |\n| 5 0 | 9999 | another-task | 500 |\n'


class ResourceTests(unittest.TestCase):
    def test_parse_owned_memory_separately_from_other_task(self):
        rows = parse_process_table(TABLE)
        self.assertEqual(rows[0]["pid"], 1234)
        self.assertEqual(rows[1]["memory_mb"], 500)

    def test_unrecognized_output_never_counts_as_release(self):
        with self.assertRaises(ValueError):
            parse_process_table('npu-smi failed to initialize driver')

    def test_waits_until_only_owned_pid_disappears(self):
        outputs = iter([TABLE, TABLE.replace('| 4 0 | 1234 | python | 100 |\n', '')])
        def runner(*args, **kwargs):
            return SimpleNamespace(stdout=next(outputs), stderr='')
        with tempfile.TemporaryDirectory() as directory:
            report = verify_release({1234}, Path(directory) / 'release.json', runner=runner, interval=0)
        self.assertEqual(report['status'], 'released')
        self.assertEqual(len(report['samples']), 2)

    def test_remaining_owned_allocation_blocks_next_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'allocations remain'):
                verify_release({1234}, Path(directory) / 'release.json', timeout=0,
                               runner=lambda *a, **k: SimpleNamespace(stdout=TABLE, stderr=''))


if __name__ == '__main__':
    unittest.main()
