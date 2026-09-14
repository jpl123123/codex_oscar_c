import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("compare_runs", Path(__file__).resolve().parents[1] / "benchmarks/compare_runs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BenchmarkTests(unittest.TestCase):
    def data(self):
        manifest = {key: "fixed" for key in module.IDENTITIES}
        manifest["repetitions"] = 2
        run = dict(ttft_ms=20, tpot_ms=2, e2e_ms=100, generation_tokens_per_s=50,
                   failed_requests=0, completed_requests=10)
        return {"manifest": manifest, "cases": [{"id": "short", "runs": [dict(run), dict(run)]},
                                                  {"id": "50k", "runs": [dict(run), dict(run)]}]}

    def test_one_slow_case_cannot_be_hidden_by_fast_case(self):
        native = self.data()
        oscar = copy.deepcopy(native)
        for run in oscar["cases"][0]["runs"]:
            run["tpot_ms"] = 1
        for run in oscar["cases"][1]["runs"]:
            run["tpot_ms"] = 2.1
        report = module.compare(native, oscar)
        self.assertTrue(report["cases"][1]["failed"])
        self.assertEqual(report["status"], "measured_degradation_or_failure")

    def test_mismatched_inputs_are_not_comparable(self):
        native, oscar = self.data(), self.data()
        oscar["manifest"]["inputs"] = "different"
        with self.assertRaisesRegex(ValueError, "inputs"):
            module.compare(native, oscar)


if __name__ == "__main__":
    unittest.main()
