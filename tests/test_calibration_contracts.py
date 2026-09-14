"""Host-only calibration lifecycle and identity contracts; no numeric fallback."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from oscar_ascend.calibrate import native_llm_kwargs, run_calibration, validate_request, validate_workers
from oscar_ascend.calibration_data import digest
from oscar_ascend.calibration_worker import SampleTrace, verify_passes
from oscar_ascend.runtime import OSCAR_REFERENCE_COMMIT, ROTATION_OBJECTIVES


def request_fixture(directory):
    root = Path(directory)
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size":256}')
    library = root / "operators.so"
    library.write_bytes(b"not an executable: host identity fixture")
    service = json.loads((Path(__file__).parents[1] / "configs/target_service.json").read_text())
    service["argv"][2] = str(model)
    service_path = root / "service.json"
    service_path.write_text(json.dumps(service))
    prompts = ["Deterministic calibration fixture prompt"]
    return {
        "format": "oscar-ascend-calibration-request-v1", "model": str(model),
        "reference_commit": OSCAR_REFERENCE_COMMIT, "tensor_parallel_size": 4,
        "max_tokens": 1, "seed": 0, "token_budget": 30000, "rpc_timeout_seconds": 10,
        "group_size": 256, "clip_ratios": {"k": 0, "v": 0}, "objectives": ROTATION_OBJECTIVES,
        "prompts": prompts, "prompt_sha256": digest(prompts), "calibration_request_sha256": "1" * 64,
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "library_path": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "corpus_sha256": "2" * 64, "target_service_config": str(service_path),
        "target_argv_sha256": digest(service["argv"]), "calibration_profile": {"solver": "ascendc_symmetric_jacobi"},
    }


class CalibrationContracts(unittest.TestCase):
    def test_budget_truncation_is_identical_across_passes(self):
        traces = []
        for _ in range(2):
            trace = SampleTrace(9)
            self.assertEqual(trace.select(6, stage="prefill", query_heads=6, kv_heads=1, head_dim=256), (0, 6))
            self.assertEqual(trace.select(8, stage="prefill", query_heads=6, kv_heads=1, head_dim=256), (6, 3))
            self.assertEqual(trace.select(8, stage="prefill", query_heads=6, kv_heads=1, head_dim=256), (9, 0))
            traces.append({"layer": trace.summary()})
        verify_passes(*traces)
        traces[1]["layer"]["tokens"] -= 1
        with self.assertRaisesRegex(RuntimeError, "different token counts"):
            verify_passes(*traces)

    def test_missing_mtp_or_missing_samples_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "missing real samples"):
            verify_passes({"mtp": {"tokens": 0}}, {"mtp": {"tokens": 0}})
        reports = [dict(rank=i, pid=10 + i, full_layers=["body", "mtp"], mtp_layers=["mtp"])
                   for i in range(4)]
        self.assertEqual(validate_workers(reports), [10, 11, 12, 13])
        reports[3]["mtp_layers"] = []
        with self.assertRaisesRegex(RuntimeError, "MTP"):
            validate_workers(reports)

    def test_native_parameters_preserve_mtp_graph_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            request = request_fixture(directory)
            validate_request(request)
            kwargs = native_llm_kwargs(request)
            self.assertEqual(kwargs["tensor_parallel_size"], 4)
            self.assertEqual(kwargs["quantization"], "ascend")
            self.assertEqual(kwargs["speculative_config"],
                             {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": True})
            self.assertEqual(kwargs["compilation_config"]["cudagraph_mode"], "FULL_DECODE_ONLY")
            self.assertTrue(kwargs["async_scheduling"])
            self.assertNotIn("enforce_eager", kwargs)
            self.assertNotIn("port", kwargs)
            self.assertIsInstance(kwargs["worker_extension_cls"], str)
            request["prompts"].append("changed")
            with self.assertRaisesRegex(ValueError, "prompt hash"):
                validate_request(request)

    def test_failure_still_aborts_hooks_shuts_engine_and_verifies_workers(self):
        events = []
        class FakeLLM:
            def __init__(self, **kwargs):
                self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                    shutdown=lambda timeout: events.append(("shutdown", timeout))))
            def collective_rpc(self, method, **kwargs):
                self_test.assertIsInstance(method, str)
                events.append((method, kwargs))
                if method == "oscar_calibration_begin":
                    return [dict(rank=i, pid=10 + i, full_layers=["body", "mtp"], mtp_layers=["mtp"])
                            for i in range(4)]
                return []
            def reset_prefix_cache(self):
                events.append(("reset_prefix",))
                return True
            def chat(self, *args, **kwargs):
                raise RuntimeError("native forward failed")
        self_test = self
        fake_vllm = SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            request = request_fixture(directory)
            output = Path(directory) / "rotations.pt"
            with patch.dict(sys.modules, {"vllm": fake_vllm, "torch": SimpleNamespace()}), \
                 patch("oscar_ascend.calibrate.wait_for_worker_exit", side_effect=lambda pids: events.append(("exited", pids))):
                with self.assertRaisesRegex(RuntimeError, "native forward failed"):
                    run_calibration(request, output)
            self.assertFalse(output.exists())
            report = json.loads(output.with_suffix(".calibration.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["worker_exit_verified"])
            names = [entry[0] for entry in events]
            self.assertLess(names.index("oscar_calibration_abort"), names.index("shutdown"))
            self.assertLess(names.index("shutdown"), names.index("exited"))


if __name__ == "__main__":
    unittest.main()
