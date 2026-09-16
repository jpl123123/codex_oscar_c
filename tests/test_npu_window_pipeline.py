"""Opt-in real NPU pipeline regression; all dense reference math stays on NPU.

Run with OSCAR_ASCEND_TEST_LIBRARY=/abs/liboscar_ascend_ops.so and the task's
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7. CPU-only collection skips without importing torch.
"""
import os
import json
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("OSCAR_ASCEND_TEST_LIBRARY"),
                     "requires built AscendC library and real NPU")
class NpuWindowPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "4,5,6,7":
            raise RuntimeError("this task's NPU tests require physical devices 4,5,6,7")
        import torch
        import torch_npu  # noqa: F401
        from oscar_ascend.ops import load_library
        cls.torch = torch
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")
        cls.operators = load_library(os.environ["OSCAR_ASCEND_TEST_LIBRARY"], require_window=True)

    def setup_pipeline(self):
        from oscar_ascend.ops import (
            FullAttentionPipeline, LayerWindowState, ScratchBudget, SharedScratch,
        )
        torch, device, ops = self.torch, self.device, self.operators
        workspace = max(ops.workspace_size(16, 4, 64, 1),
                        ops.workspace_size(8, 4, 64, 8))
        budget = ScratchBudget(16, 2, 4, 1, 64, 8, 4, workspace)
        scratch = SharedScratch.allocate(budget, device)
        self.assertEqual(scratch.allocated_bytes, budget.tensor_bytes)
        self.pipeline = FullAttentionPipeline(ops, scratch, sink=4, recent=8, max_pending=4)
        self.cache = SimpleNamespace(
            history=torch.zeros((8, 128, 1, 40), dtype=torch.uint8, device=device),
            sink_recent_k=torch.zeros((2, 16, 1, 64), dtype=torch.bfloat16, device=device),
            sink_recent_v=torch.zeros((2, 16, 1, 64), dtype=torch.bfloat16, device=device),
            layout=SimpleNamespace(head_size=64, head_size_v=64, group_size=64, block_size=128),
            staging_k=torch.zeros((2, 128, 1, 64), dtype=torch.bfloat16, device=device),
            staging_v=torch.zeros((2, 128, 1, 64), dtype=torch.bfloat16, device=device),
            staging_owner=torch.full((2, 128), -1, dtype=torch.int64, device=device),
        )
        self.state = LayerWindowState.allocate(max_requests=2, capacity=16, device=device)
        self.rotation = torch.eye(64, dtype=torch.float32, device=device)

    def step(self, computed, count, *, prefill, seed):
        from oscar_ascend.ops import PipelineMetadata
        torch, device = self.torch, self.device
        torch.manual_seed(seed)
        q = torch.randn((count, 4, 64), dtype=torch.bfloat16, device=device)
        k = torch.randn((count, 1, 64), dtype=torch.bfloat16, device=device)
        v = torch.randn((count, 1, 64), dtype=torch.bfloat16, device=device)
        metadata = PipelineMetadata(
            torch.tensor([computed + count], dtype=torch.int32, device=device),
            torch.tensor([0, count], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int64, device=device),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device),
            torch.arange(128 + computed, 128 + computed + count, dtype=torch.int64, device=device),
            torch.tensor([prefill], dtype=torch.bool, device=device), count, 128,
        )
        output = torch.empty_like(q)
        self.pipeline.forward(query=q, key=k, value=v, cache=self.cache,
                              state=self.state, metadata=metadata,
                              rotation_k=self.rotation, rotation_v=self.rotation,
                              output=output, scale=0.125)
        torch.npu.synchronize()
        # Small diagnostic status only; online K/V are never read back.
        self.assertEqual(self.state.transactions[0, 3].item(), 0)
        return q, k, v, output

    def dense_reference(self, q, keys, values, computed):
        torch = self.torch
        k = keys.repeat_interleave(4, dim=1).float()
        v = values.repeat_interleave(4, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) * 0.125
        qpos = torch.arange(computed, computed + q.shape[0], device=self.device)
        kpos = torch.arange(keys.shape[0], device=self.device)
        scores.masked_fill_(kpos.unsqueeze(0) > qpos.unsqueeze(1), float("-inf"))
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).to(q.dtype)

    def assert_relative_l2(self, actual, expected, stage):
        torch = self.torch
        relative = ((actual.float() - expected.float()).norm()
                    / expected.float().norm().clamp_min(1e-12)).item()
        print(json.dumps({"test": self.id(), "stage": stage,
                          "relative_l2": relative, "threshold": 0.02}), flush=True)
        self.assertLess(relative, 0.02, f"{stage}: PR window relative-L2 threshold")

    def test_prefill_verify_reject_next_verify(self):
        self.setup_pipeline()
        torch = self.torch
        q0, k0, v0, out0 = self.step(0, 10, prefill=True, seed=1)
        self.assert_relative_l2(out0, self.dense_reference(q0, k0, v0, 0), "prefill")
        q1, k1, v1, out1 = self.step(10, 4, prefill=False, seed=2)
        self.assert_relative_l2(out1, self.dense_reference(
            q1, torch.cat((k0, k1)), torch.cat((v0, v1)), 10), "verify")
        # Only position 10 survived: all three draft candidates were rejected.
        q2, k2, v2, out2 = self.step(11, 4, prefill=False, seed=3)
        self.assert_relative_l2(out2, self.dense_reference(
            q2, torch.cat((k0, k1[:1], k2)), torch.cat((v0, v1[:1], v2)), 11),
            "next_verify_after_rejection")
        self.assertEqual(self.state.transactions[0, 1].item(), 11)
        self.assertEqual(self.state.transactions[0, 2].item(), 4)
        torch.testing.assert_close(self.cache.sink_recent_k[0, 12:16], k2,
                                   atol=0, rtol=0)

    def test_pipeline_history_matches_pr_decoded_mixed_golden(self):
        self.setup_pipeline()
        torch = self.torch
        from oscar_ascend.probe import decoded_cache

        _, k0, v0, _ = self.step(0, 16, prefill=True, seed=11)
        # Persisted History [4,8) exists after first prompt finish. Decode a
        # small explicit test result; this helper is excluded from online code.
        before_history = self.cache.history.clone()
        q1, k1, v1, output = self.step(16, 4, prefill=False, seed=12)
        dk, dv = decoded_cache(torch, self.cache.history, [1, 2], 16, 64)
        mixed_k, mixed_v = torch.cat((k0, k1)), torch.cat((v0, v1))
        mixed_k[4:8].copy_(dk[4:8].to(device=self.device, dtype=torch.bfloat16))
        mixed_v[4:8].copy_(dv[4:8].to(device=self.device, dtype=torch.bfloat16))
        self.assert_relative_l2(output, self.dense_reference(q1, mixed_k, mixed_v, 16),
                                "history_sink_recent_current")
        # Pending verify must not rewrite even one byte of committed History.
        torch.testing.assert_close(self.cache.history, before_history, atol=0, rtol=0)
        q2, k2, v2, output2 = self.step(18, 4, prefill=False, seed=13)
        dk2, dv2 = decoded_cache(torch, self.cache.history, [1, 2], 18, 64)
        mixed_k2 = torch.cat((k0, k1[:2], k2))
        mixed_v2 = torch.cat((v0, v1[:2], v2))
        mixed_k2[4:10].copy_(dk2[4:10].to(device=self.device, dtype=torch.bfloat16))
        mixed_v2[4:10].copy_(dv2[4:10].to(device=self.device, dtype=torch.bfloat16))
        self.assert_relative_l2(output2, self.dense_reference(q2, mixed_k2, mixed_v2, 18),
                                "history_after_partial_accept")
        # Old compressed rows are never recompressed; only positions 8 and 9
        # have newly migrated to History.
        torch.testing.assert_close(self.cache.history[1, 4:8], before_history[1, 4:8],
                                   atol=0, rtol=0)

    def test_window_migration_partial_accept_and_long_chunk(self):
        self.setup_pipeline()
        torch, device = self.torch, self.device
        s = self.pipeline.scratch

        def phase(computed, count, prefill, number):
            positions = torch.arange(computed, computed + count, dtype=torch.float32,
                                     device=device).view(count, 1, 1)
            key = (positions / 100 + torch.arange(64, device=device).view(1, 1, 64) / 10000).to(torch.bfloat16)
            value = (-key).contiguous()
            self.operators.window_state_out(
                key, value,
                torch.tensor([computed + count], dtype=torch.int32, device=device),
                torch.tensor([0, count], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.tensor([1], dtype=torch.int64, device=device),
                torch.tensor([[1, 2]], dtype=torch.int32, device=device),
                torch.arange(128 + computed, 128 + computed + count, dtype=torch.int64, device=device),
                torch.tensor([prefill], dtype=torch.bool, device=device),
                self.cache.sink_recent_k, self.cache.sink_recent_v,
                self.state.positions, self.state.transactions,
                s.history_starts[:1], s.history_ends[:1], s.query_positions[:count],
                s.migration_k[:12], s.migration_v[:12], s.migration_slots[:12],
                s.current_history_slots[:count], number, 4, 8, 4, 128,
            )
            torch.npu.synchronize()
            self.assertEqual(self.state.transactions[0, 3].item(), 0)
            return key, value

        phase(0, 16, True, 0)
        old_k, _ = phase(0, 16, True, 1)
        expected_slots = torch.tensor([-1] * 4 + list(range(132, 136)) + [-1] * 8,
                                      dtype=torch.int64, device=device)
        torch.testing.assert_close(s.current_history_slots[:16], expected_slots)
        self.assertEqual(self.state.transactions[0, 1].item(), 16)
        phase(16, 4, False, 0)
        pending_k, _ = phase(16, 4, False, 1)
        phase(18, 4, False, 0)  # two of four materialized rows accepted
        torch.testing.assert_close(s.migration_slots[:2],
                                   torch.tensor([136, 137], dtype=torch.int64, device=device))
        torch.testing.assert_close(s.migration_k[:2], old_k[8:10], atol=0, rtol=0)
        # Positions 16 and 17 occupy ring indices 8 and 9, after rejected tail is discarded.
        torch.testing.assert_close(self.cache.sink_recent_k[0, 8:10], pending_k[:2], atol=0, rtol=0)
        self.assertEqual(self.state.transactions[0, 2].item(), 0)
        phase(18, 16, True, 0)
        phase(18, 16, True, 1)
        torch.testing.assert_close(s.migration_slots[:8],
                                   torch.arange(138, 146, dtype=torch.int64, device=device))
        torch.testing.assert_close(s.current_history_slots[:8],
                                   torch.arange(146, 154, dtype=torch.int64, device=device))
        self.assertEqual(self.state.transactions[0, 1].item(), 34)

    def test_prefix_hit_restores_window_exactly_then_counts_lossy(self):
        self.setup_pipeline()
        torch, device, ops = self.torch, self.device, self.operators
        # One real prefill populates staging for Sink [0,4) and Recent [8,16)
        # plus INT2 History, exactly as the online path would.
        q0, k0, v0, out0 = self.step(0, 16, prefill=True, seed=21)
        self.assert_relative_l2(out0, self.dense_reference(q0, k0, v0, 0), "prefill")
        staged = (self.cache.staging_owner != -1).sum().item()
        self.assertEqual(staged, 12)

        def restore(epoch):
            self.state.transactions[0, 0] = 0  # simulate a new request identity
            ops.prefix_restore_out(
                torch.tensor([17], dtype=torch.int32, device=device),
                torch.tensor([0, 1], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.tensor([epoch], dtype=torch.int64, device=device),
                torch.tensor([[1, 2]], dtype=torch.int32, device=device),
                self.state.positions, self.cache.sink_recent_k, self.cache.sink_recent_v,
                self.state.transactions, self.cache.staging_k, self.cache.staging_v,
                self.cache.staging_owner, self.cache.history,
                self.rotation, self.rotation, self.state.lossy, 4, 8, 4, 128)
            torch.npu.synchronize()

        self.state.lossy.zero_()
        restore(2)
        self.assertEqual(self.state.transactions[0, :3].tolist(), [2, 16, 0])
        self.assertEqual(self.state.transactions[0, 3].item(), 0)
        self.assertEqual(self.state.lossy[0].item(), 0)
        # Staged restore is exact BF16: sink and ring hold the original rows.
        torch.testing.assert_close(self.cache.sink_recent_k[0, 0:4], k0[0:4], atol=0, rtol=0)
        # Ring order: p=12..15 wrap into slots 4..7, p=8..11 keep 8..11.
        torch.testing.assert_close(self.cache.sink_recent_k[0, 4:12],
                                   torch.cat((k0[12:16], k0[8:12])), atol=0, rtol=0)
        torch.testing.assert_close(self.cache.sink_recent_v[0, 4:12],
                                   torch.cat((v0[12:16], v0[8:12])), atol=0, rtol=0)
        self.assertEqual(self.state.positions[0, 0:4].tolist(), [0, 1, 2, 3])
        self.assertEqual(self.state.positions[0, 4:12].tolist(), [12, 13, 14, 15, 8, 9, 10, 11])

        # Evicted staging falls back to bounded INT2 recovery and is counted.
        self.cache.staging_owner.fill_(-1)
        self.state.lossy.zero_()
        restore(3)
        self.assertEqual(self.state.lossy[0].item(), 12)
        self.assertEqual(self.state.transactions[0, :3].tolist(), [3, 16, 0])
        self.assertEqual(self.state.transactions[0, 3].item(), 0)
        # Quantized fallback approximates the original window rows.
        fallback = self.cache.sink_recent_k[0, 0:4].float()
        self.assertLess((fallback - k0[0:4].float()).norm().item()
                        / k0[0:4].float().norm().item(), 0.6)


if __name__ == "__main__":
    unittest.main()
