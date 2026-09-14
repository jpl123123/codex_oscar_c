"""Verify NPU call ordering and shape admission without a numeric fallback."""
import unittest
from types import SimpleNamespace

from oscar_ascend.ops import (
    FullAttentionPipeline, LayerWindowState, PipelineMetadata, PREFIX_OPS,
    PRIMITIVE_OPS, ScratchBudget, SharedScratch,
)


class TensorShape:
    def __init__(self, *shape):
        self.shape = shape
        self.device = SimpleNamespace(type="npu")

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.start not in (None, 0):
            raise AssertionError("test expects prefix views only")
        return TensorShape(min(item.stop, self.shape[0]), *self.shape[1:])

    def fill_(self, value):
        return self

    def zero_(self):
        return self

    def view(self, *shape):
        return TensorShape(*shape)


class RecordingOps:
    def __init__(self):
        self.calls = []
        for name in PRIMITIVE_OPS + ("window_state_out",) + PREFIX_OPS:
            if name not in ("abi_version", "workspace_size"):
                setattr(self, name, self.record(name))

    def record(self, name):
        def call(*args):
            self.calls.append((name, args))
        return call

    def abi_version(self):
        return 2

    def workspace_size(self, *args):
        return 512


class PipelineContracts(unittest.TestCase):
    def fixture(self):
        b = ScratchBudget(16, 2, 4, 1, 64, 8, 4, 1024)
        q = TensorShape(4, 4, 64)
        k = TensorShape(4, 1, 64)
        scratch = SharedScratch(
            b, TensorShape(16, 4, 64), TensorShape(16, 4, 64),
            TensorShape(16, 4, 64), TensorShape(16, 4), TensorShape(16, 4),
            TensorShape(24, 1, 64), TensorShape(24, 1, 64), TensorShape(24),
            TensorShape(16), TensorShape(16), TensorShape(2), TensorShape(2),
            TensorShape(1024),
        )
        cache = SimpleNamespace(
            history=TensorShape(10, 768, 1, 40),
            sink_recent_k=TensorShape(2, 16, 1, 64),
            sink_recent_v=TensorShape(2, 16, 1, 64),
            layout=SimpleNamespace(head_size=64, head_size_v=64, group_size=256,
                                   block_size=768),
            staging_k=TensorShape(3, 768, 1, 64),
            staging_v=TensorShape(3, 768, 1, 64),
            staging_owner=TensorShape(3, 768),
        )
        metadata = PipelineMetadata(
            TensorShape(1), TensorShape(2), TensorShape(1), TensorShape(1),
            TensorShape(1, 64), TensorShape(4), TensorShape(1), 4, 128,
        )
        state = LayerWindowState(TensorShape(2, 16), TensorShape(2, 4), TensorShape(1))
        return scratch, dict(query=q, key=k, value=k, cache=cache, state=state,
                             metadata=metadata, rotation_k=TensorShape(64, 64),
                             rotation_v=TensorShape(64, 64), output=TensorShape(4, 256),
                             scale=0.125)

    def test_finish_does_not_overwrite_window_before_attention(self):
        scratch, inputs = self.fixture()
        operators = RecordingOps()
        pipeline = FullAttentionPipeline(operators, scratch, sink=4, recent=8, max_pending=4)
        output = pipeline.forward(**inputs)
        self.assertIs(output, inputs["output"])
        self.assertEqual([name for name, args in operators.calls], [
            "stage_window_out", "prefix_restore_out", "window_state_out", "store_int2",
            "rotate_out", "history_attention_out", "window_attention_out", "merge_out",
            "window_state_out", "store_int2", "store_int2",
        ])
        # Staging and restore precede the first window phase; the restore call
        # carries the lossy counter and the same table granularity as phase 0.
        self.assertEqual(operators.calls[1][1][-1], 128)
        self.assertEqual(operators.calls[2][1][-5], 0)
        self.assertEqual(operators.calls[8][1][-5], 1)
        # Virtual table IDs are flattened with table granularity 128, not the
        # physical history manager page size 768.
        self.assertEqual(operators.calls[2][1][-1], 128)
        self.assertEqual(operators.calls[5][1][-1], 128)

    def test_workspace_failure_happens_before_any_kv_mutation(self):
        scratch, inputs = self.fixture()
        operators = RecordingOps()
        operators.workspace_size = lambda *args: 2048
        pipeline = FullAttentionPipeline(operators, scratch, sink=4, recent=8, max_pending=4)
        with self.assertRaisesRegex(ValueError, "workspace"):
            pipeline.forward(**inputs)
        self.assertEqual(operators.calls, [])

    def test_clip_zero_is_supported_and_invalid_ratios_fail(self):
        scratch, _ = self.fixture()
        FullAttentionPipeline(RecordingOps(), scratch, sink=4, recent=8,
                              max_pending=4, k_clip=0, v_clip=0)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(clip=invalid), self.assertRaises(ValueError):
                FullAttentionPipeline(RecordingOps(), scratch, sink=4, recent=8,
                                      max_pending=4, k_clip=invalid)


if __name__ == "__main__":
    unittest.main()
