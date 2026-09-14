"""Host contracts for drafter routing, device-error propagation and prefix staging.

These tests pin the external integration invariants; NPU execution remains the
separate acceptance track in tests/test_npu_window_pipeline.py.
"""
import unittest
from types import SimpleNamespace

from oscar_ascend.cache_budget import CachePlan, FullLayout, Pool, Section, staging_rows
from oscar_ascend.cache_integration import build_cache_plan
from oscar_ascend.metadata import restore_window_ranges
from oscar_ascend.plugin import PluginConfig
from oscar_ascend.runtime import TargetRuntime, collect_window_error_reports
from oscar_ascend.ops import PREFIX_OPS


class Shape:
    def __init__(self, *shape):
        self.shape = shape
        self.device = SimpleNamespace(type="npu")

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.start not in (None, 0):
            raise AssertionError("test expects prefix views only")
        return Shape(min(item.stop, self.shape[0]), *self.shape[1:])


class DrafterContracts(unittest.TestCase):
    def runtime(self, *, draft_layers=("draft0",)):
        runtime = TargetRuntime.__new__(TargetRuntime)
        runtime.configured = True
        runtime.options = SimpleNamespace()
        runtime.config = SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=True))
        runtime.row_map = Shape(2)
        runtime.epochs = Shape(2)
        runtime.capture_row_map = Shape(2)
        runtime.capture_epochs = Shape(2)
        fixed = SimpleNamespace(
            seq_lens=Shape(2), query_start_loc=Shape(3), block_table=Shape(2, 8),
            slot_mapping=Shape(4), row_to_window=Shape(2), epochs=Shape(2),
            is_prefilling=Shape(2),
            update=lambda common, **kwargs: None,
            addresses=lambda: (1,))
        runtime.metadata_by_layer = {"draft0": fixed, "full0": fixed}
        runtime.group_id_by_layer = {"draft0": 0, "full0": 1}
        runtime.runner = SimpleNamespace(
            drafter=SimpleNamespace(attn_layer_names=list(draft_layers)),
            input_batch=SimpleNamespace(
                block_table=[SimpleNamespace(block_size=128),
                             SimpleNamespace(block_size=128)]))
        return runtime

    def common(self, rows=1, tokens=1):
        return SimpleNamespace(
            num_reqs=rows, num_actual_tokens=tokens, num_input_tokens=tokens,
            query_start_loc=Shape(rows + 1), seq_lens=Shape(rows),
            block_table_tensor=Shape(rows, 8), slot_mapping=Shape(tokens),
            is_prefilling=Shape(rows), max_query_len=1)

    def test_drafter_and_prefix_capabilities_are_implemented(self):
        for name in ("draft", "prefix_lifecycle"):
            self.assertTrue(getattr(TargetRuntime.capabilities, name),
                            f"{name} must be implemented, not silently enabled")

    def test_draft_only_layer_set_builds_through_the_same_metadata(self):
        runtime = self.runtime()
        prepared = runtime.prepare(self.common(), ("draft0",), for_capture=False)
        self.assertEqual(prepared.max_query_len, 1)
        self.assertEqual(prepared.block_table_block_size, 128)

    def test_draft_graph_capture_fails_closed(self):
        runtime = self.runtime()
        from oscar_ascend.backend import RuntimeNotReady
        with self.assertRaisesRegex(RuntimeNotReady, "drafter graph capture"):
            runtime.prepare(self.common(), ("draft0",), for_capture=True)

    def test_mixed_or_target_layer_sets_do_not_require_draft(self):
        runtime = self.runtime(draft_layers=())
        prepared = runtime.prepare(self.common(), ("full0",), for_capture=False)
        self.assertEqual(prepared.block_table_block_size, 128)


class ErrorPropagationContracts(unittest.TestCase):
    class Column:
        def copy_(self, other, non_blocking=False):
            return self

    class Transactions:
        def __getitem__(self, item):
            return ErrorPropagationContracts.Column()

    class ErrorsDevice:
        def __getitem__(self, index):
            return ErrorPropagationContracts.Column()

    class Mirror(list):
        def copy_(self, other, non_blocking=False):
            return self

        def tolist(self):
            return list(self)

    def test_reports_name_layer_row_and_request(self):
        reports = collect_window_error_reports(
            [[0, 0, 0, 0], [1, 0, 0, 2]], ("full0", "draft0"),
            ["req-a", "req-b", None, "req-c"])
        self.assertEqual(len(reports), 2)
        self.assertIn("layer=draft0", reports[0])
        self.assertIn("window_row=0", reports[0])
        self.assertIn("request='req-a'", reports[0])
        self.assertIn("commit exceeds materialized", reports[0])
        self.assertIn("code=2", reports[1])
        self.assertIn("window_row=3", reports[1])
        self.assertIn("request='req-c'", reports[1])

    def test_zero_column_produces_no_reports(self):
        self.assertEqual(collect_window_error_reports(
            [[0, 0], [0, 0]], ("a", "b"), [None, None]), [])

    def test_observe_is_delayed_by_one_step_and_fails_loudly(self):
        runtime = TargetRuntime.__new__(TargetRuntime)
        runtime._layer_order = ("full0",)
        runtime.state_by_layer = {"full0": SimpleNamespace(
            transactions=self.Transactions())}
        runtime._errors_device = self.ErrorsDevice()
        runtime._error_mirror = self.Mirror([[0, 0, 0]])
        runtime._error_mirror_valid = False
        runtime._slot_request_ids = ["req-a", None, None]
        runtime.observe_window_errors()  # first step: publish only, judge nothing
        self.assertTrue(runtime._error_mirror_valid)
        runtime._error_mirror[0][1] = 4  # error appears in the previous snapshot
        with self.assertRaisesRegex(RuntimeError, "window state reported"):
            runtime.observe_window_errors()


class StagingBudgetContracts(unittest.TestCase):
    def test_staging_rows_follow_the_upstream_floor(self):
        self.assertEqual(staging_rows(128, 64, 256, 0), 0 + 3 + 2)
        self.assertEqual(staging_rows(128, 64, 256, 8192), 64)
        self.assertEqual(staging_rows(768, 64, 256, 8192), 11)
        self.assertEqual(staging_rows(128, 0, 0, 0), 3)

    def test_plan_counts_staging_as_fixed_bytes(self):
        layout = FullLayout(128, 2, 256, 256, 256)
        pool = Pool("p", "full", ("full0",), layout.sections(), layout)
        plan = CachePlan((pool,), 8, 324, 1024, staging_tokens=8192, sink=64, recent=256)
        self.assertEqual(plan.staging_bytes,
                         64 * 128 * 2 * (256 + 256) * 2 + 64 * 128 * 8)
        self.assertEqual(plan.fixed_bytes,
                         plan.window_bytes + plan.staging_bytes + 1024)

    def test_configuration_validates_and_defaults_to_the_pr_budget(self):
        self.assertEqual(PluginConfig().staging_tokens, 8192)
        with self.assertRaises(ValueError):
            PluginConfig(staging_tokens=-1)

    def test_native_plan_carries_staging_options(self):
        class FullAttentionSpec:
            block_size = 128
            num_kv_heads = 2
            head_size = 256
            head_size_v = 256
            dtype = "bfloat16"

        group = SimpleNamespace(layer_names=["full0"], kv_cache_spec=FullAttentionSpec())
        tensor = SimpleNamespace(size=348160, shared_by=["full0"])
        native = SimpleNamespace(num_blocks=10, kv_cache_groups=[group],
                                 kv_cache_tensors=[tensor])
        plan = build_cache_plan(native, max_requests=8, sink=64, recent=256,
                                speculative_tokens=3, group_size=256,
                                staging_tokens=8192)
        self.assertEqual(plan.staging_tokens, 8192)
        self.assertEqual(plan.sink, 64)
        self.assertEqual(plan.recent, 256)
        self.assertGreater(plan.staging_bytes, 0)
        self.assertGreater(plan.physical_bytes(10), plan.window_bytes + 10 * plan.bytes_per_block)


class RestoreRangeContracts(unittest.TestCase):
    def test_restore_ranges_cover_all_window_geometries(self):
        sink, recent = 4, 8
        self.assertEqual(restore_window_ranges(3, sink, recent), ((0, 3), (3, 3)))
        self.assertEqual(restore_window_ranges(4, sink, recent), ((0, 4), (4, 4)))
        self.assertEqual(restore_window_ranges(12, sink, recent), ((0, 4), (4, 12)))
        self.assertEqual(restore_window_ranges(20, sink, recent), ((0, 4), (12, 20)))
        self.assertEqual(restore_window_ranges(20, sink, 0), ((0, 4), (20, 20)))
        self.assertEqual(restore_window_ranges(0, sink, recent), ((0, 0), (0, 0)))
        with self.assertRaises(ValueError):
            restore_window_ranges(-1, sink, recent)

    def test_prefix_ops_are_part_of_the_required_bundle(self):
        self.assertEqual(PREFIX_OPS, ("dequant_history_out", "stage_window_out",
                                      "prefix_restore_out"))


if __name__ == "__main__":
    unittest.main()
