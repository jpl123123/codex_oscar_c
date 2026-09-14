"""Physical budget and global-page aliasing contracts; no KV values on CPU."""
import sys
import inspect
from math import prod
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from oscar_ascend.cache_budget import CachePlan, DTYPE_BYTES, FullLayout, Pool, Section
from oscar_ascend.cache_integration import (CacheAllocation, CacheHookHandle, FullCacheView,
                                           PackedBlockZeroer, allocate_cache_views, build_cache_plan,
                                           install_cache_hooks, minimum_cache_bytes,
                                           replan_kv_cache_config)


@dataclass
class FullAttentionSpec:
    block_size: int = 768
    num_kv_heads: int = 1
    head_size: int = 256
    head_size_v: int = 256
    dtype: str = "bfloat16"
    page_size_bytes: int = 801792

    def max_memory_usage_bytes(self, config):
        length = config.model_config.max_model_len
        return ((length + self.block_size - 1) // self.block_size) * self.page_size_bytes


@dataclass
class MambaSpec:
    shapes: tuple = ((3, 2560), (12, 128, 128))
    dtypes: tuple = ("bfloat16", "bfloat16")
    page_size_bytes: int = 801792

    def max_memory_usage_bytes(self, config):
        return 4 * self.page_size_bytes  # one state plus native MTP=3 state blocks


@dataclass
class KVCacheTensor:
    size: int
    shared_by: list


def native_config():
    full, gdn = FullAttentionSpec(), MambaSpec()
    groups = [SimpleNamespace(layer_names=["full"], kv_cache_spec=full)]
    groups.extend(SimpleNamespace(layer_names=[name], kv_cache_spec=gdn)
                  for name in ("g0", "g1", "g2"))
    return SimpleNamespace(num_blocks=10, kv_cache_groups=groups,
                           kv_cache_tensors=[KVCacheTensor(8017920, ["full", "g0", "g1", "g2"])])


def options():
    return dict(max_requests=8, sink=64, recent=256, speculative_tokens=3,
                group_size=256, runtime_reserve_bytes=4096)


class StorageShape:
    """Address/shape oracle only: stores no KV elements or numerical data."""
    def __init__(self, storage_bytes, dtype, start=0, end=None):
        self.storage_bytes = storage_bytes
        self.dtype = dtype
        self.start = start
        self.end = storage_bytes if end is None else end

    def __getitem__(self, sl):
        return StorageShape(self.storage_bytes, self.dtype, sl.start, sl.stop)

    def view(self, dtype):
        assert self.start % DTYPE_BYTES[dtype] == 0
        assert (self.end - self.start) % DTYPE_BYTES[dtype] == 0
        return StorageShape(self.storage_bytes, dtype, self.start, self.end)


class ShapeOnlyTorch:
    uint8 = "uint8"
    bfloat16 = "bfloat16"
    float16 = "float16"
    int64 = "int64"

    @staticmethod
    def zeros(shape, dtype, device):
        elements = shape if isinstance(shape, int) else prod(shape)
        return StorageShape(elements * DTYPE_BYTES[dtype], dtype)

    @staticmethod
    def full(shape, value, dtype, device):
        elements = shape if isinstance(shape, int) else prod(shape)
        return StorageShape(elements * DTYPE_BYTES[dtype], dtype)

    @staticmethod
    def as_strided(raw, size, stride, storage_offset):
        item_size = DTYPE_BYTES[raw.dtype]
        last = storage_offset + sum((dim - 1) * step for dim, step in zip(size, stride))
        assert storage_offset * item_size >= raw.start
        assert (last + 1) * item_size <= raw.end
        return SimpleNamespace(shape=size, stride=stride, storage_offset=storage_offset,
                               storage_bytes=raw.storage_bytes, dtype=raw.dtype)


class CacheBudgetTest(unittest.TestCase):
    def test_int2_affine_bytes_explain_136_without_magic_padding(self):
        layout = FullLayout(768, 1, 256, 256, 256)
        self.assertEqual(layout.bf16_bytes_per_token, 1024)
        self.assertEqual(layout.packed_bytes_per_token, 128)
        self.assertEqual(layout.metadata_bytes_per_token, 8)
        self.assertEqual(layout.slot_bytes, 136)
        self.assertEqual(layout.sections()[0].page_bytes, 768 * 136)
        with self.assertRaisesRegex(ValueError, "one quantization group"):
            FullLayout(768, 1, 256, 256, 128)

    def test_smaller_heads_and_pages_account_for_padding(self):
        layout = FullLayout(1, 3, 128, 256, 256)
        section = layout.sections()[0]
        self.assertEqual(layout.packed_bytes_per_token, 288)
        self.assertEqual(layout.metadata_bytes_per_token, 24)
        self.assertEqual(section.payload_bytes, 312)
        self.assertEqual(section.page_bytes, 320)

    def test_splitting_shared_tensor_releases_physical_padding(self):
        original = native_config()
        plan = build_cache_plan(original, **options())
        gdn, full = plan.pools
        self.assertEqual(gdn.shared_by, ("g0", "g1", "g2"))
        self.assertEqual(gdn.bytes_per_block, 15360 + 393216)
        self.assertEqual(full.bytes_per_block, 768 * 136)
        self.assertEqual(plan.pool_bytes_per_block, 513024)
        self.assertEqual(plan.bytes_per_block, 513024 + 8)
        self.assertLess(plan.bytes_per_block, 801792)
        self.assertEqual(original.num_blocks, 10)
        self.assertEqual(plan.window_capacity, 324)
        self.assertEqual(plan.window_bytes, 8 * 324 * 1024)

    def test_admission_uses_actual_bytes_and_native_group_lifetimes(self):
        native = native_config()
        config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=76800))
        plan = build_cache_plan(native, **options())
        required = minimum_cache_bytes(native, config, **options())
        self.assertEqual(required, plan.physical_bytes(1 + 100 + 3 * 4))
        native_required = (100 + 3 * 4) * 801792
        self.assertLess(required, native_required)
        admitted = replan_kv_cache_config(native, required, **options())
        self.assertEqual(admitted.num_blocks, 113)
        config.model_config.max_model_len += 768
        self.assertGreater(minimum_cache_bytes(native, config, **options()), required)

    def test_installed_admission_preserves_signature_reserve_and_atomic_undo(self):
        def original_planner(vllm_config, kv_cache_groups, available_memory):
            result = native_config()
            result.kv_cache_groups = kv_cache_groups
            result.num_blocks = available_memory // 801792
            result.kv_cache_tensors[0].size = result.num_blocks * 801792
            return result

        def original_admission(vllm_config, kv_cache_groups):
            return sum(g.kv_cache_spec.max_memory_usage_bytes(vllm_config) for g in kv_cache_groups)

        class Runner:
            def _allocate_kv_cache_tensors(self, kv_cache_config): pass
            def _reshape_kv_cache_tensors(self, kv_cache_config, raw): pass
            def initialize_kv_cache(self, kv_cache_config): pass
            def _init_kv_zero_meta(self): pass

        modules = {name: ModuleType(name) for name in (
            "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
            "vllm_ascend", "vllm_ascend.worker", "vllm_ascend.worker.model_runner_v1")}
        utils = modules["vllm.v1.core.kv_cache_utils"]
        utils.get_kv_cache_config_from_groups = original_planner
        utils._max_memory_usage_bytes_from_groups = original_admission
        modules["vllm.v1.core"].kv_cache_utils = utils
        modules["vllm_ascend.worker.model_runner_v1"].NPUModelRunner = Runner
        config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=76800),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            cache_config=SimpleNamespace(num_gpu_blocks_override=None),
            speculative_config=SimpleNamespace(num_speculative_tokens=3))
        groups = native_config().kv_cache_groups
        reserve = lambda cfg, gs: 123456
        with patch.dict(sys.modules, modules):
            handle = install_cache_hooks(sink=64, recent=256, group_size=256,
                                         runtime_reserve_bytes=4096, runtime_reserve_fn=reserve)
            self.assertEqual(inspect.signature(utils._max_memory_usage_bytes_from_groups),
                             inspect.signature(original_admission))
            required = utils._max_memory_usage_bytes_from_groups(config, groups)
            planned = utils.get_kv_cache_config_from_groups(config, groups, required)
            self.assertEqual(planned.num_blocks, 113)
            self.assertEqual(planned.oscar_cache_plan.runtime_reserve_bytes, 4096 + 123456)
            self.assertEqual(utils._max_memory_usage_bytes_from_groups(config, []), 0)
            self.assertIs(handle, install_cache_hooks(sink=64, recent=256, group_size=256,
                                                       runtime_reserve_bytes=4096, runtime_reserve_fn=reserve))
            config.cache_config.num_gpu_blocks_override = 9999
            with self.assertRaisesRegex(RuntimeError, "override"):
                utils._max_memory_usage_bytes_from_groups(config, groups)
            handle.undo()
            self.assertIs(utils._max_memory_usage_bytes_from_groups, original_admission)
            self.assertIs(utils.get_kv_cache_config_from_groups, original_planner)

    def test_gdn_state_shapes_are_runtime_inputs_including_mtp(self):
        original = native_config()
        for group in original.kv_cache_groups[1:]:
            group.kv_cache_spec = MambaSpec(shapes=((6, 2560), (12, 128, 128)))
        plan = build_cache_plan(original, **options())
        self.assertEqual(plan.pools[0].bytes_per_block, 30720 + 393216)
        self.assertEqual(plan.pools[0].sections[0].page_shape, (6, 2560))

    def test_budget_admits_maximum_with_null_block_and_fixed_reserve(self):
        plan = build_cache_plan(native_config(), **options())
        budget = plan.physical_bytes(27) + plan.bytes_per_block - 1
        self.assertEqual(plan.max_num_blocks(budget), 27)
        self.assertLessEqual(plan.physical_bytes(27), budget)
        self.assertGreater(plan.physical_bytes(28), budget)
        with self.assertRaises(ValueError):
            plan.max_num_blocks(plan.physical_bytes(1))

    def test_replanning_preserves_groups_and_tp_minimum_shrink(self):
        original = native_config()
        plan = build_cache_plan(original, **options())
        updated = replan_kv_cache_config(original, plan.physical_bytes(19), **options())
        self.assertEqual(updated.num_blocks, 19)
        self.assertEqual(updated.kv_cache_groups, original.kv_cache_groups)
        self.assertEqual(sum(t.size for t in updated.kv_cache_tensors), 19 * plan.pool_bytes_per_block)
        for tensor in updated.kv_cache_tensors:
            self.assertEqual(tensor.size % updated.num_blocks, 0)
            tensor.size = tensor.size // updated.num_blocks * 11
        updated.num_blocks = 11
        self.assertEqual(sum(t.size for t in updated.kv_cache_tensors), 11 * plan.pool_bytes_per_block)
        self.assertEqual(updated.oscar_cache_plan.window_bytes, plan.window_bytes)
        self.assertEqual(original.kv_cache_tensors[0].size, 8017920)

    def test_window_does_not_alias_different_full_layer_values(self):
        layout = FullLayout(128, 1, 256, 256, 256)
        pool = Pool("history", "full", ("full0", "full1"), layout.sections(), layout)
        plan = CachePlan((pool,), 8, 324)
        self.assertEqual(plan.window_bytes, 2 * 8 * 324 * 1024)
        self.assertEqual(plan.pool_bytes_per_block, 128 * 136)

    def test_each_layer_has_exactly_one_owner_and_unknown_specs_rejected(self):
        original = native_config()
        original.kv_cache_tensors[0].shared_by.append("full")
        with self.assertRaises(ValueError):
            build_cache_plan(original, **options())
        original = native_config()
        original.kv_cache_groups[0].kv_cache_spec = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "unverified cache spec"):
            build_cache_plan(original, **options())

    def test_cpu_allocation_is_rejected_before_torch_import(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                allocate_cache_views(native_config(), "cpu")

    def test_allocation_views_use_absolute_storage_offsets_and_actual_bytes(self):
        plan = build_cache_plan(native_config(), **options())
        config = replan_kv_cache_config(native_config(), plan.physical_bytes(5), **options())
        with patch.dict(sys.modules, {"torch": ShapeOnlyTorch}):
            allocation = allocate_cache_views(config, "npu:0")
        self.assertEqual(allocation.allocated_bytes + allocation.reserved_bytes,
                         plan.physical_bytes(5))
        conv, ssm = allocation.views["g0"]
        self.assertIs(conv, allocation.views["g1"][0])
        self.assertIs(ssm, allocation.views["g2"][1])
        self.assertEqual(conv.storage_offset, 0)
        self.assertEqual(ssm.storage_offset, 5 * 15360 // 2)
        self.assertEqual(ssm.stride, (12 * 128 * 128, 128 * 128, 128, 1))
        self.assertEqual(allocation.views["full"].history.shape, (5, 768, 1, 136))
        self.assertEqual(len(allocation.raw_pools), 2)

    def test_hook_undo_checks_every_target_before_restoring_any(self):
        owner = SimpleNamespace(a="new-a", b="changed-after-install")
        handle = CacheHookHandle(((owner, "a", "old-a", "new-a"),
                                  (owner, "b", "old-b", "new-b")))
        with self.assertRaisesRegex(RuntimeError, "changed cache hook"):
            handle.undo()
        self.assertEqual(owner.a, "new-a")
        owner.b = "new-b"
        handle.undo()
        self.assertEqual((owner.a, owner.b), ("old-a", "old-b"))

    def test_zeroer_only_clears_unique_full_history_pools(self):
        calls = []

        class MetadataBuffer:
            def __init__(self, length, device):
                self.length = length
                self.device = device

            def __getitem__(self, sl):
                return MetadataBuffer(sl.stop, self.device)

            def copy_(self, other, non_blocking=False):
                calls.append(("metadata_copy", self.device, self.length, non_blocking))
                return self

        fake_torch = SimpleNamespace(
            int64="int64",
            empty=lambda size, dtype, device, **kw: MetadataBuffer(size, device),
            tensor=lambda ids, dtype, device: MetadataBuffer(len(ids), device),
            ops=SimpleNamespace(oscar_ascend=SimpleNamespace(
                zero_blocks_out=lambda history, ids: calls.append(("zero", history, ids.length)))),
        )
        raw = object()
        layout = FullLayout(768, 1, 256, 256, 256)
        full = FullCacheView("packed", "window-k", "window-v", layout, 5, raw)
        allocation = CacheAllocation({"f0": full, "f1": full, "g0": ["conv", "ssm"]}, (), 0, 40)
        runner = SimpleNamespace(device="npu:0", pin_memory=True,
                                 kv_cache_config=SimpleNamespace(num_blocks=5))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            zeroer = PackedBlockZeroer(runner, allocation)
            zeroer.zero_block_ids([1, 4])
            zeroer.zero_block_ids([])
            with self.assertRaises(ValueError):
                zeroer.zero_block_ids([5])
        self.assertEqual([call for call in calls if call[0] == "zero"],
                         [("zero", "packed", 2)])
        self.assertIn(("metadata_copy", "npu:0", 2, True), calls)
        self.assertEqual(zeroer._capacity, 5)

    def test_invalid_shape_values_are_not_silently_truncated(self):
        for value in (0, -1, 1.25, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FullLayout(value, 1, 256, 256, 256)
        with self.assertRaises(ValueError):
            FullLayout(128, 1, 256, 256, 100)
        with self.assertRaises(ValueError):
            Pool("bad", "gdn", ("g",),
                 (Section("a", "uint8", (1,)), Section("b", "float32", (1,))))

    def test_shared_pool_intervals_preserve_native_global_block_id(self):
        plan = build_cache_plan(native_config(), **options())
        gdn = plan.pools[0]
        self.assertEqual(gdn.section_spans(5),
                         (("state_0", 0, 76800), ("state_1", 76800, 2042880)))
        # For every state stripe and two different global block IDs, physical
        # intervals are disjoint. Three GDN groups intentionally see the same
        # stripe; the native shared BlockPool supplies distinct live IDs.
        for section, (_, start, end) in zip(gdn.sections, gdn.section_spans(5)):
            intervals = [(start + block * section.page_bytes,
                          start + (block + 1) * section.page_bytes)
                         for block in range(5)]
            self.assertEqual(intervals[-1][1], end)
            self.assertTrue(all(left[1] <= right[0]
                                for left, right in zip(intervals, intervals[1:])))


if __name__ == "__main__":
    unittest.main()
