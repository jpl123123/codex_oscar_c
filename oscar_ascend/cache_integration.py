"""External physical allocation for native FULL/GDN cache groups.

This module does not provide an attention implementation. Its hooks must only be
installed with the matching OSCAR backend and lifecycle metadata implementation.
All tensor allocation and views below are NPU-only; shape planning is host-only.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .cache_budget import CachePlan, DTYPE_BYTES, FullLayout, Pool, Section, staging_rows


@dataclass(frozen=True)
class FullCacheView:
    history: Any
    sink_recent_k: Any
    sink_recent_v: Any
    layout: FullLayout
    num_blocks: int
    raw_history: Any
    staging_k: Any = None
    staging_v: Any = None
    staging_owner: Any = None


@dataclass(frozen=True)
class CacheAllocation:
    views: dict[str, Any]
    raw_pools: tuple[Any, ...]
    allocated_bytes: int
    reserved_bytes: int


def _dtype_name(dtype: Any) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in DTYPE_BYTES:
        raise ValueError(f"unsupported native cache dtype: {dtype}")
    return name


def _layer_specs(native_config: Any) -> dict[str, Any]:
    specs = {}
    for group in native_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if type(spec).__name__ not in ("FullAttentionSpec", "MambaSpec"):
            raise ValueError(f"unverified cache spec type: {type(spec).__name__}")
        if type(spec).__name__ == "FullAttentionSpec":
            if getattr(spec, "sliding_window", None) is not None:
                raise ValueError("sliding-window FULL spec is outside the OSCAR contract")
            if getattr(spec, "attention_chunk_size", None) is not None:
                raise ValueError("chunked local attention spec is outside the OSCAR contract")
            if _dtype_name(spec.dtype) != "bfloat16":
                raise ValueError("OSCAR Sink/Recent require native BF16 KV")
        for layer in group.layer_names:
            if layer in specs:
                raise ValueError(f"layer appears in multiple native groups: {layer}")
            specs[layer] = spec
    return specs


def build_cache_plan(native_config: Any, *, max_requests: int, sink: int,
                     recent: int, speculative_tokens: int, group_size: int,
                     runtime_reserve_bytes: int = 0,
                     staging_tokens: int = 0) -> CachePlan:
    """Derive owners and state dimensions from actual native runtime specs.

    No layer count, TP dimension, GDN state byte count, or page size is hardcoded.
    The native caller retains responsibility for deriving groups/block tables.
    """
    from .cache_budget import nonnegative

    for name, value in (("sink", sink), ("recent", recent),
                        ("speculative_tokens", speculative_tokens),
                        ("staging_tokens", staging_tokens)):
        nonnegative(value, name)
    specs = _layer_specs(native_config)
    pools = []
    for index, tensor in enumerate(native_config.kv_cache_tensors):
        if not tensor.shared_by:
            raise ValueError("native tensor has no layer owners")
        gdn_layers, full_layers = [], []
        for layer in tensor.shared_by:
            if layer not in specs:
                raise ValueError(f"unknown layer in native physical tensor: {layer}")
            target = gdn_layers if type(specs[layer]).__name__ == "MambaSpec" else full_layers
            target.append(layer)
        if gdn_layers:
            spec = specs[gdn_layers[0]]
            signature = (tuple(spec.shapes), tuple(map(_dtype_name, spec.dtypes)))
            if len(spec.shapes) != len(spec.dtypes):
                raise ValueError("GDN state shape/dtype counts disagree")
            for layer in gdn_layers[1:]:
                other = specs[layer]
                if (tuple(other.shapes), tuple(map(_dtype_name, other.dtypes))) != signature:
                    raise ValueError("shared GDN pools require equal state shapes and dtypes")
            sections = tuple(Section(f"state_{i}", _dtype_name(dtype), tuple(shape))
                             for i, (shape, dtype) in enumerate(zip(spec.shapes, spec.dtypes)))
            pools.append(Pool(f"native_{index}_gdn", "gdn", tuple(gdn_layers), sections))
        if full_layers:
            def layout_for(layer):
                spec = specs[layer]
                return FullLayout(spec.block_size, spec.num_kv_heads, spec.head_size,
                                  getattr(spec, "head_size_v", spec.head_size), group_size)
            layout = layout_for(full_layers[0])
            if any(layout_for(layer) != layout for layer in full_layers[1:]):
                raise ValueError("shared FULL history pools require equal layouts")
            pools.append(Pool(f"native_{index}_full", "full", tuple(full_layers),
                              layout.sections(), layout))
    if {layer for pool in pools for layer in pool.shared_by} != set(specs):
        raise ValueError("native physical tensor owners do not cover the cache groups")
    if not any(pool.kind == "full" for pool in pools):
        raise ValueError("OSCAR allocation requires at least one FULL layer")
    # One extra target token plus the draft budget. This is bounded tentative
    # storage, not a copy of History; commit/rollback belongs to the NPU kernel.
    capacity = sink + recent + speculative_tokens + 1
    return CachePlan(tuple(pools), max_requests, capacity, runtime_reserve_bytes,
                     staging_tokens=staging_tokens, sink=sink, recent=recent)


def minimum_cache_bytes(native_config: Any, vllm_config: Any,
                        **plan_options: Any) -> int:
    """Actual cache bytes for one maximal native request plus fixed reserves.

    Uses the native per-group lifecycle demand, including speculative GDN
    states; every group consumes distinct IDs from the shared global pool.
    """
    from .cache_budget import nonnegative, positive

    plan = build_cache_plan(native_config, **plan_options)
    required_blocks = 1  # native null block
    for group in native_config.kv_cache_groups:
        spec = group.kv_cache_spec
        page_bytes = positive(spec.page_size_bytes, "native page bytes")
        demand = nonnegative(spec.max_memory_usage_bytes(vllm_config), "native group demand")
        required_blocks += (demand + page_bytes - 1) // page_bytes
    return plan.physical_bytes(max(2, required_blocks))


def replan_kv_cache_config(native_config: Any, available_memory: int,
                          **plan_options: Any) -> Any:
    """Return actual physical descriptors while retaining native groups/specs.

    Descriptor sizes are linear in num_blocks so upstream TP-minimum shrinking
    remains correct. Fixed request windows and workspace reserve are accounted
    separately and are not incorrectly shrunk with the per-rank block count.
    """
    if hasattr(native_config, "oscar_cache_plan"):
        raise ValueError("cache config has already been planned for OSCAR")
    plan = build_cache_plan(native_config, **plan_options)
    config = deepcopy(native_config)
    config.num_blocks = plan.max_num_blocks(available_memory)
    tensor_type = type(native_config.kv_cache_tensors[0])
    config.kv_cache_tensors = [tensor_type(size=pool.bytes_per_block * config.num_blocks,
                                         shared_by=list(pool.shared_by))
                               for pool in plan.pools]
    config.oscar_cache_plan = plan
    config.oscar_available_memory = available_memory
    return config


def _section_view(torch: Any, raw: Any, num_blocks: int, section: Section,
                  start: int, end: int) -> Any:
    dtype = getattr(torch, section.dtype)
    item_bytes = DTYPE_BYTES[section.dtype]
    shape = (num_blocks, *section.page_shape)
    stride = []
    running = 1
    for size in reversed(section.page_shape):
        stride.append(running)
        running *= size
    strides = (section.page_bytes // item_bytes, *reversed(stride))
    # storage_offset is explicitly relative to the underlying raw allocation;
    # defaulting it after slicing is not portable across dtype reinterpretation.
    return torch.as_strided(raw[start:end].view(dtype), size=shape, stride=strides,
                            storage_offset=start // item_bytes)


def allocate_cache_views(kv_cache_config: Any, device: Any) -> CacheAllocation:
    """Allocate packed History, bounded BF16 windows, and native GDN states."""
    if str(device).split(":", 1)[0] != "npu":
        raise RuntimeError("OSCAR cache allocation requires an NPU device; no CPU fallback")
    plan = getattr(kv_cache_config, "oscar_cache_plan", None)
    if not isinstance(plan, CachePlan):
        raise RuntimeError("missing OSCAR physical plan; refusing a native BF16 History pool")
    import torch

    num_blocks = kv_cache_config.num_blocks
    if plan.physical_bytes(num_blocks) > kv_cache_config.oscar_available_memory:
        raise RuntimeError("physical cache exceeds its admitted memory budget")
    if len(kv_cache_config.kv_cache_tensors) != len(plan.pools):
        raise RuntimeError("physical descriptor count changed after planning")
    views, raw_pools = {}, []
    actual_bytes = 0
    staging_total = 0
    for pool, descriptor in zip(plan.pools, kv_cache_config.kv_cache_tensors):
        expected_bytes = pool.bytes_per_block * num_blocks
        if descriptor.size != expected_bytes or tuple(descriptor.shared_by) != pool.shared_by:
            raise RuntimeError("physical descriptor changed after planning")
        raw = torch.zeros(expected_bytes, dtype=torch.uint8, device=device)
        raw_pools.append(raw)
        actual_bytes += expected_bytes
        section_views = {
            section.name: _section_view(torch, raw, num_blocks, section, start, end)
            for section, (_, start, end) in zip(pool.sections, pool.section_spans(num_blocks))
        }
        for layer in pool.shared_by:
            if pool.kind == "gdn":
                views[layer] = [section_views[section.name] for section in pool.sections]
            else:
                layout = pool.layout
                k_shape = (plan.max_requests, plan.window_capacity,
                           layout.num_kv_heads, layout.head_size)
                v_shape = (*k_shape[:-1], layout.head_size_v)
                k_window = torch.zeros(k_shape, dtype=torch.bfloat16, device=device)
                v_window = torch.zeros(v_shape, dtype=torch.bfloat16, device=device)
                actual_bytes += plan.max_requests * plan.window_capacity * layout.bf16_bytes_per_token
                rows = staging_rows(layout.block_size, plan.sink, plan.recent,
                                    plan.staging_tokens)
                staging_k = torch.zeros((rows, layout.block_size,
                                         layout.num_kv_heads, layout.head_size),
                                        dtype=torch.bfloat16, device=device)
                staging_v = torch.zeros((rows, layout.block_size,
                                         layout.num_kv_heads, layout.head_size_v),
                                        dtype=torch.bfloat16, device=device)
                owner = torch.full((rows, layout.block_size), -1,
                                   dtype=torch.int64, device=device)
                staging_total += (rows * layout.block_size * layout.bf16_bytes_per_token
                                  + rows * layout.block_size * 8)
                views[layer] = FullCacheView(section_views["packed_kv"], k_window, v_window,
                                             layout, num_blocks, raw,
                                             staging_k, staging_v, owner)
    actual_bytes += staging_total
    if staging_total != plan.staging_bytes:
        raise RuntimeError("staging allocation does not match the planned budget")
    reserved_bytes = plan.runtime_reserve_bytes + 8 * num_blocks
    if actual_bytes + reserved_bytes != plan.physical_bytes(num_blocks):
        raise RuntimeError("allocation byte accounting does not match plan")
    return CacheAllocation(views, tuple(raw_pools), actual_bytes, reserved_bytes)


class PackedBlockZeroer:
    """Native scheduler block-ID batches -> AscendC packed-history clearing.

    Only scheduler integer metadata is materialized on Host. KV bytes stay on
    NPU. GDN state initialization/update remains in the native GDN path. Request
    window validity/epoch must be maintained independently by the runtime.
    """
    def __init__(self, runner: Any, allocation: CacheAllocation):
        if str(runner.device).split(":", 1)[0] != "npu":
            raise RuntimeError("packed block zeroing requires NPU")
        import torch

        try:
            self._zero = torch.ops.oscar_ascend.zero_blocks_out
        except AttributeError as exc:
            raise RuntimeError("AscendC zero_blocks_out is not loaded") from exc
        self._torch = torch
        self._device = runner.device
        self._histories = []
        seen = set()
        for view in allocation.views.values():
            if isinstance(view, FullCacheView) and id(view.raw_history) not in seen:
                self._histories.append(view.history)
                seen.add(id(view.raw_history))
        if not self._histories:
            raise RuntimeError("packed block zeroer has no FULL pools")
        self._num_blocks = runner.kv_cache_config.num_blocks
        # A batch of newly allocated native IDs cannot contain more IDs than
        # the shared BlockPool. This device allocation is explicitly budgeted.
        self._capacity = self._num_blocks
        self._ids_pinned = torch.empty(self._capacity, dtype=torch.int64,
                                       device="cpu", pin_memory=runner.pin_memory)
        self._ids_device = torch.empty(self._capacity, dtype=torch.int64,
                                       device=self._device)

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        if any(not isinstance(index, int) or index < 0 or index >= self._num_blocks
               for index in block_ids):
            raise ValueError("native block ID outside the OSCAR physical pool")
        torch = self._torch
        count = len(block_ids)
        if count > self._capacity:
            raise ValueError("native zeroing batch exceeds the global BlockPool")
        # This tensor contains scheduler IDs, never keys, values, or SSM state.
        source = torch.tensor(block_ids, dtype=torch.int64, device="cpu")
        self._ids_pinned[:count].copy_(source)
        device_ids = self._ids_device[:count]
        device_ids.copy_(self._ids_pinned[:count], non_blocking=True)
        for history in self._histories:
            self._zero(history, device_ids)


@dataclass(frozen=True)
class CacheHookHandle:
    replacements: tuple[tuple[Any, str, Any, Any], ...]

    def undo(self) -> None:
        # Do not overwrite another plugin's later patch when reverting ours.
        for owner, name, original, replacement in self.replacements:
            if getattr(owner, name) is not replacement:
                raise RuntimeError(f"cannot revert changed cache hook: {name}")
        for owner, name, original, replacement in reversed(self.replacements):
            setattr(owner, name, original)


def install_cache_hooks(*, sink: int, recent: int, group_size: int,
                        runtime_reserve_bytes: int = 0,
                        staging_tokens: int = 0,
                        block_zeroer_factory: Any = PackedBlockZeroer,
                        runtime_reserve_fn: Any = None) -> CacheHookHandle:
    """Install external planner and allocation/view hooks at worker setup.

    The plugin must also install the FULL backend and complete block-reuse zeroing.
    It must reject unimplemented offload/connector lifecycles before use.
    """
    from vllm.v1.core import kv_cache_utils
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    options = (sink, recent, group_size, runtime_reserve_bytes, staging_tokens,
               block_zeroer_factory, runtime_reserve_fn)
    original_planner = kv_cache_utils.get_kv_cache_config_from_groups
    installed = getattr(original_planner, "_oscar_cache_options", None)
    if installed is not None:
        if installed != options:
            raise RuntimeError("OSCAR cache hooks already installed with different options")
        return original_planner._oscar_cache_handle

    original_allocate = NPUModelRunner._allocate_kv_cache_tensors
    original_reshape = NPUModelRunner._reshape_kv_cache_tensors
    original_initialize = NPUModelRunner.initialize_kv_cache
    original_zero_meta = NPUModelRunner._init_kv_zero_meta
    original_admission = kv_cache_utils._max_memory_usage_bytes_from_groups

    def plan_options(vllm_config, kv_cache_groups):
        if vllm_config.cache_config.num_gpu_blocks_override is not None:
            raise RuntimeError("num_gpu_blocks_override has no verified OSCAR admission path")
        spec_config = vllm_config.speculative_config
        extra_reserve = 0 if runtime_reserve_fn is None else runtime_reserve_fn(vllm_config, kv_cache_groups)
        from .cache_budget import nonnegative
        nonnegative(extra_reserve, "runtime_reserve_fn result")
        return dict(
            max_requests=vllm_config.scheduler_config.max_num_seqs,
            sink=sink, recent=recent, group_size=group_size,
            speculative_tokens=spec_config.num_speculative_tokens if spec_config else 0,
            runtime_reserve_bytes=runtime_reserve_bytes + extra_reserve,
            staging_tokens=staging_tokens,
        )

    @wraps(original_planner)
    def planner(vllm_config, kv_cache_groups, available_memory):
        settings = plan_options(vllm_config, kv_cache_groups)
        native = original_planner(vllm_config, kv_cache_groups, available_memory)
        return replan_kv_cache_config(native, available_memory, **settings)

    @wraps(original_admission)
    def admission(vllm_config, kv_cache_groups):
        if not kv_cache_groups:
            return 0
        settings = plan_options(vllm_config, kv_cache_groups)
        # Original planning performs no tensor allocation and tolerates N=0;
        # only its verified group -> physical-sharing descriptors are needed.
        native = original_planner(vllm_config, kv_cache_groups, 0)
        return minimum_cache_bytes(native, vllm_config, **settings)

    @wraps(original_allocate)
    def allocate(runner, kv_cache_config):
        allocation = allocate_cache_views(kv_cache_config, runner.device)
        runner.oscar_cache_allocation = allocation
        return allocation

    @wraps(original_reshape)
    def reshape(runner, kv_cache_config, allocation):
        if not isinstance(allocation, CacheAllocation):
            raise RuntimeError("OSCAR reshape received an unverified native allocation")
        return allocation.views

    @wraps(original_initialize)
    def initialize(runner, kv_cache_config):
        if runner.vllm_config.kv_transfer_config is not None:
            raise RuntimeError("packed OSCAR cache connector lifecycle is not implemented")
        if getattr(runner, "enable_hamming_sparse", False):
            raise RuntimeError("Hamming KV compression cannot consume OSCAR FullCacheView")
        if block_zeroer_factory is None:
            raise RuntimeError("OSCAR native-page recycling/zeroing is not connected; refusing cache initialization")
        return original_initialize(runner, kv_cache_config)

    @wraps(original_zero_meta)
    def zero_meta(runner):
        if block_zeroer_factory is None:
            raise RuntimeError("OSCAR native-page recycling/zeroing is not connected")
        zeroer = block_zeroer_factory(runner, runner.oscar_cache_allocation)
        if not callable(getattr(zeroer, "zero_block_ids", None)):
            raise RuntimeError("OSCAR zeroer must implement native zero_block_ids")
        runner._kv_block_zeroer = zeroer

    replacements = (
        (kv_cache_utils, "get_kv_cache_config_from_groups", original_planner, planner),
        (kv_cache_utils, "_max_memory_usage_bytes_from_groups", original_admission, admission),
        (NPUModelRunner, "_allocate_kv_cache_tensors", original_allocate, allocate),
        (NPUModelRunner, "_reshape_kv_cache_tensors", original_reshape, reshape),
        (NPUModelRunner, "initialize_kv_cache", original_initialize, initialize),
        (NPUModelRunner, "_init_kv_zero_meta", original_zero_meta, zero_meta),
    )
    handle = CacheHookHandle(replacements)
    planner._oscar_cache_options = options
    planner._oscar_cache_handle = handle
    installed_replacements = []
    try:
        for owner, name, original, replacement in replacements:
            setattr(owner, name, replacement)
            installed_replacements.append((owner, name, original))
    except Exception:
        for owner, name, original in reversed(installed_replacements):
            setattr(owner, name, original)
        raise
    return handle
