"""Attach the target FULL pipeline to the unmodified NPU model runner.

The eager qwen3_5_mtp drafter reuses the same two-phase window machine: each
draft step is a q_len=1 row whose raw K/V enters pending, and the next round's
prepare commits exactly the target-accepted prefix. Prefix-cache hits restore
the BF16 Sink/Recent window through the staging pool (see cache_integration).
Device window-state errors are observed through a one-step-delayed pinned
mirror; no online K/V value ever reaches the host.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import hashlib
from pathlib import Path
from typing import Any

from .backend import RuntimeCapabilities, RuntimeNotReady
from .metadata import DeviceMetadata, RequestSlots
from .ops import (FullAttentionPipeline, LayerWindowState, PipelineMetadata,
                  ScratchBudget, SharedScratch, load_library)

OSCAR_REFERENCE_COMMIT = "57286d5d2cb08c3dcd8c17bb59e132d6985e6796"
ROTATION_OBJECTIVES = {"k": "qqt_r_h_pbr", "v": "sst_r_h_pbr"}


def validate_rotation_metadata(artifact: Any, config: Any, options: Any) -> None:
    if not isinstance(artifact, dict) or artifact.get("format") != "oscar-ascend-rotations-v1":
        raise RuntimeNotReady("unrecognized OSCAR rotation artifact format")
    config_path = Path(config.model_config.model) / "config.json"
    expected = {
        "model": config.model_config.model,
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "reference_commit": OSCAR_REFERENCE_COMMIT,
        "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
        "group_size": options.group_size,
        "objectives": ROTATION_OBJECTIVES,
        "clip_ratios": {"k": options.k_clip, "v": options.v_clip},
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise RuntimeNotReady(f"rotation artifact {field} does not match the target contract")


def check_rotation_on_npu(rotation: Any, operators: Any) -> float:
    """Initialization-only numerical check using the actual AscendC rotation.

    BF16 basis -> R -> R^T introduces rounding, so this is a max-error contract
    of 0.02, not a claim of mathematically exact orthogonality. No online K/V is
    read back. The one scalar transfer is outside the request hot path.
    """
    import torch
    d = rotation.shape[0]
    basis = torch.eye(d, dtype=torch.bfloat16, device=rotation.device).view(d, 1, d)
    rotated, reconstructed = torch.empty_like(basis), torch.empty_like(basis)
    operators.rotate_out(basis, rotation, rotated, False)
    operators.rotate_out(rotated, rotation, reconstructed, True)
    error = float((reconstructed.float() - basis.float()).abs().amax().item())
    if not 0 <= error <= 0.02:  # also rejects NaN and infinity
        raise RuntimeNotReady(f"rotation failed AscendC orthogonality roundtrip: max_error={error}")
    return error


def library_path(options: Any) -> Path:
    if options.library_path:
        return Path(options.library_path)
    return Path(__file__).resolve().parents[1] / "build/ascendc/liboscar_ascend_ops.so"


def _full_groups(groups):
    return [g for g in groups if type(g.kv_cache_spec).__name__ == "FullAttentionSpec"]


def scratch_budgets(config: Any, groups: Any, options: Any, operators: Any):
    max_requests = config.scheduler_config.max_num_seqs
    max_tokens = config.scheduler_config.max_num_batched_tokens
    pending = 1 + (config.speculative_config.num_speculative_tokens
                   if config.speculative_config else 0)
    heads = config.model_config.get_num_attention_heads(config.parallel_config)
    result = {}
    for group in _full_groups(groups):
        spec = group.kv_cache_spec
        key = (heads, spec.num_kv_heads, spec.head_size)
        if key in result:
            continue
        workspace = max(
            operators.workspace_size(max_tokens, heads, spec.head_size, 1),
            operators.workspace_size(min(max_tokens, max_requests * pending),
                                     heads, spec.head_size, 8),
        )
        result[key] = ScratchBudget(max_tokens, max_requests, heads, spec.num_kv_heads,
                                    spec.head_size, options.recent, pending, workspace)
    return result


def reserve_bytes(config: Any, groups: Any, options: Any) -> int:
    """Admit runtime tensors before the cache manager chooses its block count."""
    operators = load_library(library_path(options), require_window=True)
    budgets = scratch_budgets(config, groups, options, operators)
    total = sum(b.tensor_bytes for b in budgets.values())
    w = config.scheduler_config.max_num_seqs
    n = config.scheduler_config.max_num_batched_tokens
    pending = 1 + (config.speculative_config.num_speculative_tokens
                   if config.speculative_config else 0)
    capacity = options.sink + options.recent + pending
    # Persistent device row maps plus dummy-capture maps. Pinned host mirrors
    # are host RAM and are intentionally not included in the HBM amount.
    total += 2 * w * (4 + 8)
    for group in _full_groups(groups):
        spec = group.kv_cache_spec
        # Backend retains native virtual block size 128; reserve that larger
        # table width, even if the manager page is a multiple such as 768.
        pages = (config.model_config.max_model_len + 127) // 128
        total += (w + 1) * 4 + w * 4 + w * pages * 4 + n * 8 + w * (4 + 8 + 1) + 8
        for _ in group.layer_names:
            total += w * capacity * 4 + w * 4 * 8
            # Lossy prefix-restore counter and one row of the error mirror.
            total += 4 + w * 8
            total += 2 * spec.head_size * spec.head_size * 4
    # Allow actual tensor allocator alignment for every planned allocation.
    allocations = 4 + len(budgets) * 13 + len(_full_groups(groups)) * 8
    allocations += sum(len(g.layer_names) * 4 for g in _full_groups(groups))
    # Peak temporary basis/reconstruction and FP32 error buffers for one matrix.
    max_dim = max((g.kv_cache_spec.head_size for g in _full_groups(groups)), default=0)
    return total + allocations * 512 + max_dim * max_dim * 22


def collect_window_error_reports(error_rows: list[list[int]], layer_names: tuple[str, ...],
                                 slot_request_ids: list[str | None]) -> list[str]:
    """Turn one mirrored error-column snapshot into diagnosable reports.

    Codes follow the AscendC window kernel: 1 commit beyond materialized KV,
    2 window-position invariant, 3 verify length beyond pending budget,
    4 invalid block-table entry, 6 prefix restore failure.
    """
    descriptions = {1: "commit exceeds materialized pending K/V",
                    2: "window position invariant violated",
                    3: "multi-query row exceeds the pending budget",
                    4: "block table entry invalid for migration",
                    6: "prefix window restore failed"}
    reports = []
    for layer_index, row in enumerate(error_rows):
        for window_row, code in enumerate(row):
            if code:
                request = (slot_request_ids[window_row]
                           if window_row < len(slot_request_ids) else None)
                reports.append(
                    f"layer={layer_names[layer_index]} window_row={window_row} "
                    f"request={request!r} code={code} "
                    f"({descriptions.get(code, 'unknown window error')})")
    return reports


class TargetRuntime:
    capabilities = RuntimeCapabilities(
        window_commit=True, prefill=True, chunked_prefill=True,
        decode=True, verify=True, request_lifecycle=True, graph_capture=True,
        draft=True, prefix_lifecycle=True,
    )

    def __init__(self, runner: Any, options: Any):
        self.runner, self.config, self.options = runner, runner.vllm_config, options
        self.identities = RequestSlots(self.config.scheduler_config.max_num_seqs)
        self.configured = False
        self._slot_request_ids = [None] * self.identities.capacity
        self._slot_epochs = [0] * self.identities.capacity
        self.metadata_by_layer = {}
        self.group_id_by_layer = {}
        self.state_by_layer = {}
        self.pipeline_by_layer = {}
        self.rotations = {}
        self._addresses = {}
        self._layer_order = ()
        self._errors_device = None
        self._error_mirror = None
        self._error_mirror_valid = False

    def configure(self, cache_config: Any) -> None:
        if self.configured:
            raise RuntimeError("OSCAR runtime cache initialization was repeated")
        if not self.options.rotations_path:
            raise RuntimeNotReady("model-matched OSCAR rotation checkpoint is required")
        import torch

        device, options = self.runner.device, self.options
        operators = load_library(library_path(options), require_window=True)
        # Deserializing rotation constants is not K/V processing. Load storage
        # on host so other ranks' matrices never occupy this rank's HBM.
        artifact = torch.load(options.rotations_path, map_location="cpu", weights_only=True)
        validate_rotation_metadata(artifact, self.config, options)
        rank = self.config.parallel_config.rank
        layer_rotations = artifact.get("ranks", {}).get(str(rank))
        if not isinstance(layer_rotations, dict):
            raise RuntimeNotReady(f"rotation artifact has no rank {rank}")
        expected_layers = {layer for group in _full_groups(cache_config.kv_cache_groups)
                           for layer in group.layer_names}
        if set(layer_rotations) != expected_layers:
            raise RuntimeNotReady("rotation artifact must cover exactly this rank's FULL layers")
        w, n = self.config.scheduler_config.max_num_seqs, self.config.scheduler_config.max_num_batched_tokens
        pending = 1 + (self.config.speculative_config.num_speculative_tokens
                       if self.config.speculative_config else 0)
        capacity = options.sink + options.recent + pending
        self.row_map = torch.full((w,), -1, dtype=torch.int32, device=device)
        self.epochs = torch.zeros(w, dtype=torch.int64, device=device)
        self.capture_row_map = torch.full_like(self.row_map, -1)
        self.capture_epochs = torch.zeros_like(self.epochs)
        self.row_map_host = torch.full((w,), -1, dtype=torch.int32, pin_memory=True)
        self.epochs_host = torch.zeros(w, dtype=torch.int64, pin_memory=True)
        budgets = scratch_budgets(self.config, cache_config.kv_cache_groups, options, operators)
        pipelines = {
            key: FullAttentionPipeline(operators, SharedScratch.allocate(budget, device),
                                       sink=options.sink, recent=options.recent,
                                       max_pending=pending, k_clip=options.k_clip,
                                       v_clip=options.v_clip)
            for key, budget in budgets.items()
        }
        heads = self.config.model_config.get_num_attention_heads(self.config.parallel_config)
        for group_id, group in enumerate(cache_config.kv_cache_groups):
            if type(group.kv_cache_spec).__name__ != "FullAttentionSpec":
                continue
            spec = group.kv_cache_spec
            pages = (self.config.model_config.max_model_len + 127) // 128
            fixed = DeviceMetadata.allocate(max_requests=w, max_tokens=n,
                                             max_blocks=pages, device=device)
            for layer in group.layer_names:
                rotations = layer_rotations.get(layer)
                if not isinstance(rotations, dict) or set(rotations) != {"k", "v"}:
                    raise RuntimeNotReady(f"missing K/V rotations for {layer}")
                for rotation in rotations.values():
                    if (tuple(rotation.shape) != (spec.head_size, spec.head_size)
                            or rotation.dtype != torch.float32
                            or not rotation.is_contiguous()):
                        raise RuntimeNotReady(f"rotation shape/dtype/device mismatch for {layer}")
                self.rotations[layer] = {name: rotation.to(device=device)
                                         for name, rotation in rotations.items()}
                for rotation in self.rotations[layer].values():
                    check_rotation_on_npu(rotation, operators)
                self.metadata_by_layer[layer] = fixed
                self.group_id_by_layer[layer] = group_id
                self.state_by_layer[layer] = LayerWindowState.allocate(
                    max_requests=w, capacity=capacity, device=device)
                self.pipeline_by_layer[layer] = pipelines[(heads, spec.num_kv_heads, spec.head_size)]
                self._addresses[layer] = fixed.addresses()
        # One-step-delayed observation of device window-state error codes.
        # The int64 column is transaction metadata; K/V values never cross
        # this path. The mirror copy is enqueued on the same stream as the
        # forward kernels, so reading it one scheduler step later cannot
        # observe a partially transferred entry.
        self._layer_order = tuple(self.state_by_layer)
        self._errors_device = torch.zeros((len(self._layer_order), w),
                                          dtype=torch.int64, device=device)
        self._error_mirror = torch.zeros((len(self._layer_order), w),
                                         dtype=torch.int64, pin_memory=True)
        self._error_mirror_valid = False
        self.configured = True

    def update_lifecycle(self, scheduler_output: Any) -> None:
        retired = set(scheduler_output.finished_req_ids)
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted:
            retired.update(preempted)
        # These loops process lifecycle changes, not every scheduled request or
        # any K/V element. The steady-state row mapping below is a bulk join.
        for request_id in retired:
            try:
                old = self.identities.rows((request_id,))[0]
            except KeyError:
                continue
            self._slot_request_ids[old.index] = None
        self.identities.retire(tuple(retired))
        for request in scheduler_output.scheduled_new_reqs:
            slot = self.identities.begin(request.req_id)
            self._slot_request_ids[slot.index] = request.req_id
            self._slot_epochs[slot.index] = slot.epoch
        for request_id in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            try:
                old = self.identities.rows((request_id,))[0]
                self._slot_request_ids[old.index] = None
            except KeyError:
                pass
            slot = self.identities.begin(request_id, resumed=True)
            self._slot_request_ids[slot.index] = request_id
            self._slot_epochs[slot.index] = slot.epoch

    def refresh_rows(self) -> None:
        if not self.configured:
            return
        # This bulk join uses CPU request-ID metadata only. Unlike a Python loop
        # over each live request, it also handles row reorder and paused returns
        # without adding per-request hot-path callbacks or reading NPU values.
        import numpy as np
        requests = np.asarray(self.runner.input_batch.req_ids, dtype=object)
        owners = np.asarray(self._slot_request_ids, dtype=object)
        matches = requests[:, None] == owners[None, :]
        if not np.all(matches.sum(axis=1) == 1):
            raise RuntimeError("OSCAR batch contains a missing or duplicate request identity")
        slots = matches.argmax(axis=1)
        self.row_map_host.fill_(-1)
        self.epochs_host.zero_()
        self.row_map_host.numpy()[:len(requests)] = slots
        self.epochs_host.numpy()[:len(requests)] = np.asarray(self._slot_epochs, dtype=np.int64)[slots]
        self.row_map.copy_(self.row_map_host, non_blocking=True)
        self.epochs.copy_(self.epochs_host, non_blocking=True)
        self.observe_window_errors()

    def observe_window_errors(self) -> None:
        """Publish device error codes asynchronously and judge the previous copy.

        refresh_rows runs between steps on the same stream as the forward
        kernels. The copy enqueued in the previous call necessarily completed
        before this step's kernels started, so the pinned mirror read below
        never races an in-flight transfer and adds no device synchronization.
        """
        for index, layer in enumerate(self._layer_order):
            self._errors_device[index].copy_(
                self.state_by_layer[layer].transactions[:, 3], non_blocking=True)
        self._error_mirror.copy_(self._errors_device, non_blocking=True)
        if self._error_mirror_valid:
            reports = collect_window_error_reports(
                self._error_mirror.tolist(), self._layer_order, self._slot_request_ids)
            if reports:
                raise RuntimeError(
                    "OSCAR NPU window state reported transaction errors: "
                    + "; ".join(reports))
        self._error_mirror_valid = True

    def check_window_errors(self) -> None:
        """Synchronized check for probes and shutdown; outside the hot path."""
        import torch
        for index, layer in enumerate(self._layer_order):
            self._errors_device[index].copy_(
                self.state_by_layer[layer].transactions[:, 3], non_blocking=True)
        self._error_mirror.copy_(self._errors_device, non_blocking=True)
        torch.npu.synchronize()
        reports = collect_window_error_reports(
            self._error_mirror.tolist(), self._layer_order, self._slot_request_ids)
        if reports:
            raise RuntimeError(
                "OSCAR NPU window state reported transaction errors: "
                + "; ".join(reports))

    def prepare(self, common: Any, layer_names: tuple[str, ...], *, for_capture: bool):
        if not self.configured:
            raise RuntimeNotReady("OSCAR runtime physical buffers have not been configured")
        drafter = getattr(self.runner, "drafter", None)
        draft_layers = set(getattr(drafter, "attn_layer_names", ()))
        # Target cache groups may include a draft layer with the same spec. The
        # drafter's independently built metadata contains draft layers only.
        if layer_names and set(layer_names) <= draft_layers:
            self.capabilities.require("draft")
            if for_capture:
                # The target config runs the drafter eager (speculative
                # enforce_eager=true). Captured draft multi-step graphs keep
                # per-step slot mappings outside our fixed metadata and are
                # unverified, so they fail closed instead of replaying stale
                # windows.
                raise RuntimeNotReady("OSCAR drafter graph capture is not implemented")
        if self.config.cache_config.enable_prefix_caching:
            self.capabilities.require("prefix_lifecycle")
        fixed = self.metadata_by_layer[layer_names[0]]
        fixed.update(common,
                     row_to_window=self.capture_row_map if for_capture else self.row_map,
                     epochs=self.capture_epochs if for_capture else self.epochs)
        rows = common.num_reqs
        table_size = self.runner.input_batch.block_table[
            self.group_id_by_layer[layer_names[0]]].block_size
        # Keep whole fixed-width block-table rows contiguous. The unused page
        # columns are zero and never accessed beyond native sequence lengths.
        prepared = PipelineMetadata(
            fixed.seq_lens[:rows], fixed.query_start_loc[:rows + 1],
            fixed.row_to_window[:rows], fixed.epochs[:rows], fixed.block_table[:rows],
            fixed.slot_mapping[:common.num_input_tokens], fixed.is_prefilling[:rows],
            common.max_query_len, table_size,
        )
        return prepared

    def forward(self, *, layer, query, key, value, cache, metadata, output, scale):
        name = layer.layer_name
        if name in set(getattr(getattr(self.runner, "drafter", None), "attn_layer_names", ())):
            self.capabilities.require("draft")
        if self.metadata_by_layer[name].addresses() != self._addresses[name]:
            raise RuntimeError("OSCAR captured metadata address changed")
        rotations = self.rotations[name]
        return self.pipeline_by_layer[name].forward(
            query=query, key=key, value=value, cache=cache,
            state=self.state_by_layer[name], metadata=metadata,
            rotation_k=rotations["k"], rotation_v=rotations["v"], output=output, scale=scale)

    def check_graph_addresses(self) -> None:
        for layer, metadata in self.metadata_by_layer.items():
            if metadata.addresses() != self._addresses[layer]:
                raise RuntimeError("OSCAR graph metadata address changed")


@dataclass
class RunnerHooks:
    owner: type
    originals: dict
    replacements: dict

    def undo(self):
        for name, replacement in self.replacements.items():
            if getattr(self.owner, name) is not replacement:
                raise RuntimeError(f"cannot undo OSCAR runner hook replaced by another component: {name}")
        for name, original in self.originals.items():
            setattr(self.owner, name, original)


def install_runtime_hooks(options: Any) -> RunnerHooks:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    cls = NPUModelRunner
    names = ("__init__", "initialize_kv_cache", "_update_states", "_prepare_inputs")
    originals = {name: getattr(cls, name) for name in names}
    installed = getattr(originals["__init__"], "_oscar_runtime_hooks", None)
    if installed is not None:
        return installed

    @wraps(originals["__init__"])
    def initialize(runner, *args, **kwargs):
        originals["__init__"](runner, *args, **kwargs)
        runner.vllm_config._oscar_attention_runtime = TargetRuntime(runner, options)

    @wraps(originals["initialize_kv_cache"])
    def cache(runner, kv_cache_config):
        runner.vllm_config._oscar_attention_runtime.configure(kv_cache_config)
        return originals["initialize_kv_cache"](runner, kv_cache_config)

    @wraps(originals["_update_states"])
    def states(runner, scheduler_output):
        result = originals["_update_states"](runner, scheduler_output)
        runner.vllm_config._oscar_attention_runtime.update_lifecycle(scheduler_output)
        return result

    @wraps(originals["_prepare_inputs"])
    def prepare(runner, *args, **kwargs):
        result = originals["_prepare_inputs"](runner, *args, **kwargs)
        runner.vllm_config._oscar_attention_runtime.refresh_rows()
        return result

    replacements = dict(zip(names, (initialize, cache, states, prepare)))
    handle = RunnerHooks(cls, originals, replacements)
    initialize._oscar_runtime_hooks = handle
    for name, replacement in replacements.items():
        setattr(cls, name, replacement)
    return handle
