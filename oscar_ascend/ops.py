"""AscendC library loading and physical workspace planning.

Only explicit NPU custom operators may process K/V. CPU-only installations can
inspect schemas and budgets without importing torch. Numerical acceptance is
performed by the separate target-machine probe, never by this loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIMITIVE_OPS = (
    "rotate_out", "store_int2", "history_attention_out",
    "window_attention_out", "merge_out", "zero_blocks_out", "workspace_size", "abi_version",
)
WINDOW_OP = "window_state_out"
PREFIX_OPS = ("dequant_history_out", "stage_window_out", "prefix_restore_out")
ABI_VERSION = 2


def load_library(path: str | Path, *, require_window: bool = False) -> Any:
    """Load a specified artifact, then require real registered operator schemas."""
    library = Path(path).expanduser().resolve(strict=True)
    if not library.is_file():
        raise ValueError("AscendC artifact must be a shared library file")
    import torch
    import torch_npu  # noqa: F401 - registers the actual NPU backend

    torch.ops.load_library(str(library))
    namespace = torch.ops.oscar_ascend
    required = PRIMITIVE_OPS + ((WINDOW_OP,) if require_window else ())
    missing = [name for name in required if not hasattr(namespace, name)]
    if missing:
        raise RuntimeError("AscendC library is missing required schemas: " + ", ".join(missing))
    if namespace.abi_version() != ABI_VERSION:
        raise RuntimeError(f"AscendC operator ABI mismatch (expected version {ABI_VERSION})")
    return namespace


@dataclass(frozen=True)
class ScratchBudget:
    """Exact tensor payload budget; allocator alignment must be added separately."""
    max_tokens: int
    max_requests: int
    query_heads: int
    kv_heads: int
    head_dim: int
    recent: int
    max_queries: int
    kernel_workspace: int

    def __post_init__(self):
        for name in ("max_tokens", "max_requests", "query_heads", "kv_heads",
                     "head_dim", "max_queries"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recent < 0 or self.kernel_workspace < 0:
            raise ValueError("negative scratch budget")

    @property
    def migration_tokens(self) -> int:
        # A long chunk may evict all R old recent rows. Verify commits <= M+1.
        return self.max_requests * (self.recent + self.max_queries)

    @property
    def tensor_bytes(self) -> int:
        q_elements = self.max_tokens * self.query_heads * self.head_dim
        stats = self.max_tokens * self.query_heads
        migration_elements = self.migration_tokens * self.kv_heads * self.head_dim
        return (
            q_elements * 2                 # BF16 rotated Q
            + q_elements * 4 * 2          # FP32 history and window output
            + stats * 4 * 2               # independent logsumexp
            + migration_elements * 2 * 2 # BF16 migration K and V
            + self.migration_tokens * 8  # migration physical slots
            + self.max_tokens * (8 + 4)  # current masked slots and query positions
            + self.max_requests * 4 * 2  # history start and end
            + self.kernel_workspace
        )


@dataclass
class LayerWindowState:
    positions: Any
    transactions: Any
    lossy: Any

    @classmethod
    def allocate(cls, *, max_requests: int, capacity: int, device: Any):
        import torch

        if str(device).split(":", 1)[0] != "npu":
            raise ValueError("window state must stay on NPU")
        if min(max_requests, capacity) <= 0:
            raise ValueError("window state capacities must be positive")
        return cls(
            torch.full((max_requests, capacity), -1, dtype=torch.int32, device=device),
            torch.zeros((max_requests, 4), dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )


@dataclass
class SharedScratch:
    budget: ScratchBudget
    query_rotated: Any
    history_output: Any
    window_output: Any
    history_lse: Any
    window_lse: Any
    migration_k: Any
    migration_v: Any
    migration_slots: Any
    current_history_slots: Any
    query_positions: Any
    history_starts: Any
    history_ends: Any
    workspace: Any

    @classmethod
    def allocate(cls, budget: ScratchBudget, device: Any):
        import torch

        if str(device).split(":", 1)[0] != "npu":
            raise ValueError("attention scratch must stay on NPU")
        n, h, d = budget.max_tokens, budget.query_heads, budget.head_dim
        m = budget.migration_tokens
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        return cls(
            budget, empty((n, h, d), torch.bfloat16),
            empty((n, h, d), torch.float32), empty((n, h, d), torch.float32),
            empty((n, h), torch.float32), empty((n, h), torch.float32),
            empty((m, budget.kv_heads, d), torch.bfloat16),
            empty((m, budget.kv_heads, d), torch.bfloat16),
            empty(m, torch.int64), empty(n, torch.int64), empty(n, torch.int32),
            empty(budget.max_requests, torch.int32),
            empty(budget.max_requests, torch.int32),
            empty(budget.kernel_workspace, torch.uint8),
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for name, value in vars(self).items()
                   if name != "budget")


@dataclass(frozen=True)
class PipelineMetadata:
    """Device-only inputs already reconciled with native request identities."""
    seq_lens: Any
    query_start_loc: Any
    row_to_window: Any
    epochs: Any
    block_table: Any
    slot_mapping: Any
    is_prefilling: Any
    max_query_len: int
    block_table_block_size: int


class FullAttentionPipeline:
    """Execute the actual AscendC target FULL path, with bounded two-phase KV.

    This class is directly probeable without loading a vLLM model. Callers own
    request-identity reconciliation, rotations, cache admission, stream ordering,
    graph-state isolation and device-error observation. The full service adapter
    must satisfy these contracts before attaching this pipeline to the backend.
    """

    def __init__(self, operators: Any, scratch: SharedScratch, *,
                 sink: int, recent: int, max_pending: int,
                 k_clip: float = 1.0, v_clip: float = 1.0):
        if sink < 0 or recent < 0 or not 1 <= max_pending <= 16:
            raise ValueError("invalid window configuration")
        if not 0 <= k_clip <= 1 or not 0 <= v_clip <= 1:
            raise ValueError("OSCAR clip ratios must be in [0, 1]")
        missing = [name for name in PRIMITIVE_OPS + (WINDOW_OP,) + PREFIX_OPS
                   if not hasattr(operators, name)]
        if missing:
            raise RuntimeError("incomplete NPU operator bundle: " + ", ".join(missing))
        if operators.abi_version() != ABI_VERSION:
            raise RuntimeError("unsupported OSCAR AscendC ABI")
        if scratch.budget.recent != recent or scratch.budget.max_queries != max_pending:
            raise ValueError("migration scratch does not match the window configuration")
        self.operators = operators
        self.scratch = scratch
        self.sink, self.recent, self.max_pending = sink, recent, max_pending
        self.k_clip, self.v_clip = float(k_clip), float(v_clip)

    def forward(self, *, query: Any, key: Any, value: Any, cache: Any,
                state: LayerWindowState, metadata: PipelineMetadata,
                rotation_k: Any, rotation_v: Any, output: Any, scale: float) -> Any:
        """No CPU K/V arithmetic, history materialization, or tensor .item()."""
        n, h, d = query.shape
        rows = metadata.seq_lens.shape[0]
        s, ops, budget = self.scratch, self.operators, self.scratch.budget
        if query.device.type != "npu":
            raise ValueError("OSCAR FULL pipeline requires actual NPU tensors")
        if (n > budget.max_tokens or rows > budget.max_requests
                or h != budget.query_heads or d != budget.head_dim):
            raise ValueError("query exceeds the admitted shared workspace shape")
        if key.shape != value.shape or key.shape != (n, budget.kv_heads, d):
            raise ValueError("raw K/V shape does not match the pipeline")
        if metadata.max_query_len < 1:
            raise ValueError("max_query_len must be positive")
        if metadata.block_table_block_size <= 0:
            raise ValueError("native block table granularity is required")
        if tuple(cache.history.shape[2:]) != (budget.kv_heads, 2 * (d // 4 + 4)):
            raise ValueError("packed History layout does not match the AscendC ABI")
        if cache.layout.head_size != d or cache.layout.head_size_v != d:
            raise ValueError("the AscendC ABI requires equal K and V head dimensions")
        if cache.layout.group_size < d:
            raise ValueError("the AscendC ABI uses one quantization group per head")
        if cache.sink_recent_k.shape[1] != self.sink + self.recent + self.max_pending:
            raise ValueError("window allocation does not match the pipeline")
        staging = (cache.staging_k, cache.staging_v, cache.staging_owner)
        if any(tensor is None for tensor in staging):
            raise ValueError("prefix staging pool is required by the OSCAR pipeline")
        if (tuple(cache.staging_k.shape[2:]) != (budget.kv_heads, d)
                or cache.staging_k.shape[:2] != cache.staging_owner.shape
                or cache.staging_k.shape != cache.staging_v.shape
                or cache.staging_owner.shape[1] != cache.layout.block_size):
            raise ValueError("staging pool geometry does not match the cache layout")
        if tuple(output.shape) not in ((n, h, d), (n, h * d)):
            raise ValueError("native attention output shape is incompatible")
        # Shape-only decisions are fixed for each captured descriptor. No device
        # scalar is converted to Python. Long prefill uses one split to bound HBM.
        splits = 8 if metadata.max_query_len <= self.max_pending else 1
        required = ops.workspace_size(n, h, d, splits)
        if required > budget.kernel_workspace:
            raise ValueError("kernel workspace exceeds the reserved physical budget")
        migration_count = rows * (self.recent + self.max_pending)
        migration_k = s.migration_k[:migration_count]
        migration_v = s.migration_v[:migration_count]
        migration_slots = s.migration_slots[:migration_count]
        current_slots = s.current_history_slots[:n]
        qpos = s.query_positions[:n]
        hist_start, hist_end = s.history_starts[:rows], s.history_ends[:rows]
        qrot = s.query_rotated[:n]
        hout, wout = s.history_output[:n], s.window_output[:n]
        hlse, wlse = s.history_lse[:n], s.window_lse[:n]
        # Ensure padding never inherits the previous batch's task descriptors.
        qpos.fill_(-1)
        hist_start.zero_()
        hist_end.zero_()
        migration_slots.fill_(-1)
        current_slots.fill_(-1)

        def window_phase(phase):
            ops.window_state_out(
                key, value, metadata.seq_lens, metadata.query_start_loc,
                metadata.row_to_window, metadata.epochs, metadata.block_table,
                metadata.slot_mapping, metadata.is_prefilling,
                cache.sink_recent_k, cache.sink_recent_v, state.positions,
                state.transactions, hist_start, hist_end, qpos,
                migration_k, migration_v, migration_slots, current_slots,
                phase, self.sink, self.recent, self.max_pending,
                metadata.block_table_block_size,
            )

        def store(k, v, slots):
            ops.store_int2(k, v, rotation_k, rotation_v, slots, cache.history,
                           self.k_clip, self.v_clip)

        # Mirror raw Sink/Recent rows into the staging pool before anything can
        # recycle this step's inputs, then rebuild windows for prefix-hit rows.
        # Restore must precede phase 0: it establishes the epoch/committed state
        # whose invariants phase 0 re-checks. Rows without an identity change
        # return immediately inside the kernel.
        ops.stage_window_out(
            key, value, metadata.seq_lens, metadata.query_start_loc,
            metadata.slot_mapping, cache.staging_k, cache.staging_v,
            cache.staging_owner, self.sink, self.recent)
        ops.prefix_restore_out(
            metadata.seq_lens, metadata.query_start_loc, metadata.row_to_window,
            metadata.epochs, metadata.block_table, state.positions,
            cache.sink_recent_k, cache.sink_recent_v, state.transactions,
            cache.staging_k, cache.staging_v, cache.staging_owner,
            cache.history, rotation_k, rotation_v, state.lossy,
            self.sink, self.recent, self.max_pending,
            metadata.block_table_block_size)

        window_phase(0)
        store(migration_k, migration_v, migration_slots)
        ops.rotate_out(query, rotation_k, qrot, False)
        ops.history_attention_out(
            qrot, cache.history, metadata.block_table, metadata.query_start_loc,
            hist_start, hist_end, qpos, hout, hlse, s.workspace,
            float(scale), splits, metadata.max_query_len,
            metadata.block_table_block_size,
        )
        ops.window_attention_out(
            query, key, value, cache.sink_recent_k, cache.sink_recent_v,
            state.positions, metadata.row_to_window, metadata.query_start_loc,
            qpos, wout, wlse, float(scale), s.workspace, metadata.max_query_len,
        )
        ops.merge_out(hout, hlse, wout, wlse, rotation_v, output.view(n, h, d))
        # Preserve the old window until attention has consumed it.
        window_phase(1)
        store(migration_k, migration_v, migration_slots)
        store(key, value, current_slots)
        return output
