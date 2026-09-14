"""Scheduling contracts and fixed-address NPU metadata for OSCAR FULL layers.

The host helpers handle scheduler metadata only. They never accept K/V values.
Importing this module does not initialize torch, vLLM, or the NPU platform.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq
from typing import Any, Sequence


class Stage(IntEnum):
    PAD = 0
    PREFILL = 1
    CHUNKED_PREFILL = 2
    DECODE = 3
    VERIFY = 4
    DRAFT = 5


def classify_rows(
    query_start_loc: Sequence[int],
    computed_tokens: Sequence[int],
    prompt_tokens: Sequence[int],
    scheduled_drafts: Sequence[int],
    *,
    is_draft: bool = False,
) -> tuple[Stage, ...]:
    """Classify CPU scheduler metadata, independently of graph dispatch labels."""
    rows = len(computed_tokens)
    if not (len(query_start_loc) == rows + 1
            and len(prompt_tokens) == rows and len(scheduled_drafts) == rows):
        raise ValueError("inconsistent request metadata lengths")
    if query_start_loc[0] != 0:
        raise ValueError("query_start_loc must start at zero")
    result = []
    for i, (computed, prompt, drafts) in enumerate(
        zip(computed_tokens, prompt_tokens, scheduled_drafts)
    ):
        q_len = query_start_loc[i + 1] - query_start_loc[i]
        if min(q_len, computed, prompt, drafts) < 0:
            raise ValueError("negative scheduler metadata")
        if q_len == 0:
            result.append(Stage.PAD)
        elif is_draft:
            result.append(Stage.DRAFT)
        elif computed < prompt:
            result.append(Stage.PREFILL if computed == 0 else Stage.CHUNKED_PREFILL)
        elif drafts:
            if drafts + 1 != q_len:
                raise ValueError("verify q_len must equal scheduled drafts plus one")
            result.append(Stage.VERIFY)
        elif q_len == 1:
            result.append(Stage.DECODE)
        else:
            raise ValueError("multi-query target row lacks prefill or verify metadata")
    return tuple(result)


@dataclass(frozen=True)
class WindowTransition:
    """A host-only oracle for one pending transaction, not the online KV path."""
    old_computed: int
    next_computed: int
    pending_queries: int
    sink: int
    recent: int

    def __post_init__(self):
        if min(self.old_computed, self.next_computed, self.pending_queries,
               self.sink, self.recent) < 0:
            raise ValueError("negative window bound")
        if not self.old_computed <= self.next_computed <= (
            self.old_computed + self.pending_queries
        ):
            raise ValueError("commit lies outside the materialized pending K/V range")

    @property
    def accepted_kv(self) -> int:
        return self.next_computed - self.old_computed

    @property
    def rejected_kv(self) -> int:
        return self.pending_queries - self.accepted_kv

    @property
    def newly_historical(self) -> tuple[int, int]:
        old_end = max(self.sink, self.old_computed - self.recent)
        new_end = max(self.sink, self.next_computed - self.recent)
        return old_end, new_end


def restore_window_ranges(committed: int, sink: int, recent: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Host mirror of the NPU prefix-restore ranges for one prefix-hit row.

    Sink covers [0, min(S, C)); Recent covers [max(S, C-R), C) and is empty
    when R=0. Everything between stays compressed INT2 History in the shared
    pages and is never rebuilt as BF16.
    """
    if min(committed, sink, recent) < 0:
        raise ValueError("negative restore bound")
    recent_start = min(max(sink, committed - recent), committed) if recent else committed
    return (0, min(sink, committed)), (recent_start, committed)


@dataclass(frozen=True)
class RequestSlot:
    index: int
    epoch: int


class RequestSlots:
    """Stable request identities across row reordering and request-ID reuse.

    Unscheduled requests retain slots. Explicit finish/preemption retires them.
    In v1, the runner does not always receive preemption IDs: callers must use
    scheduler lifecycle evidence, never infer preemption from an absent row.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("request slot capacity must be positive")
        self.capacity = capacity
        self._free = list(range(capacity))
        self._epochs = [0] * capacity
        self._requests: dict[str, RequestSlot] = {}

    def retire(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            old = self._requests.pop(request_id, None)
            if old is not None:
                heapq.heappush(self._free, old.index)

    def begin(self, request_id: str, *, resumed: bool = False) -> RequestSlot:
        old = self._requests.get(request_id)
        if old is not None and not resumed:
            return old
        if old is not None:
            self.retire((request_id,))
        if not self._free:
            raise RuntimeError("OSCAR request window capacity exhausted; no silent eviction")
        index = heapq.heappop(self._free)
        self._epochs[index] += 1
        slot = RequestSlot(index, self._epochs[index])
        self._requests[request_id] = slot
        return slot

    def rows(self, request_ids: Sequence[str]) -> tuple[RequestSlot, ...]:
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate request IDs in batch")
        return tuple(self._requests[request_id] for request_id in request_ids)


@dataclass
class DeviceMetadata:
    """Persistent buffers. Only their contents change between graph replays."""
    query_start_loc: Any
    seq_lens: Any
    block_table: Any
    slot_mapping: Any
    row_to_window: Any
    epochs: Any
    is_prefilling: Any
    actual_sizes: Any
    max_requests: int
    max_tokens: int
    max_blocks: int

    @classmethod
    def allocate(cls, *, max_requests: int, max_tokens: int,
                 max_blocks: int, device: Any) -> "DeviceMetadata":
        import torch

        if min(max_requests, max_tokens, max_blocks) <= 0:
            raise ValueError("metadata capacities must be positive")
        if torch.device(device).type != "npu":
            raise ValueError("OSCAR online metadata must be allocated on NPU")
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(
            zeros(max_requests + 1, torch.int32),
            zeros(max_requests, torch.int32),
            zeros((max_requests, max_blocks), torch.int32),
            torch.full((max_tokens,), -1, dtype=torch.int64, device=device),
            torch.full((max_requests,), -1, dtype=torch.int32, device=device),
            zeros(max_requests, torch.int64),
            zeros(max_requests, torch.bool),
            zeros(2, torch.int32),
            max_requests, max_tokens, max_blocks,
        )

    def addresses(self) -> tuple[int, ...]:
        return tuple(getattr(self, field).data_ptr() for field in (
            "query_start_loc", "seq_lens", "block_table", "slot_mapping",
            "row_to_window", "epochs", "is_prefilling", "actual_sizes",
        ))

    def update(self, common: Any, *, row_to_window: Any, epochs: Any) -> None:
        """Copy native NPU metadata without reading seq_lens back to the host.

        Caller supplies fixed device row maps updated from scheduler identities.
        This is run outside graph capture; all copied buffers retain addresses.
        Padding receives invalid window/slot identities, not request zero.
        """
        rows = common.num_reqs
        tokens = common.num_actual_tokens
        cols = common.block_table_tensor.shape[1]
        if rows > self.max_requests or tokens > self.max_tokens or cols > self.max_blocks:
            raise ValueError("OSCAR metadata capacity exceeded")
        if common.is_prefilling is None:
            raise ValueError("is_prefilling is required; q_len alone is ambiguous")
        if common.seq_lens.device.type != "npu":
            raise ValueError("native device seq_lens must remain authoritative")
        if row_to_window.shape[0] < rows or epochs.shape[0] < rows:
            raise ValueError("request identity map is shorter than batch")
        self.query_start_loc.zero_()
        self.seq_lens.zero_()
        self.block_table.zero_()
        self.slot_mapping.fill_(-1)
        self.row_to_window.fill_(-1)
        self.epochs.zero_()
        self.is_prefilling.zero_()
        self.query_start_loc[:rows + 1].copy_(common.query_start_loc, non_blocking=True)
        # Fill cumulative padding boundaries with the final boundary, on device.
        self.query_start_loc[rows + 1:].copy_(self.query_start_loc[rows].expand(
            self.max_requests - rows), non_blocking=True)
        self.seq_lens[:rows].copy_(common.seq_lens, non_blocking=True)
        self.block_table[:rows, :cols].copy_(common.block_table_tensor, non_blocking=True)
        self.slot_mapping[:tokens].copy_(common.slot_mapping[:tokens], non_blocking=True)
        self.row_to_window[:rows].copy_(row_to_window[:rows], non_blocking=True)
        self.epochs[:rows].copy_(epochs[:rows], non_blocking=True)
        self.is_prefilling[:rows].copy_(common.is_prefilling[:rows], non_blocking=True)
        self.actual_sizes[0] = rows
        self.actual_sizes[1] = tokens
