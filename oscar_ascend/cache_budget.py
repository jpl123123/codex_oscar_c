"""Host-only shape and physical-byte planning; never accepts KV tensor values.

Native block IDs remain the index of every pool. Splitting backing allocations
removes GDN's unused V stripe without changing GDN state shapes or page ownership.
"""
from dataclasses import dataclass
from math import prod


DTYPE_BYTES = {"uint8": 1, "int8": 1, "float16": 2, "bfloat16": 2,
               "float32": 4, "int32": 4, "int64": 8}


def positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def align_up(value: int, alignment: int) -> int:
    nonnegative(value, "value")
    positive(alignment, "alignment")
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class FullLayout:
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    group_size: int
    alignment: int = 32

    def __post_init__(self):
        for name in ("block_size", "num_kv_heads", "head_size", "head_size_v",
                     "group_size", "alignment"):
            positive(getattr(self, name), name)
        if self.alignment % 4:
            raise ValueError("alignment must preserve all dtype boundaries")
        if self.head_size % 4 or self.head_size_v % 4:
            raise ValueError("current packed layout requires head dimensions divisible by 4")
        if self.group_size < max(self.head_size, self.head_size_v):
            raise ValueError("verified OSCAR ABI requires one quantization group per head")

    @property
    def bf16_bytes_per_token(self) -> int:
        return 2 * self.num_kv_heads * (self.head_size + self.head_size_v)

    @property
    def packed_bytes_per_token(self) -> int:
        return self.num_kv_heads * (self.head_size // 4 + self.head_size_v // 4)

    @property
    def metadata_bytes_per_token(self) -> int:
        # Independent FP16 affine scale + minimum for every K and V group.
        groups = sum((d + self.group_size - 1) // self.group_size
                     for d in (self.head_size, self.head_size_v))
        return self.num_kv_heads * groups * 4

    @property
    def slot_bytes(self) -> int:
        return (self.packed_bytes_per_token + self.metadata_bytes_per_token) // self.num_kv_heads

    def sections(self) -> tuple["Section", ...]:
        # Per-head slot: K codes, K scales, K minima, V codes, V scales,
        # V minima. The verified kernel accepts one group per head.
        return (Section("packed_kv", "uint8",
                        (self.block_size, self.num_kv_heads, self.slot_bytes),
                        self.alignment),)


@dataclass(frozen=True)
class Section:
    name: str
    dtype: str
    page_shape: tuple[int, ...]
    alignment: int = 1

    def __post_init__(self):
        if not self.name or self.dtype not in DTYPE_BYTES:
            raise ValueError("section needs a name and supported dtype")
        if not self.page_shape:
            raise ValueError("section shape must not be empty")
        for size in self.page_shape:
            positive(size, "section dimension")
        positive(self.alignment, "alignment")

    @property
    def payload_bytes(self) -> int:
        return prod(self.page_shape) * DTYPE_BYTES[self.dtype]

    @property
    def page_bytes(self) -> int:
        return align_up(self.payload_bytes, self.alignment)


@dataclass(frozen=True)
class Pool:
    name: str
    kind: str
    shared_by: tuple[str, ...]
    sections: tuple[Section, ...]
    layout: FullLayout | None = None

    def __post_init__(self):
        if self.kind not in ("gdn", "full") or not self.shared_by or not self.sections:
            raise ValueError("pool needs a kind, owners, and sections")
        if len(set(self.shared_by)) != len(self.shared_by):
            raise ValueError("duplicate layer in a pool")
        if len({s.name for s in self.sections}) != len(self.sections):
            raise ValueError("duplicate section in a pool")
        # Every section start must be valid for N=1; consequently for all N.
        offset = 0
        for section in self.sections:
            if offset % DTYPE_BYTES[section.dtype]:
                raise ValueError("state stripes have incompatible dtype alignment")
            offset += section.page_bytes

    @property
    def bytes_per_block(self) -> int:
        return sum(s.page_bytes for s in self.sections)

    def section_spans(self, num_blocks: int) -> tuple[tuple[str, int, int], ...]:
        positive(num_blocks, "num_blocks")
        spans = []
        offset = 0
        for section in self.sections:
            end = offset + num_blocks * section.page_bytes
            spans.append((section.name, offset, end))
            offset = end
        return tuple(spans)


def staging_rows(block_size: int, sink: int, recent: int, staging_tokens: int) -> int:
    """BF16 staging pool depth in blocks, mirroring the upstream PR's sizing.

    The floor keeps enough rows for one cached prefix's sink pages plus the
    recent tail pages (which can straddle one extra page); the configured
    staging budget can raise it. Evicted rows fall back to bounded INT2
    recovery and are counted, never reported as lossless.
    """
    positive(block_size, "block_size")
    nonnegative(sink, "sink")
    nonnegative(recent, "recent")
    nonnegative(staging_tokens, "staging_tokens")
    sink_pages = sink // block_size
    tail_pages = (recent + block_size - 1) // block_size + 1
    return max((staging_tokens + block_size - 1) // block_size,
               sink_pages + tail_pages + 2)


@dataclass(frozen=True)
class CachePlan:
    pools: tuple[Pool, ...]
    max_requests: int
    window_capacity: int
    runtime_reserve_bytes: int = 0
    staging_tokens: int = 0
    sink: int = 0
    recent: int = 0

    def __post_init__(self):
        positive(self.max_requests, "max_requests")
        positive(self.window_capacity, "window_capacity")
        nonnegative(self.runtime_reserve_bytes, "runtime_reserve_bytes")
        nonnegative(self.staging_tokens, "staging_tokens")
        nonnegative(self.sink, "sink")
        nonnegative(self.recent, "recent")
        if self.sink + self.recent >= self.window_capacity:
            raise ValueError("window capacity must exceed sink plus recent")
        names = [name for pool in self.pools for name in pool.shared_by]
        if not self.pools or len(names) != len(set(names)):
            raise ValueError("each layer must occur in exactly one physical pool")
        if any(pool.kind == "full" and pool.layout is None for pool in self.pools):
            raise ValueError("FULL pools require their explicit layout")

    @property
    def pool_bytes_per_block(self) -> int:
        return sum(pool.bytes_per_block for pool in self.pools)

    @property
    def bytes_per_block(self) -> int:
        # One persistent int64 scheduler block ID per global block for the
        # zeroer; reserved at planning time and allocated at zeroer setup.
        return self.pool_bytes_per_block + 8

    @property
    def window_bytes(self) -> int:
        # Different FULL groups can share a history pool, but their request
        # windows contain different layer values and must have separate storage.
        return sum(len(pool.shared_by) * self.max_requests * self.window_capacity
                   * pool.layout.bf16_bytes_per_token
                   for pool in self.pools if pool.layout is not None)

    @property
    def staging_bytes(self) -> int:
        # Per FULL layer: BF16 K/V staging rows keyed by native virtual slots,
        # plus one int64 owner tag per staged token. Sizing follows the
        # upstream PR; it is a working-set bound, not a correctness guarantee.
        total = 0
        for pool in self.pools:
            if pool.layout is None:
                continue
            layout = pool.layout
            rows = staging_rows(layout.block_size, self.sink, self.recent,
                                self.staging_tokens)
            per_layer = (rows * layout.block_size * layout.num_kv_heads
                         * (layout.head_size + layout.head_size_v) * 2
                         + rows * layout.block_size * 8)
            total += len(pool.shared_by) * per_layer
        return total

    @property
    def fixed_bytes(self) -> int:
        return self.window_bytes + self.staging_bytes + self.runtime_reserve_bytes

    def physical_bytes(self, num_blocks: int) -> int:
        positive(num_blocks, "num_blocks")
        return self.fixed_bytes + num_blocks * self.bytes_per_block

    def max_num_blocks(self, available_bytes: int) -> int:
        nonnegative(available_bytes, "available_bytes")
        count = (available_bytes - self.fixed_bytes) // self.bytes_per_block
        # Native BlockPool reserves block_id=0 as the null block. There must
        # be at least one additional allocatable block.
        if count < 2:
            raise ValueError("budget cannot hold windows, reserves, null block, and one usable block")
        return count
