"""External FULL backend seam with explicit runtime readiness contracts.

The independently buildable AscendC primitives do not by themselves implement
request-window migration, MTP commit, or prefix-cache recovery. This bridge never
substitutes native BF16 attention or a CPU implementation for missing pieces.
Heavy imports occur only when vLLM resolves the backend class after registration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RuntimeNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeCapabilities:
    window_commit: bool = False
    prefill: bool = False
    chunked_prefill: bool = False
    decode: bool = False
    verify: bool = False
    draft: bool = False
    graph_capture: bool = False
    prefix_lifecycle: bool = False
    request_lifecycle: bool = False

    def require(self, *features: str) -> None:
        missing = [name for name in features if not getattr(self, name)]
        if missing:
            raise RuntimeNotReady("OSCAR runtime is incomplete: " + ", ".join(missing))


class AttentionRuntime(Protocol):
    """Implemented by the concrete NPU window/attention orchestration layer.

    prepare consumes native device seq_lens after async correction. It must
    derive computed=seq_len-q_len on NPU and commit only materialized pending KV,
    with padding excluded. Each layer has a distinct mutable window state.
    forward must execute the INT2 AscendC route with fixed graph addresses.
    """
    capabilities: RuntimeCapabilities

    def prepare(self, common: Any, layer_names: tuple[str, ...], *,
                for_capture: bool) -> Any: ...

    def forward(self, *, layer: Any, query: Any, key: Any, value: Any,
                cache: Any, metadata: Any, output: Any, scale: float) -> Any: ...

    def check_graph_addresses(self) -> None: ...


def require_runtime(config: Any) -> AttentionRuntime:
    runtime = getattr(config, "_oscar_attention_runtime", None)
    if runtime is None:
        raise RuntimeNotReady(
            "OSCAR FULL backend selected, but the NPU request-window runtime is "
            "not installed. Independent AscendC operator probes remain available; "
            "service inference requires the runner hooks that provide request "
            "identity, window commit/rollback, drafter steps and prefix-window "
            "recovery. No native fallback was used."
        )
    if not isinstance(getattr(runtime, "capabilities", None), RuntimeCapabilities):
        raise RuntimeNotReady("OSCAR runtime lacks a typed capability contract")
    runtime.capabilities.require("window_commit", "request_lifecycle")
    return runtime


@dataclass
class OscarMetadata:
    prepared: Any
    runtime: AttentionRuntime
    num_actual_tokens: int


def _make_backend_class():
    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend, AscendAttentionBackendImpl,
    )

    class OscarMetadataBuilder(AttentionMetadataBuilder):
        reorder_batch_threshold = None

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.config = vllm_config
            self.layer_names = tuple(layer_names)

        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            runtime = require_runtime(vllm_config)
            runtime.capabilities.require("graph_capture")
            # Returned only for a concrete runtime advertising implemented support;
            # deployment graph capture/replay acceptance is a separate check.
            return AttentionCGSupport.UNIFORM_BATCH

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            runtime = require_runtime(self.config)
            runtime.capabilities.require("prefill", "chunked_prefill", "decode", "verify")
            if self.config.cache_config.enable_prefix_caching or common_prefix_len:
                runtime.capabilities.require("prefix_lifecycle")
            prepared = runtime.prepare(common_attn_metadata, self.layer_names,
                                       for_capture=False)
            return OscarMetadata(prepared, runtime, common_attn_metadata.num_actual_tokens)

        def build_for_cudagraph_capture(self, common_attn_metadata):
            runtime = require_runtime(self.config)
            runtime.capabilities.require("graph_capture")
            prepared = runtime.prepare(common_attn_metadata, self.layer_names,
                                       for_capture=True)
            return OscarMetadata(prepared, runtime, common_attn_metadata.num_actual_tokens)

    class OscarAttentionImpl(AscendAttentionBackendImpl):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.sliding_window is not None or self.alibi_slopes is not None:
                raise RuntimeNotReady("OSCAR supports standard causal FULL attention only")
            if self.kv_sharing_target_layer_name is not None or self.sinks is not None:
                raise RuntimeNotReady("OSCAR learned sinks and cross-layer KV sharing are unverified")
            require_runtime(get_current_vllm_config())

        @staticmethod
        def update_graph_params(update_stream, forward_context, num_tokens,
                                vllm_config, speculative_config=None,
                                num_dcp_pcp_tokens=None, draft_attn_metadatas=None):
            # OSCAR custom ops read mutable lengths/pages from persistent tensor
            # addresses, already copied on the model stream by prepare(). There
            # are no native FIA task-group handles or scalar-length parameters
            # to update on update_stream. Keeping the address contract is enough
            # for this specific custom-op path; actual graph replay is untested.
            require_runtime(vllm_config).check_graph_addresses()

        def do_kv_cache_update(self, *args, **kwargs):
            raise RuntimeNotReady("OSCAR KV commit/store must run through its fused FULL forward")

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            if output is None:
                raise ValueError("OSCAR requires the native output buffer")
            if output_scale is not None or output_block_scale is not None:
                raise RuntimeNotReady("OSCAR fused output quantization is unimplemented")
            if attn_metadata is None:
                # Native memory-profile dummy execution has no KV metadata.
                return output.zero_()
            if not isinstance(attn_metadata, OscarMetadata):
                raise RuntimeNotReady("OSCAR received another backend's metadata")
            from .cache_integration import FullCacheView
            if not isinstance(kv_cache, FullCacheView):
                raise RuntimeNotReady("OSCAR requires the independently allocated packed cache")
            if query.device.type != "npu":
                raise ValueError("OSCAR has no CPU KV or attention fallback")
            return attn_metadata.runtime.forward(
                layer=layer, query=query, key=key, value=value, cache=kv_cache,
                metadata=attn_metadata.prepared, output=output, scale=self.scale,
            )

    class OscarAttentionBackend(AscendAttentionBackend):
        forward_includes_kv_cache_update = True

        @staticmethod
        def get_impl_cls():
            return OscarAttentionImpl

        @staticmethod
        def get_builder_cls():
            return OscarMetadataBuilder

        @staticmethod
        def swap_blocks(*args, **kwargs):
            # v1 shares cached prefixes as whole read-only blocks and never
            # routes byte swaps through attention backends; reaching this
            # branch means an unverified connector/offload lifecycle.
            raise RuntimeNotReady("OSCAR compressed block swap lifecycle is unimplemented")

        @staticmethod
        def copy_blocks(*args, **kwargs):
            raise RuntimeNotReady("OSCAR compressed block copy lifecycle is unimplemented")

    # Stable import identities for logging, grouping and worker introspection.
    for name, cls in (("OscarAttentionBackend", OscarAttentionBackend),
                      ("OscarAttentionImpl", OscarAttentionImpl),
                      ("OscarMetadataBuilder", OscarMetadataBuilder)):
        cls.__qualname__ = name
        globals()[name] = cls
    return OscarAttentionBackend


def __getattr__(name: str):
    if name == "OscarAttentionBackend":
        backend = _make_backend_class()
        globals()[name] = backend
        return backend
    raise AttributeError(name)
