"""Lightweight vLLM general-plugin entry point.

No platform, tensor, model-runner, or DeviceOperator import is allowed at module
scope. vLLM swallows entry-point import failures, but propagates callback errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
import json
import os
from pathlib import Path
from typing import Any


NATIVE_FULL_BACKEND = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
OSCAR_FULL_BACKEND = "oscar_ascend.backend.OscarAttentionBackend"


@dataclass(frozen=True)
class PluginConfig:
    sink: int = 64
    recent: int = 256
    group_size: int = 256
    runtime_reserve_bytes: int = 0
    staging_tokens: int = 8192
    library_path: str | None = None
    rotations_path: str | None = None
    k_clip: float = 1.0
    v_clip: float = 1.0

    def __post_init__(self):
        for name in ("sink", "recent", "runtime_reserve_bytes", "staging_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        for name in ("library_path", "rotations_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty path string")
        if not 0 <= self.k_clip <= 1 or not 0 <= self.v_clip <= 1:
            raise ValueError("clip ratios must be in [0, 1]")


def enabled(environ: Any = None) -> bool:
    value = (os.environ if environ is None else environ).get("OSCAR_ASCEND_ENABLED", "0")
    if value not in ("0", "1"):
        raise ValueError("OSCAR_ASCEND_ENABLED must be 0 or 1")
    return value == "1"


def load_config(environ: Any = None) -> PluginConfig:
    """Optional JSON config; no model or runtime capabilities can be overridden."""
    source = (os.environ if environ is None else environ).get("OSCAR_ASCEND_CONFIG")
    if source is None:
        return PluginConfig()
    data = json.loads(Path(source).read_text())
    if not isinstance(data, dict):
        raise ValueError("OSCAR_ASCEND_CONFIG must contain a JSON object")
    return PluginConfig(**data)


@dataclass
class ClassMethodPatch:
    owner: type
    name: str
    original: classmethod
    replacement: classmethod

    def undo(self) -> None:
        current = inspect.getattr_static(self.owner, self.name)
        if current is self.original:
            return
        if current is not self.replacement:
            raise RuntimeError(f"cannot undo {self.name}: another component replaced the hook")
        setattr(self.owner, self.name, self.original)


def install_backend_route(platform_cls: type) -> ClassMethodPatch:
    """Wrap only the native standard FULL backend result, preserving binding."""
    name = "get_attn_backend_cls"
    descriptor = inspect.getattr_static(platform_cls, name)
    if not isinstance(descriptor, classmethod):
        raise TypeError("native get_attn_backend_cls must remain a classmethod")
    installed = getattr(descriptor.__func__, "_oscar_route_patch", None)
    if installed is not None:
        return installed
    original = descriptor.__func__
    signature = inspect.signature(original)
    if not {"selected_backend", "attn_selector_config"} <= set(signature.parameters):
        raise RuntimeError("unverified native attention selector signature")

    @wraps(original)
    def route(cls, *args, **kwargs):
        selected = original(cls, *args, **kwargs)
        return OSCAR_FULL_BACKEND if selected == NATIVE_FULL_BACKEND else selected

    replacement = classmethod(route)
    handle = ClassMethodPatch(platform_cls, name, descriptor, replacement)
    route._oscar_route_patch = handle
    setattr(platform_cls, name, replacement)
    return handle


def register() -> None:
    """Install in each enabled process; all capability failures remain fatal.

    Registration is not a runtime acceptance signal. The external backend checks
    for the concrete window/attention runtime before model execution, and the
    deployment readiness gate separately checks the target acceptance matrix.
    """
    if not enabled():
        return
    config = load_config()

    # Entry module is fully imported before accessing the lazy platform singleton.
    from vllm.platforms import current_platform
    from vllm_ascend.platform import NPUPlatform
    from .cache_integration import install_cache_hooks
    from .runtime import install_runtime_hooks, reserve_bytes

    if not isinstance(current_platform, NPUPlatform):
        raise RuntimeError("OSCAR was enabled but the active platform is not Ascend NPU")
    existing = getattr(NPUPlatform, "_oscar_plugin_config", None)
    if existing is not None:
        if existing != config:
            raise RuntimeError("OSCAR was registered with a different configuration")
        return
    from .compatibility import inspect_native_interfaces
    compatibility_report = inspect_native_interfaces()
    route = install_backend_route(NPUPlatform)
    runtime_handle = None
    try:
        runtime_handle = install_runtime_hooks(config)
        cache_handle = install_cache_hooks(
            sink=config.sink, recent=config.recent, group_size=config.group_size,
            runtime_reserve_bytes=config.runtime_reserve_bytes,
            staging_tokens=config.staging_tokens,
            runtime_reserve_fn=lambda vllm_config, groups: reserve_bytes(vllm_config, groups, config),
        )
    except BaseException:
        if runtime_handle is not None:
            runtime_handle.undo()
        route.undo()
        raise
    NPUPlatform._oscar_plugin_config = config
    NPUPlatform._oscar_route_handle = route
    NPUPlatform._oscar_cache_handle = cache_handle
    NPUPlatform._oscar_runtime_handle = runtime_handle
    NPUPlatform._oscar_compatibility_report = compatibility_report


def unregister() -> None:
    """Restore entry points before any live engine uses OSCAR-owned cache views."""
    from vllm_ascend.platform import NPUPlatform

    route = getattr(NPUPlatform, "_oscar_route_handle", None)
    if route is None:
        return
    cache_handle = getattr(NPUPlatform, "_oscar_cache_handle", None)
    if cache_handle is None or not hasattr(cache_handle, "undo"):
        raise RuntimeError("cache hooks cannot yet be safely undone; restart the worker")
    cache_handle.undo()
    NPUPlatform._oscar_runtime_handle.undo()
    route.undo()
    for name in ("_oscar_plugin_config", "_oscar_route_handle", "_oscar_cache_handle",
                 "_oscar_runtime_handle", "_oscar_compatibility_report"):
        delattr(NPUPlatform, name)
