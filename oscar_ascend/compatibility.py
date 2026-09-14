"""Record version families and verify the native symbols actually wrapped."""
from __future__ import annotations

import importlib.metadata
import inspect


def version_family(version: str) -> tuple[int, ...]:
    try:
        from packaging.version import Version
    except ImportError:
        # Host-only source checkout may have pip but not installed dependencies.
        # This is the same PEP440 parser, not a hand-written string comparison.
        from pip._vendor.packaging.version import Version
    return Version(version).release


def check_version_families(vllm_version: str, ascend_version: str) -> dict:
    native = version_family(vllm_version)
    ascend = version_family(ascend_version)
    if native[:2] != (0, 23) or ascend[:2] != (0, 23):
        raise RuntimeError(f"Unverified vLLM/Ascend release families: {vllm_version}, {ascend_version}")
    return {"vllm": vllm_version, "vllm_ascend": ascend_version,
            "vllm_release": native, "ascend_release": ascend,
            "status": "version_family_matches_only"}


def require_parameters(function, names: set[str], label: str) -> str:
    signature = inspect.signature(function)
    if not names <= set(signature.parameters):
        raise RuntimeError(f"Native interface changed: {label}{signature}; expected parameters {sorted(names)}")
    return str(signature)


def inspect_native_interfaces() -> dict:
    """Call only after the native platform initialized, before installing hooks.

    This verifies loadable Python seams. It does not assert device behavior or
    accept an unknown native commit merely because the version family matches.
    """
    from vllm.v1.core import kv_cache_utils
    from vllm_ascend.platform import NPUPlatform
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    report = check_version_families(importlib.metadata.version("vllm"),
                                    importlib.metadata.version("vllm-ascend"))
    seams = (
        (kv_cache_utils.get_kv_cache_config_from_groups, {"vllm_config", "kv_cache_groups", "available_memory"}, "get_kv_cache_config_from_groups"),
        (kv_cache_utils._max_memory_usage_bytes_from_groups, {"vllm_config", "kv_cache_groups"}, "_max_memory_usage_bytes_from_groups"),
        (NPUModelRunner._allocate_kv_cache_tensors, {"self", "kv_cache_config"}, "NPUModelRunner._allocate_kv_cache_tensors"),
        (NPUModelRunner._reshape_kv_cache_tensors, {"self", "kv_cache_config"}, "NPUModelRunner._reshape_kv_cache_tensors"),
        (NPUModelRunner.initialize_kv_cache, {"self", "kv_cache_config"}, "NPUModelRunner.initialize_kv_cache"),
        (NPUModelRunner._update_states, {"self", "scheduler_output"}, "NPUModelRunner._update_states"),
        (NPUModelRunner._prepare_inputs, {"self", "scheduler_output"}, "NPUModelRunner._prepare_inputs"),
    )
    report["interfaces"] = {label: require_parameters(function, parameters, label)
                            for function, parameters, label in seams}
    selector = inspect.getattr_static(NPUPlatform, "get_attn_backend_cls")
    if not isinstance(selector, classmethod):
        raise RuntimeError("NPUPlatform.get_attn_backend_cls no longer has classmethod binding")
    report["interfaces"]["get_attn_backend_cls"] = require_parameters(
        selector.__func__, {"cls", "selected_backend", "attn_selector_config"}, "get_attn_backend_cls")
    for spec, attributes in ((FullAttentionSpec, ("page_size_bytes", "max_memory_usage_bytes")),
                             (MambaSpec, ("page_size_bytes", "max_memory_usage_bytes"))):
        if not all(hasattr(spec, name) for name in attributes):
            raise RuntimeError(f"Cache spec interface changed: {spec.__name__}")
    report["module_paths"] = {"kv_cache_utils": inspect.getfile(kv_cache_utils),
                              "NPUPlatform": inspect.getfile(NPUPlatform),
                              "NPUModelRunner": inspect.getfile(NPUModelRunner)}
    report["status"] = "python_interfaces_present_device_unverified"
    return report
