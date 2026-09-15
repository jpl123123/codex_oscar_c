"""Native worker extension for two-pass NPU calibration of real FULL Q/K/V.

Every public method has a unique RPC name. The frontend sends strings and JSON
control values only; no function serialization or raw K/V transfer is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CALIBRATION_OPS = (
    "calib_q_moments_out", "calib_q_cov_out", "calib_sst_moments_out",
    "calib_sst_cov_out", "calib_eigh_rhp_out", "calib_fingerprint_out",
)
CALIBRATION_TOKEN_TILE = 256
_MISSING = object()


@dataclass
class SampleTrace:
    budget: int
    count: int = 0
    calls: list = field(default_factory=list)

    def select(self, available: int, *, stage: str, query_heads: int,
               kv_heads: int, head_dim: int) -> tuple[int, int]:
        if available < 0:
            raise ValueError("negative calibration token count")
        take = min(available, self.budget - self.count)
        offset = self.count
        if take:
            self.calls.append((stage, available, take, query_heads, kv_heads, head_dim))
            self.count += take
        return offset, take

    def summary(self) -> dict:
        content = json.dumps(self.calls, separators=(",", ":")).encode()
        return {"tokens": self.count, "calls": len(self.calls),
                "trace_sha256": hashlib.sha256(content).hexdigest()}


def verify_passes(first: dict, second: dict) -> None:
    if first != second:
        raise RuntimeError("calibration passes captured different token counts or forward traces")
    if not first or any(entry["tokens"] <= 0 for entry in first.values()):
        raise RuntimeError("calibration is missing real samples for one or more FULL layers")


class CalibrationWorkerExtension:
    def oscar_calibration_agree(self, local_error: str | None, label: str) -> None:
        """All TP ranks either enter the next numerical collective or fail."""
        import torch
        state = self._oscar_calibration
        failed = torch.tensor([1.0 if local_error else 0.0], device=self.device)
        if state["tp"].all_reduce(failed).item() != 0:
            raise RuntimeError(f"{label}: {local_error or 'another TP rank failed validation'}")

    def oscar_calibration_begin(self, request: dict, shard_directory: str) -> dict:
        if getattr(self, "_oscar_calibration", None) is not None:
            raise RuntimeError("calibration collection is already active")
        if os.environ.get("OSCAR_ASCEND_ENABLED", "0") != "0":
            raise RuntimeError("calibration must use the native cache and attention backend")
        import torch
        from vllm.distributed import get_tp_group
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.config import get_layers_from_vllm_config
        from .ops import load_library

        runner, config = self.model_runner, self.vllm_config
        tp = get_tp_group()
        if tp.world_size != 4 or config.parallel_config.tensor_parallel_size != 4:
            raise RuntimeError("calibration requires the native TP=4 engine")
        total_kv_heads = config.model_config.get_total_num_kv_heads()
        if total_kv_heads < 4 or total_kv_heads % 4:
            raise RuntimeError("replicated or uneven TP KV-head calibration is not implemented")
        operators = load_library(request["library_path"])
        missing = [name for name in CALIBRATION_OPS if not hasattr(operators, name)]
        if missing:
            raise RuntimeError("calibration operator ABI is incomplete: " + ", ".join(missing))
        layers = get_layers_from_vllm_config(config, AttentionLayerBase)
        full = {name: layer for name, layer in layers.items()
                if type(layer.get_kv_cache_spec(config)).__name__ == "FullAttentionSpec"}
        if not full:
            raise RuntimeError("native engine has no discoverable FULL attention layers")
        draft_layers = set(getattr(getattr(runner, "drafter", None), "attn_layer_names", ()))
        if not draft_layers or not draft_layers <= set(full):
            raise RuntimeError("native MTP FULL layers are not discoverable for calibration")
        device = self.device
        state = {"phase": "q", "request": request, "operators": operators, "layers": {},
                 "hooks": [], "first": None, "first_fingerprints": None,
                 "shard_directory": str(Path(shard_directory).resolve()),
                 "tp": tp, "total_kv_heads": total_kv_heads, "draft_layers": draft_layers}
        state["weights_workspaces"] = {
            hk: torch.empty((CALIBRATION_TOKEN_TILE, hk), dtype=torch.float32, device=device)
            for hk in {layer.num_kv_heads for layer in full.values()}
        }
        state["fingerprint_partial"] = torch.zeros((6, 32), dtype=torch.int64, device=device)
        for name, layer in sorted(full.items()):
            if type(layer.impl).__module__.startswith("oscar_ascend"):
                raise RuntimeError("OSCAR inference backend was active during native calibration")
            hk, d, hq = layer.num_kv_heads, layer.head_size, layer.num_heads
            if hk * 4 != total_kv_heads or hq % hk:
                raise RuntimeError("FULL layer KV-head sharding differs from the verified TP contract")
            if d not in (64, 128, 256):
                raise RuntimeError(f"unsupported calibration head dimension {d}")
            zeros = lambda shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype, device=device)
            entry = dict(hq=hq, hk=hk, d=d, layer=layer,
                         trace=SampleTrace(request["token_budget"]),
                         moments=zeros((hk, d, d)), counts=zeros(hk, torch.int64),
                         per_head=zeros((hk, d, d)), global_q=zeros((d, d)),
                         weighted=zeros((hk, d, d)), weight_sum=zeros(hk),
                         global_v=zeros((d, d)), fingerprint=zeros(6, torch.int64))
            state["layers"][name] = entry
        self._oscar_calibration = state
        try:
            for name, entry in state["layers"].items():
                impl = entry["layer"].impl
                original = impl.forward
                prior_attribute = impl.__dict__.get("forward", _MISSING)

                def make_hook(layer_name, native_forward):
                    @wraps(native_forward)
                    def hook(layer, query, key, value, kv_cache, attn_metadata,
                             output=None, output_scale=None, output_block_scale=None):
                        self.oscar_calibration_collect(layer_name, query, key, value, attn_metadata)
                        return native_forward(layer, query, key, value, kv_cache, attn_metadata,
                                              output, output_scale, output_block_scale)
                    return hook

                hook = make_hook(name, original)
                impl.forward = hook
                state["hooks"].append((impl, prior_attribute, hook))
        except BaseException:
            self.oscar_calibration_abort()
            raise
        return {"rank": config.parallel_config.rank, "pid": os.getpid(),
                "full_layers": sorted(full), "mtp_layers": sorted(draft_layers),
                "total_kv_heads": total_kv_heads, "tp_world_size": tp.world_size,
                "scope": "native real prefill FULL post-RoPE Q/K/V; no GDN hooks"}

    def oscar_calibration_collect(self, name: str, query: Any, key: Any,
                                  value: Any, metadata: Any) -> None:
        state = getattr(self, "_oscar_calibration", None)
        if state is None or state["phase"] not in ("q", "sst") or metadata is None:
            return
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        if getattr(context, "capturing", False):
            return
        stage = getattr(getattr(metadata, "attn_state", None), "name", "")
        if name in state["draft_layers"]:
            # Native MTP step 1+ marks SpecDecoding; collect only its real first
            # prefill pass, whose Q/K/V are not rejected speculative padding.
            if stage not in ("PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill"):
                return
        else:
            batch = self.model_runner.input_batch
            if batch.num_reqs != 1:
                raise RuntimeError("deterministic calibration requires one active prompt at a time")
            if batch.num_computed_tokens_cpu[0] >= batch.num_prompt_tokens[0]:
                return
        if getattr(metadata, "num_decodes", 0) and getattr(metadata, "num_prefills", 0):
            raise RuntimeError("mixed prefill/decode calibration is not a reproducible sample selection")
        available = int(metadata.num_actual_tokens)
        entry = state["layers"][name]
        offset, take = entry["trace"].select(
            available, stage=stage, query_heads=entry["hq"], kv_heads=entry["hk"],
            head_dim=entry["d"])
        if not take:
            return
        if key is None or value is None:
            raise RuntimeError("real calibration FULL forward did not provide K/V")
        for tensor, heads in ((query, entry["hq"]), (key, entry["hk"]), (value, entry["hk"])):
            if (tuple(tensor.shape[1:]) != (heads, entry["d"])
                    or tensor.device.type != "npu" or str(tensor.dtype) != "torch.bfloat16"
                    or tensor.shape[0] < take):
                raise RuntimeError("calibration Q/K/V shape, dtype, or NPU placement mismatch")
        # Prefix tensor views change no K/V values and preserve native positive
        # strides. AscendC handles strided raw V directly; no torch.contiguous.
        ops = state["operators"]
        # Bound each device-statistics launch. This tiles one raw forward, not
        # the mathematical normalization: all tiles contribute to one global
        # sum/count per head and pass 2 uses the frozen full-pass Q covariance.
        for begin in range(0, take, CALIBRATION_TOKEN_TILE):
            end = min(begin + CALIBRATION_TOKEN_TILE, take)
            q, k, v = query[begin:end], key[begin:end], value[begin:end]
            ops.calib_fingerprint_out(q, k, v, entry["fingerprint"],
                                       state["fingerprint_partial"], offset + begin)
            if state["phase"] == "q":
                ops.calib_q_moments_out(q, entry["moments"], entry["counts"])
            else:
                workspace = state["weights_workspaces"][entry["hk"]][:end - begin]
                ops.calib_sst_moments_out(k, v, entry["per_head"], entry["weighted"],
                                          entry["weight_sum"], workspace)

    def oscar_calibration_status(self) -> dict:
        state = self._oscar_calibration
        return {"rank": self.vllm_config.parallel_config.rank, "phase": state["phase"],
                "layers": {name: entry["trace"].summary() for name, entry in state["layers"].items()}}

    def oscar_calibration_next_pass(self) -> dict:
        import torch
        state = self._oscar_calibration
        if state["phase"] != "q":
            raise RuntimeError("calibration is not in the first Q-moment pass")
        state["phase"] = "between"
        torch.npu.synchronize()
        first = {name: entry["trace"].summary() for name, entry in state["layers"].items()}
        local_error = None
        try:
            verify_passes(first, first)
            if any(entry["tokens"] != state["request"]["token_budget"] for entry in first.values()):
                raise RuntimeError("calibration corpus did not reach the declared token budget on every FULL/MTP layer")
            for name, entry in state["layers"].items():
                counts = entry["counts"].cpu().tolist()
                expected = entry["trace"].count * (entry["hq"] // entry["hk"])
                if counts != [expected] * entry["hk"]:
                    raise RuntimeError(f"NPU Q-moment counts disagree with selected token metadata: {name}")
        except Exception as exc:
            local_error = str(exc)
        self.oscar_calibration_agree(local_error, "first-pass coverage")
        state["first"] = first
        state["first_fingerprints"] = {}
        for name, entry in sorted(state["layers"].items()):
            state["first_fingerprints"][name] = entry["fingerprint"].cpu().tolist()
            state["operators"].calib_q_cov_out(
                entry["moments"], entry["counts"], entry["per_head"], entry["global_q"],
                state["total_kv_heads"])
            # Each local head already contributes 1/global_Hkv. Native HCCL
            # adds distinct TP head contributions without CPU statistics math.
            entry["global_q"].copy_(state["tp"].all_reduce(entry["global_q"]))
            entry["fingerprint"].zero_()
            entry["trace"] = SampleTrace(state["request"]["token_budget"])
        torch.npu.synchronize()
        state["phase"] = "sst"
        return {"rank": self.vllm_config.parallel_config.rank, "first_pass": first,
                "fingerprints": state["first_fingerprints"]}

    def oscar_calibration_finish(self) -> dict:
        import torch
        from .runtime import check_rotation_on_npu
        state = self._oscar_calibration
        if state["phase"] != "sst":
            raise RuntimeError("calibration is not in the fixed-covariance SST pass")
        state["phase"] = "solve"
        torch.npu.synchronize()
        second = {name: entry["trace"].summary() for name, entry in state["layers"].items()}
        local_error = None
        pass_deltas = {}
        try:
            verify_passes(state["first"], second)
            for name, entry in state["layers"].items():
                # Sample identity is pinned by the token traces above. The
                # fingerprint sums remain a coarse catastrophe bound only:
                # on real 910B hardware the two replay passes legitimately
                # differ by up to ~0.4 in relative bit-sum (observed 0.19-
                # 0.36 uniformly across ranks on one layer): near-zero BF16
                # elements flip sign across passes, which moves raw bit
                # patterns by ~0x8000 per element, and pass-1 kernel warmup
                # versus pass-2 steady state amplifies this. Values beyond
                # 0.9 would indicate a genuinely different dataset rather
                # than replay variance; per-layer deltas are recorded either
                # way for artifact audit.
                current = entry["fingerprint"].cpu().tolist()
                recorded = state["first_fingerprints"][name]
                worst = max(abs(a - b) / max(abs(a), abs(b), 1)
                            for a, b in zip(current[:3], recorded[:3]))
                pass_deltas[name] = worst
                if worst > 0.9:
                    raise RuntimeError(
                        f"calibration passes replayed different data "
                        f"(relative fingerprint-sum delta {worst:.3e}): {name}")
        except Exception as exc:
            local_error = str(exc)
        self.oscar_calibration_agree(local_error, "two-pass sample identity")
        layers, diagnostics = {}, {}
        for name, entry in sorted(state["layers"].items()):
            ops = state["operators"]
            ops.calib_sst_cov_out(entry["weighted"], entry["weight_sum"], entry["global_v"],
                                   state["total_kv_heads"])
            entry["global_v"].copy_(state["tp"].all_reduce(entry["global_v"]))
            layers[name], diagnostics[name] = {}, {}
            d = entry["d"]
            for side, covariance in (("k", entry["global_q"]), ("v", entry["global_v"])):
                rotation = torch.empty((d, d), dtype=torch.float32, device=self.device)
                eigenvalues = torch.empty(d, dtype=torch.float32, device=self.device)
                eigenvectors = torch.empty_like(rotation)
                workspace = torch.empty((2, d, d), dtype=torch.float32, device=self.device)
                diagnostic = torch.empty(8, dtype=torch.float32, device=self.device)
                ops.calib_eigh_rhp_out(covariance, rotation, eigenvalues, eigenvectors,
                                       workspace, diagnostic,
                                       state["request"].get("max_sweeps", 32),
                                       state["request"].get("solver_tolerance", 1e-6))
                torch.npu.synchronize()
                values = diagnostic.cpu().tolist()
                valid = (values[0] == 1 and values[7] == 0 and
                         0 <= values[4] <= 0.02 and 0 <= values[5] <= 0.02 and 0 <= values[6] <= 0.02)
                failures = torch.tensor([0.0 if valid else 1.0], device=self.device)
                if state["tp"].all_reduce(failures).item() != 0:
                    raise RuntimeError(f"a TP rank failed NPU eigensolver validation at {name}/{side}: local={values}")
                # A shared paper layer rotation has one canonical eigenbasis.
                # Avoid independent rank-local sign/basis choices even though
                # the covariance was globally reduced on every rank.
                state["tp"].broadcast(rotation, src=0)
                state["tp"].broadcast(eigenvalues, src=0)
                roundtrip, local_error = None, None
                try:
                    roundtrip = check_rotation_on_npu(rotation, ops)
                except Exception as exc:
                    local_error = str(exc)
                self.oscar_calibration_agree(local_error, f"broadcast rotation roundtrip {name}/{side}")
                # Export only a small completed rotation matrix, never raw Q/K/V.
                layers[name][side] = rotation.cpu()
                diagnostics[name][side] = {"solver": values, "rotation_roundtrip_max_error": roundtrip,
                                          "rotation_broadcast_source_tp_rank": 0,
                                          "eigenvalues": eigenvalues.cpu().tolist()}
        rank = self.vllm_config.parallel_config.rank
        destination = Path(state["shard_directory"]) / f"rank_{rank}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"rank": rank, "layers": layers, "diagnostics": diagnostics,
                    "traces": second, "fingerprints": state["first_fingerprints"],
                    "pass_fingerprint_max_relative_delta": pass_deltas,
                    "validated_on_npu": True}, destination)
        result = {"rank": rank, "pid": os.getpid(), "shard": str(destination),
                  "traces": second, "validated_on_npu": True,
                  "scope": "real NPU statistics, two-pass fingerprints, eigensolver and rotation validation"}
        self.oscar_calibration_abort()
        return result

    def oscar_calibration_abort(self) -> dict:
        state = getattr(self, "_oscar_calibration", None)
        if state is None:
            return {"pid": os.getpid(), "status": "inactive"}
        state["phase"] = "stopped"
        for impl, _, hook in state["hooks"]:
            if impl.forward is not hook:
                raise RuntimeError("another component replaced a calibration hook; refusing to overwrite it")
        for impl, original, _ in reversed(state["hooks"]):
            if original is _MISSING:
                del impl.forward
            else:
                impl.forward = original
        import torch
        torch.npu.synchronize()
        self._oscar_calibration = None
        return {"pid": os.getpid(), "status": "hooks_removed", "npu_worker_exit": "not_yet_shutdown"}
