"""Small explicit NPU numerical probes; CPU reads here are test oracles only.

This module is never imported by the production attention path. Passing it does
not establish whole-model TP, MTP, graph, accuracy, or performance acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import traceback


def decoded_cache(torch, cache, block_table, length, dim):
    """Decode a small copied test output in the PR's FP32 arithmetic."""
    raw = cache.cpu()
    bs, heads = raw.shape[1:3]
    region = dim // 4 + 4
    keys = torch.empty((length, heads, dim), dtype=torch.float32)
    values = torch.empty_like(keys)
    for token in range(length):
        for head in range(heads):
            slot = bytes(raw[block_table[token // bs], token % bs, head].tolist())
            for side, output in enumerate((keys, values)):
                offset = side * region
                scale, low = struct.unpack("<ee", slot[offset + dim // 4:offset + region])
                for j in range(dim):
                    code = (slot[offset + j // 4] >> (2 * (j % 4))) & 3
                    output[token, head, j] = code * scale + low
    return keys, values


def run_probes(ns, torch, device, *, capture=False):
    checks = []
    for dim in (64, 128, 256):
        bs, heads, blocks = 16, 1, 9
        width = 2 * (dim // 4 + 4)
        identity = torch.eye(dim, dtype=torch.float32, device=device)
        # Exact golden pattern exercises all 2-bit codes and metadata endian.
        k = torch.tensor([-1, 0, 1, 2], dtype=torch.bfloat16).repeat(2, heads, dim // 4).to(device)
        v = torch.tensor([3, 2, 1, 0], dtype=torch.bfloat16).repeat(2, heads, dim // 4).to(device)
        cache = torch.full((blocks, bs, heads, width), 165, dtype=torch.uint8, device=device)
        slots = torch.tensor([5 * bs + 3, -1], dtype=torch.int64, device=device)
        ns.store_int2(k, v, identity, identity, slots, cache, 0.0, 0.0)
        torch.npu.synchronize()
        expected = bytes([0xE4] * (dim // 4)) + struct.pack("<ee", 1, -1)
        expected += bytes([0x1B] * (dim // 4)) + struct.pack("<ee", 1, 0)
        try:
            assert bytes(cache[5, 3, 0].cpu().tolist()) == expected, f"D={dim}: INT2 golden bytes differ"
            untouched = cache.cpu()
            untouched[5, 3, 0].fill_(165)
            assert bool((untouched == 165).all()), "store overwrote an invalid/neighbor slot"
            checks.append({"name": "pack_bytes_and_negative_slot", "head_dim": dim, "status": "passed"})
        except AssertionError as exc:
            checks.append({"name": "pack_bytes_and_negative_slot", "head_dim": dim,
                           "status": "failed", "error": str(exc)})

        generator = torch.Generator().manual_seed(9014 + dim)
        lengths, tables, query_heads = [23, 40], [[5, 1, 0], [7, 3, 2]], 6
        total = sum(lengths)
        raw_k = (torch.randn(total, heads, dim, generator=generator) * 0.2).to(torch.bfloat16).to(device)
        raw_v = (torch.randn(total, heads, dim, generator=generator) * 0.2).to(torch.bfloat16).to(device)
        mapping = [table[pos // bs] * bs + pos % bs for length, table in zip(lengths, tables) for pos in range(length)]
        cache.zero_()
        ns.store_int2(raw_k, raw_v, identity, identity,
                      torch.tensor(mapping, dtype=torch.int64, device=device), cache, 0.0, 0.0)
        torch.npu.synchronize()
        decoded = [decoded_cache(torch, cache, table, length, dim) for length, table in zip(lengths, tables)]
        bt = torch.tensor(tables, dtype=torch.int32, device=device)
        hs = torch.zeros(2, dtype=torch.int32, device=device)
        he = torch.tensor(lengths, dtype=torch.int32, device=device)
        for q_len in (1, 2, 3, 4):
            n, splits = 2 * q_len, 3
            q = (torch.randn(n, query_heads, dim, generator=generator) * 0.2).to(torch.bfloat16).to(device)
            qsl = torch.tensor([0, q_len, 2 * q_len], dtype=torch.int32, device=device)
            positions = [pos for length in lengths for pos in range(length - q_len, length)]
            qpos = torch.tensor(positions, dtype=torch.int32, device=device)
            out = torch.empty((n, query_heads, dim), dtype=torch.float32, device=device)
            lse = torch.empty((n, query_heads), dtype=torch.float32, device=device)
            workspace = torch.empty(ns.workspace_size(n, query_heads, dim, splits), dtype=torch.uint8, device=device)
            scale = dim ** -0.5

            def call():
                ns.history_attention_out(q, cache, bt, qsl, hs, he, qpos, out, lse,
                                         workspace, scale, splits, q_len)

            call()
            torch.npu.synchronize()
            reference = torch.empty((n, query_heads, dim), dtype=torch.float32)
            ref_lse = torch.empty((n, query_heads), dtype=torch.float32)
            q_cpu = q.cpu().float()
            for req, (key, value) in enumerate(decoded):
                for row in range(q_len):
                    token = req * q_len + row
                    visible = positions[token] + 1
                    logits = q_cpu[token] @ key[:visible, 0].T * scale
                    reference[token] = torch.softmax(logits, dim=-1) @ value[:visible, 0]
                    ref_lse[token] = torch.logsumexp(logits, dim=-1)
            actual, actual_lse = out.cpu(), lse.cpu()
            per_token = (actual - reference).abs().amax(dim=(1, 2))
            check = {"name": "history_attention", "head_dim": dim, "q_len": q_len,
                     "splits": splits,
                     "max_abs_error": float((actual - reference).abs().max()),
                     "per_token_max_abs_error": [round(v, 5) for v in per_token.tolist()],
                     "lse_max_abs_error": float((actual_lse - ref_lse).abs().max()),
                     "workspace_bytes": workspace.numel()}
            try:
                torch.testing.assert_close(actual, reference, atol=5e-3, rtol=5e-3)
                torch.testing.assert_close(actual_lse, ref_lse, atol=5e-3, rtol=5e-3)
                check["status"] = "passed"
            except AssertionError as exc:
                # Keep going: one run must surface every failing operator with
                # numbers, not stop at the first mismatch.
                check["status"] = "failed"
                check["error"] = str(exc).split("\n")[0:4]
                worst = int((actual - reference).abs().amax(dim=(1, 2)).argmax())
                row, head = divmod(
                    int((actual - reference)[worst].abs().amax(dim=1).argmax()),
                    actual.shape[2])
                check["worst_token"] = worst
                check["worst_sample"] = {
                    "position": positions[worst],
                    "actual": [round(v, 5) for v in actual[worst, head, max(0, row - 2):row + 3].tolist()],
                    "reference": [round(v, 5) for v in reference[worst, head, max(0, row - 2):row + 3].tolist()],
                    "actual_lse": [round(v, 5) for v in actual_lse[worst, :4].tolist()],
                    "reference_lse": [round(v, 5) for v in ref_lse[worst, :4].tolist()],
                }
            if capture and dim == 256 and q_len == 4:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    call()
                torch.npu.synchronize()
                expected_replay = out.clone()
                out.fill_(float("nan"))
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(out, expected_replay, atol=5e-3, rtol=5e-3)
                check["primitive_graph_capture_replay"] = "passed"
                # Same addresses, changed contents: a captured Python length/value
                # must not mask device-side updates during replay.
                hs.fill_(0)
                he.zero_()
                graph.replay()
                torch.npu.synchronize()
                assert bool(torch.isneginf(lse).all().cpu()), "empty-history graph metadata was frozen"
                assert bool((out == 0).all().cpu()), "empty history must yield zero output"
                he.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            checks.append(check)
    return checks


def run_eigensolver_probe(ns, torch, device):
    """Solve matrices with known spectra and compare against torch.linalg.eigh.

    Runs in seconds on one rank; used to bisect the AscendC Jacobi solver
    without paying for a full model calibration.
    """
    checks = []
    for d in (8, 64, 128):
        generator = torch.Generator().manual_seed(4400 + d)
        base = torch.randn(d, d, generator=generator)
        matrix = (base + base.T) / 2 + d * torch.eye(d)
        matrix = matrix.to(device=device, dtype=torch.float32)
        rotation = torch.empty(d, d, dtype=torch.float32, device=device)
        eigenvalues = torch.empty(d, dtype=torch.float32, device=device)
        vectors = torch.empty(d, d, dtype=torch.float32, device=device)
        workspace = torch.empty(2, d, d, dtype=torch.float32, device=device)
        diagnostic = torch.empty(8, dtype=torch.float32, device=device)
        try:
            ns.calib_eigh_rhp_out(matrix, rotation, eigenvalues, vectors,
                                  workspace, diagnostic, 32, 1e-6)
            torch.npu.synchronize()
        except Exception as exc:
            checks.append({"name": "eigensolver_known_spectrum", "dim": d,
                           "status": "failed", "error": repr(exc)})
            continue
        values = diagnostic.cpu().tolist()
        reference = torch.linalg.eigvalsh(matrix.cpu())
        actual = eigenvalues.cpu()
        eigen_error = float((actual.sort().values - reference.sort().values).abs().max())
        orthogonality = float((rotation.cpu().T @ rotation.cpu()
                               - torch.eye(d)).abs().max())
        # reconstruct A @ v - lambda v through the unsorted vectors
        vectors_cpu = vectors.cpu()
        residual = float((matrix.cpu() @ vectors_cpu
                          - vectors_cpu * eigenvalues.cpu().unsqueeze(0)).abs().max())
        checks.append({"name": "eigensolver_known_spectrum", "dim": d,
                       "status": "failed" if (values[0] != 1 or values[7] != 0
                                              or eigen_error > 1e-2
                                              or orthogonality > 1e-3) else "passed",
                       "diagnostic": values, "eigenvalue_max_abs_error": eigen_error,
                       "rotation_orthogonality": orthogonality,
                       "eigenpair_residual": residual})
    return checks


def run_prefix_probes(ns, torch, device):
    """Bounded-recovery dequantization and prefix staging/restore probes."""
    checks = []
    for dim in (64, 128, 256):
        bs, heads, blocks = 16, 1, 9
        width = 2 * (dim // 4 + 4)
        identity = torch.eye(dim, dtype=torch.float32, device=device)
        k = torch.tensor([-1, 0, 1, 2], dtype=torch.bfloat16).repeat(4, heads, dim // 4).to(device)
        v = torch.tensor([3, 2, 1, 0], dtype=torch.bfloat16).repeat(4, heads, dim // 4).to(device)
        cache = torch.zeros((blocks, bs, heads, width), dtype=torch.uint8, device=device)
        slots = torch.tensor([0, bs + 1, 2 * bs + 2, 8 * bs + 15], dtype=torch.int64, device=device)
        ns.store_int2(k, v, identity, identity, slots, cache, 0.0, 0.0)
        k_out = torch.empty((4, heads, dim), dtype=torch.bfloat16, device=device)
        v_out = torch.empty_like(k_out)
        ns.dequant_history_out(cache, slots, identity, identity, k_out, v_out)
        torch.npu.synchronize()
        # With clip=0 the affine code recovers code*scale+min exactly for the
        # chosen golden rows: [-1,0,1,2] with scale=1,min=-1 and [3,2,1,0] with
        # scale=1,min=0.
        torch.testing.assert_close(k_out, k, atol=0, rtol=0)
        torch.testing.assert_close(v_out, v, atol=0, rtol=0)
        checks.append({"name": "dequant_roundtrip_golden", "head_dim": dim, "status": "passed"})

    # Staging + prefix-hit restore: staged rows restore exactly, evicted rows
    # fall back to bounded INT2 recovery and are counted.
    dim, bs, heads = 256, 16, 1
    width = 2 * (dim // 4 + 4)
    sink, recent = 4, 8
    identity = torch.eye(dim, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(77)
    tokens = 16
    k = (torch.randn(tokens, heads, dim, generator=generator) * 0.2).to(torch.bfloat16).to(device)
    v = (torch.randn(tokens, heads, dim, generator=generator) * 0.2).to(torch.bfloat16).to(device)
    cache = torch.zeros((4, bs, heads, width), dtype=torch.uint8, device=device)
    slots = torch.arange(tokens, dtype=torch.int64, device=device)
    ns.store_int2(k, v, identity, identity, slots, cache, 0.0, 0.0)
    staging_k = torch.zeros((2, bs, heads, dim), dtype=torch.bfloat16, device=device)
    staging_v = torch.zeros_like(staging_k)
    owner = torch.full((2, bs), -1, dtype=torch.int64, device=device)
    ns.stage_window_out(k, v,
                        torch.tensor([tokens], dtype=torch.int32, device=device),
                        torch.tensor([0, tokens], dtype=torch.int32, device=device),
                        slots, staging_k, staging_v, owner, sink, recent)
    torch.npu.synchronize()
    staged = int((owner != -1).sum().item())
    assert staged == sink + recent, f"staging kept {staged} rows, expected {sink + recent}"
    cap = sink + recent + 4
    window_k = torch.zeros((1, cap, heads, dim), dtype=torch.bfloat16, device=device)
    window_v = torch.zeros_like(window_k)
    positions = torch.full((1, cap), -1, dtype=torch.int32, device=device)
    state = torch.zeros((1, 4), dtype=torch.int64, device=device)
    lossy = torch.zeros(1, dtype=torch.int32, device=device)
    committed = tokens
    ns.prefix_restore_out(
        torch.tensor([committed + 1], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([5], dtype=torch.int64, device=device),
        torch.zeros((1, 4), dtype=torch.int32, device=device),
        positions, window_k, window_v, state,
        staging_k, staging_v, owner, cache, identity, identity, lossy,
        sink, recent, 4, bs)
    torch.npu.synchronize()
    assert state[0].tolist() == [5, committed, 0, 0], f"restored state {state.tolist()}"
    assert int(lossy.sum().item()) == 0, f"staged restore must be exact, lossy={lossy.sum().item()}"
    torch.testing.assert_close(window_k[0, 0:sink], k[0:sink], atol=0, rtol=0)
    torch.testing.assert_close(window_k[0, sink:sink + recent], k[committed - recent:committed],
                               atol=0, rtol=0)
    owner.fill_(-1)
    lossy.zero_()
    ns.prefix_restore_out(
        torch.tensor([committed + 1], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([6], dtype=torch.int64, device=device),
        torch.zeros((1, 4), dtype=torch.int32, device=device),
        positions, window_k, window_v, state,
        staging_k, staging_v, owner, cache, identity, identity, lossy,
        sink, recent, 4, bs)
    torch.npu.synchronize()
    assert int(lossy.sum().item()) == sink + recent, "evicted staging must count lossy tokens"
    assert state[0, 3].item() == 0 and state[0, 1].item() == committed
    checks.append({"name": "stage_and_prefix_restore", "head_dim": dim,
                   "status": "passed", "lossy_rows_on_eviction": sink + recent})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    from .service_config import PHYSICAL_DEVICES
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = PHYSICAL_DEVICES
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    report = {"rank": rank, "status": "failed", "checks": [],
              "scope": "independent small NPU operators; not model TP4/MTP/prefix/graph acceptance",
              "model_graph": "not_run", "model_accuracy": "not_run", "performance": "not_run"}
    try:
        if rank not in range(4):
            raise ValueError("local rank exceeds the four explicitly assigned devices")
        from .ops import load_library
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(rank)
        namespace = load_library(args.library)
        report["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
        report["checks"] = run_probes(namespace, torch, f"npu:{rank}", capture=args.capture)
        report["checks"] += run_prefix_probes(namespace, torch, f"npu:{rank}")
        report["checks"] += run_eigensolver_probe(namespace, torch, f"npu:{rank}")
        torch.npu.synchronize()
        failures = [c for c in report["checks"] if c.get("status") == "failed"]
        if failures:
            report["status"] = "failed"
            report["error"] = "; ".join(f"{c.get('name')}({c.get('dim', c.get('head_dim', ''))})"
                                        for c in failures)
            return 1
        report["status"] = "passed"
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        # The target machine's operators paste stdout only; surface the full
        # failure there in addition to the JSON report file.
        print(report["traceback"], flush=True)
        print(json.dumps({"rank": rank, "status": "failed",
                          "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / f"rank_{rank}.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"rank": rank, "status": report["status"], "report": str(output)}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
