"""Generate a rotation candidate using an unmodified native TP4 Ascend engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

from .calibration_data import digest
from .runtime import OSCAR_REFERENCE_COMMIT, ROTATION_OBJECTIVES
from .service_config import PHYSICAL_DEVICES, target_argv


def print_environment_fingerprint() -> None:
    """Record the dynamic-library/thread-pool environment before torch starts.

    The historical failure mode on this host class (ParallelOpenMP.cpp:64
    'Invalid thread pool!') is an OpenMP runtime conflict, typically caused by
    a system-wide LD_PRELOAD mixing runtimes with torch's bundled one. The
    fingerprint lands in the phase log so a failure paste shows exactly which
    OpenMP libraries the process tree loaded.
    """
    keys = ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
            "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS",
            "OMP_THREAD_LIMIT", "OMP_SCHEDULE", "KMP_AFFINITY", "KMP_BLOCKTIME",
            "MKL_NUM_THREADS", "ASCEND_HOME_PATH", "ASCEND_AICPU_PATH",
            "ASCEND_RT_VISIBLE_DEVICES")
    environment = {key: os.environ.get(key) for key in keys}
    print(json.dumps({"calibration_environment": environment}, ensure_ascii=False), flush=True)


def validate_request(request: dict) -> None:
    if request.get("format") != "oscar-ascend-calibration-request-v1":
        raise ValueError("unknown calibration request format")
    if request.get("reference_commit") != OSCAR_REFERENCE_COMMIT:
        raise ValueError("calibration reference commit mismatch")
    if request.get("tensor_parallel_size") != 4 or request.get("max_tokens") != 1:
        raise ValueError("calibration requires TP4 and the declared max_tokens=1 prefill recipe")
    if request.get("objectives") != ROTATION_OBJECTIVES:
        raise ValueError("calibration objectives must be exact QQT/SST with R H Pbr")
    prompts = request.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        raise ValueError("calibration requires a fixed nonempty text-prompt list")
    if digest(prompts) != request.get("prompt_sha256"):
        raise ValueError("calibration prompt hash mismatch")
    for name in ("token_budget", "rpc_timeout_seconds", "group_size"):
        value = request.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid calibration {name}")
    for name in ("calibration_request_sha256", "model_config_sha256", "library_sha256", "corpus_sha256"):
        value = request.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing calibration identity field {name}")
    model_hash = hashlib.sha256((Path(request["model"]) / "config.json").read_bytes()).hexdigest()
    if model_hash != request["model_config_sha256"]:
        raise ValueError("model config changed before calibration")
    library_hash = hashlib.sha256(Path(request["library_path"]).read_bytes()).hexdigest()
    if library_hash != request["library_sha256"]:
        raise ValueError("AscendC artifact changed before calibration")
    if not isinstance(request.get("calibration_profile"), dict):
        raise ValueError("calibration profile metadata is required")


def native_llm_kwargs(request: dict) -> dict:
    """Preserve target engine arguments; omit HTTP-only address options."""
    argv = target_argv(Path(request["target_service_config"]))
    if argv[2] != request["model"] or digest(argv) != request.get("target_argv_sha256"):
        raise ValueError("native target service arguments differ from calibration identity")
    result = {"model": argv[2]}
    booleans = {"trust_remote_code", "async_scheduling"}
    integers = {"data_parallel_size", "tensor_parallel_size", "max_model_len",
                "max_num_batched_tokens", "max_num_seqs", "mm_processor_cache_gb"}
    json_values = {"compilation_config", "speculative_config", "additional_config", "hf_overrides"}
    i = 3
    while i < len(argv):
        flag = argv[i]
        if not flag.startswith("--"):
            raise ValueError("unexpected positional target argument")
        key = flag[2:].replace("-", "_")
        if key in booleans:
            result[key] = True
            i += 1
            continue
        if i + 1 >= len(argv):
            raise ValueError(f"missing value for {flag}")
        value = argv[i + 1]
        i += 2
        if key in ("host", "port"):
            continue
        if key in integers:
            value = int(value)
        elif key == "gpu_memory_utilization":
            value = float(value)
        elif key in json_values:
            value = json.loads(value)
        result[key] = value
    result["worker_extension_cls"] = "oscar_ascend.calibration_worker.CalibrationWorkerExtension"
    result["seed"] = request.get("seed", 0)
    # Keep speculative eager scoped to the drafter. There is no global eager
    # override and no KV/weight-quantization change for calibration.
    return result


def validate_workers(reports: list[dict]) -> list[int]:
    if len(reports) != 4 or {entry["rank"] for entry in reports} != {0, 1, 2, 3}:
        raise RuntimeError("native calibration did not initialize all four TP ranks")
    layers = set(reports[0]["full_layers"])
    if not layers or any(set(entry["full_layers"]) != layers for entry in reports):
        raise RuntimeError("FULL layer coverage differs across native TP ranks")
    if any(not entry["mtp_layers"] for entry in reports):
        raise RuntimeError("MTP FULL layer coverage is missing")
    pids = [entry["pid"] for entry in reports]
    if len(set(pids)) != 4:
        raise RuntimeError("TP calibration ranks did not report distinct worker processes")
    return pids


def wait_for_worker_exit(pids: list[int], timeout: float = 60) -> None:
    if not pids:
        raise RuntimeError("worker exit cannot be verified without observed worker IDs")
    deadline = time.monotonic() + timeout
    while True:
        process = subprocess.run(["ps", "-p", ",".join(map(str, pids)), "-o", "pid=,stat="],
                                 capture_output=True, text=True, timeout=10)
        if process.returncode not in (0, 1):
            raise RuntimeError("cannot verify native worker exit: " + process.stderr.strip())
        live = [line for line in process.stdout.splitlines()
                if len(line.split()) >= 2 and not line.split()[1].startswith("Z")]
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"native calibration workers remain alive after shutdown: {live}")
        time.sleep(0.1)


def run_calibration(request: dict, output: Path) -> dict:
    print_environment_fingerprint()
    validate_request(request)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = PHYSICAL_DEVICES
    os.environ["OSCAR_ASCEND_ENABLED"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "0"
    from vllm import LLM, SamplingParams
    import torch

    # Which OpenMP runtime did this process actually bind to? The historical
    # 'Invalid thread pool!' assert comes from mixed runtimes, so the maps of
    # the parent (inherited by every spawned worker) are part of the evidence.
    try:
        mappings = [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libgomp" in line or "libomp" in line or "libaiomp" in line]
        print(json.dumps({"calibration_openmp_libraries": sorted(set(mappings))},
                         ensure_ascii=False), flush=True)
    except OSError:
        pass
    try:
        print(json.dumps({"calibration_torch_parallel_info": torch.__config__.parallel_info(),
                          "calibration_torch_num_threads": torch.get_num_threads()},
                         ensure_ascii=False), flush=True)
    except Exception:
        pass

    llm = None
    worker_pids = []
    results = None
    timeout = request["rpc_timeout_seconds"]
    report = {"scope": "native-engine NPU calibration, not model accuracy/performance acceptance",
              "status": "running", "worker_exit_verified": False,
              "npu_memory_release": "outer-owned-phase-must-verify", "worker_pids": worker_pids}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-shards-", dir=output.parent) as temporary:
        try:
            llm = LLM(**native_llm_kwargs(request))
            worker_reports = llm.collective_rpc("oscar_calibration_begin", timeout=timeout,
                                                 args=(request, temporary))
            worker_pids = validate_workers(worker_reports)
            report.update(worker_pids=worker_pids, workers=worker_reports)
            pass_traces = []
            selected_prompts = None
            for pass_index in range(2):
                if not llm.reset_prefix_cache():
                    raise RuntimeError("native prefix cache reset failed before calibration pass")
                sampling = SamplingParams(temperature=0.0, max_tokens=1, seed=request.get("seed", 0))
                traces = []
                prompts = request["prompts"] if selected_prompts is None else selected_prompts
                for prompt in prompts:
                    responses = llm.chat([{"role": "user", "content": prompt}],
                                         sampling_params=sampling, use_tqdm=False)
                    if len(responses) != 1 or not responses[0].finished:
                        raise RuntimeError("native calibration request did not finish deterministically")
                    response = responses[0]
                    token_ids = response.prompt_token_ids
                    if not token_ids:
                        raise RuntimeError("native calibration response lacks token identity metadata")
                    traces.append({"prompt_tokens": len(token_ids), "prompt_tokens_sha256": digest(token_ids),
                                   "output_tokens": [list(o.token_ids) for o in response.outputs]})
                    statuses = llm.collective_rpc("oscar_calibration_status", timeout=timeout)
                    if (pass_index == 0 and all(entry["tokens"] >= request["token_budget"]
                                               for status in statuses for entry in status["layers"].values())):
                        break
                pass_traces.append(traces)
                if pass_index == 0:
                    selected_prompts = prompts[:len(traces)]
                    report["first_pass"] = llm.collective_rpc("oscar_calibration_next_pass", timeout=timeout)
                elif pass_traces[0] != pass_traces[1]:
                    raise RuntimeError("calibration passes tokenized or generated different token sequences")
            finished = llm.collective_rpc("oscar_calibration_finish", timeout=timeout)
            if len(finished) != 4 or not all(item.get("validated_on_npu") for item in finished):
                raise RuntimeError("not every native worker validated its NPU calibration")
            results = {}
            diagnostics = {}
            for item in finished:
                path = Path(item["shard"]).resolve()
                if path.parent != Path(temporary).resolve():
                    raise RuntimeError("worker rotation output escaped the owned temporary directory")
                shard = torch.load(path, map_location="cpu", weights_only=True)
                if shard.get("rank") != item["rank"] or not shard.get("validated_on_npu"):
                    raise RuntimeError("worker rotation shard identity mismatch")
                results[str(item["rank"])] = shard["layers"]
                diagnostics[str(item["rank"])] = {key: shard[key] for key in ("diagnostics", "traces", "fingerprints")}
            report.update(pass_token_trace_sha256=digest(pass_traces[0]),
                          prompts_used=len(selected_prompts), diagnostics=diagnostics)
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            cleanup_errors = []
            if llm is not None:
                try:
                    llm.collective_rpc("oscar_calibration_abort", timeout=min(timeout, 60))
                except Exception as exc:
                    cleanup_errors.append(f"hook cleanup: {exc}")
                try:
                    # Current public LLM has no shutdown method. The verified
                    # core-client API owns its engine manager and native workers.
                    llm.llm_engine.engine_core.shutdown(timeout=60)
                except Exception as exc:
                    cleanup_errors.append(f"engine shutdown: {exc}")
                if worker_pids:
                    try:
                        wait_for_worker_exit(worker_pids)
                        report["worker_exit_verified"] = True
                    except Exception as exc:
                        cleanup_errors.append(f"worker exit: {exc}")
            report["cleanup_errors"] = cleanup_errors
            report_path = output.with_suffix(".calibration.json")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if cleanup_errors:
                raise RuntimeError("native calibration cleanup failed: " + "; ".join(cleanup_errors))
        if results is None or not report["worker_exit_verified"]:
            raise RuntimeError("calibration has no completed results or verified worker shutdown")
        artifact = {key: request[key] for key in (
            "model", "model_config_sha256", "reference_commit", "tensor_parallel_size",
            "group_size", "clip_ratios", "objectives", "calibration_request_sha256",
            "calibration_profile", "corpus_sha256", "prompt_sha256",
        )}
        artifact.update(format="oscar-ascend-rotations-v1", ranks=results,
                        calibration={"validated_on_npu": True, "diagnostics": report["diagnostics"],
                                     "solver": "ascendc_symmetric_jacobi", "statistics_dtype": "float32",
                                     "max_sweeps": request.get("max_sweeps", 32),
                                     "tolerance": request.get("solver_tolerance", 1e-6),
                                     "pass_token_trace_sha256": report["pass_token_trace_sha256"],
                                     "prompts_used": report["prompts_used"],
                                     "worker_exit_verified": True})
        torch.save(artifact, output)
        report["status"] = "candidate_generated_and_workers_exited"
        output.with_suffix(".calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_calibration(json.loads(args.request.read_text()), args.output)
        print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"status": "calibration_failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
