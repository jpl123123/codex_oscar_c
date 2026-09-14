"""Read-only environment inventory; never imports a platform or loads a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def command(argv: list[str], timeout: float = 20) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": str(exc)}


def distribution(name: str, module: str) -> dict:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec(module)
    return {"version": version, "module_path": spec.origin if spec else None}


def source_state(path: Path) -> dict:
    """Never let git walk upwards into an unrelated parent repository."""
    if not path.is_dir():
        return {"path": str(path), "status": "missing"}
    if not (path / ".git").exists():
        return {"path": str(path), "status": "snapshot_without_git"}
    top = command(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if top.get("returncode") != 0 or Path(top["stdout"].strip()).resolve() != path.resolve():
        return {"path": str(path), "status": "git_root_mismatch", "git": top}
    return {"path": str(path), "status": "git",
            "head": command(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "tracked_changes": command(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"])}


def collect(model: Path, native_sources: list[Path]) -> dict:
    packages = {name: distribution(name, module) for name, module in (
        ("vllm", "vllm"), ("vllm-ascend", "vllm_ascend"), ("torch", "torch"),
        ("torch-npu", "torch_npu"), ("triton", "triton"), ("packaging", "packaging"))}
    executables = {name: shutil.which(name) for name in ("npu-smi", "cmake", "bisheng", "ccec", "git")}
    model_config = model / "config.json"
    model_info = {"path": str(model), "exists": model.is_dir(), "config": None}
    if model_config.is_file():
        raw = model_config.read_bytes()
        model_info.update(config=json.loads(raw), config_sha256=hashlib.sha256(raw).hexdigest())
    # Only named build/platform variables; do not dump credentials from the environment.
    environment = {key: os.environ.get(key) for key in (
        "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH",
        "ASCEND_CUSTOM_OPP_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
        "VLLM_PLUGINS", "VLLM_WORKER_MULTIPROC_METHOD")}
    missing = []
    if platform.system() != "Linux":
        missing.append("Ascend execution requires the target Linux environment")
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        if not packages[name]["module_path"]:
            missing.append(f"missing {name}")
    if not model_info["exists"]:
        missing.append("model directory missing")
    for name in ("npu-smi", "cmake"):
        if not executables[name]:
            missing.append(f"missing executable {name}")
    return {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "release": platform.release(), "python": sys.version, "executable": sys.executable},
            "packages": packages, "executables": executables, "environment": environment,
            "model": model_info, "native_sources": [source_state(p) for p in native_sources],
            "npu_smi": command([executables["npu-smi"], "info"]) if executables["npu-smi"] else None,
            "preflight_missing": missing, "npu_kernel_validation": "not_run",
            "graph_capture": "not_run", "graph_replay": "not_run",
            "accuracy": "not_run", "performance": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp"))
    parser.add_argument("--native-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-target", action="store_true")
    args = parser.parse_args()
    report = collect(args.model, args.native_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Environment report: {args.output.resolve()}")
    for missing in report["preflight_missing"]:
        print(missing, file=sys.stderr)
    return 2 if args.require_target and report["preflight_missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
