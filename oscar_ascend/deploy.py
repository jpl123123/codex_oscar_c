"""Bounded build/probe phases for the external package; never edits native code."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .environment import collect
from .processes import run_phase
from .service_config import target_argv

ROOT = Path(__file__).resolve().parents[1]

# Match common compiler/cmake/ninja error markers so a failed phase prints the
# actionable lines straight to the console. The target machine may have no
# outbound access, so stdout is the only channel back to development.
ERROR_LINE = re.compile(
    r"error:|error C\d|fatal:|FATAL|CMake Error|ninja: build stopped|"
    r"undefined reference|undefined symbol|No such file|not found|"
    r"No rule to make target|collect2:|ld returned|make(\[\d+\])?: \*\*\*|"
    r"ASCEND_HOME_PATH", re.IGNORECASE)
# Compiler command lines can exceed a thousand characters; cap what is echoed
# so a pasted report stays readable.
LINE_LIMIT = 600


def _clip(line: str) -> str:
    # Compiler/linker command lines can exceed a thousand characters and push
    # the actual diagnostic to the end of the line; keep a head sample plus the
    # full tail so a pasted report always contains the error text itself.
    if len(line) <= LINE_LIMIT:
        return line
    head, tail = 250, 500
    return f"{line[:head]} ...<middle clipped>... {line[-tail:]}"


def print_failure_details(logdir: Path, phase_name: str, *,
                          max_error_lines: int = 220, tail_lines: int = 60) -> None:
    log = logdir / (phase_name + ".log")
    if not log.is_file():
        return
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError as exc:
        print(f"[oscar-ascendc] could not read {log}: {exc}", flush=True)
        return
    print(f"\n[oscar-ascendc] ===== {phase_name}.log: {len(lines)} lines =====", flush=True)
    marked = [i for i, line in enumerate(lines) if ERROR_LINE.search(line)]
    if marked:
        spans, start, end = [], None, None
        for i in marked:
            lo, hi = max(0, i - 3), min(len(lines), i + 3)
            if start is None:
                start, end = lo, hi
            elif lo <= end + 2:
                end = max(end, hi)
            else:
                spans.append((start, end))
                start, end = lo, hi
        if start is not None:
            spans.append((start, end))
        shown = 0
        print("[oscar-ascendc] ----- error context -----", flush=True)
        for lo, hi in spans:
            if shown >= max_error_lines:
                print(f"[oscar-ascendc] ... {len(spans)} error regions total; "
                      f"showing first {max_error_lines} lines", flush=True)
                break
            for i in range(lo, hi):
                if shown >= max_error_lines:
                    break
                print(f"{i + 1:6d}| {_clip(lines[i])}", flush=True)
                shown += 1
    print(f"[oscar-ascendc] ----- last {tail_lines} lines -----", flush=True)
    for i, line in enumerate(lines[-tail_lines:], start=max(1, len(lines) - tail_lines + 1)):
        print(f"{i:6d}| {_clip(line)}", flush=True)
    print(f"[oscar-ascendc] ===== end {phase_name}.log =====\n", flush=True)
    print_artifact_diagnostics(lines)
    build_root = logdir.parent.parent / "build/ascendc"
    print_rule_context(lines, build_root)
    print_recent_aux_logs(build_root, logdir)


def print_artifact_diagnostics(lines: list[str]) -> None:
    """Identify object files the linker rejected as 'unknown file type'.

    Runs the host `file`/`ls` tools on the exact paths ld.lld rejected and
    prints the result, so a pasted report shows whether the artifact is empty,
    text, or a stale directory without another round trip.
    """
    paths = []
    for line in lines:
        if "unknown file type" not in line:
            continue
        for match in re.finditer(r"error:\s*([^\s:]+(?:/[^\s:]+)*)", line):
            candidate = match.group(1)
            if candidate not in paths:
                paths.append(candidate)
    for path in paths[:8]:
        print(f"[oscar-ascendc] artifact check: {path}", flush=True)
        for tool in (("file",), ("ls", "-la")):
            result = subprocess.run(list(tool) + [path], text=True, capture_output=True, timeout=15)
            output = (result.stdout or result.stderr).strip()
            if output:
                print(f"  $ {' '.join(tool)} -> {output}", flush=True)


def print_rule_context(lines: list[str], build_root: Path) -> None:
    """Print the make recipes referenced by failing build.make:NN markers.

    Some framework custom commands redirect their own output; the rule text
    is then the only evidence of what actually ran.
    """
    references: dict[str, set[int]] = {}
    for line in lines:
        for match in re.finditer(r"([\w./+-]*build\.make):(\d+)", line):
            references.setdefault(match.group(1), set()).add(int(match.group(2)))
    shown_rules = 0
    for name, linenos in sorted(references.items()):
        if shown_rules >= 3:
            break
        candidates = [Path(name)] if Path(name).is_absolute() else [build_root / name]
        rule = next((c for c in candidates if c.is_file()), None)
        if rule is None:
            matches = sorted(build_root.rglob(pathlib_name(name))) if name else []
            rule = matches[0] if matches else None
        if rule is None:
            print(f"[oscar-ascendc] rule file not found for {name}", flush=True)
            continue
        try:
            rule_lines = rule.read_text(errors="replace").splitlines()
        except OSError:
            continue
        shown_rules += 1
        for lineno in sorted(linenos)[:3]:
            lo, hi = max(0, lineno - 9), min(len(rule_lines), lineno + 12)
            print(f"[oscar-ascendc] ----- {rule}:{lineno} (recipe context) -----", flush=True)
            for i in range(lo, hi):
                print(f"{i + 1:6d}| {_clip(rule_lines[i])}", flush=True)


def pathlib_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def print_recent_aux_logs(build_root: Path, phase_logdir: Path) -> None:
    """Tail recently written auxiliary logs under the build tree."""
    if not build_root.is_dir():
        return
    cutoff = time.time() - 45 * 60
    reports = []
    for candidate in list(build_root.rglob("*.log")) + list(build_root.rglob("*.txt")):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if stat.st_mtime < cutoff or stat.st_size > 200_000:
            continue
        reports.append((stat.st_mtime, candidate))
    for _, path in sorted(reports, reverse=True)[:3]:
        try:
            tail = path.read_text(errors="replace").splitlines()[-40:]
        except OSError:
            continue
        print(f"[oscar-ascendc] ----- aux log {path} (last 40 lines) -----", flush=True)
        for i, line in enumerate(tail, start=max(1, len(tail) - 39)):
            print(f"{i:6d}| {_clip(line)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-check", action="store_true", help="run only local contract tests")
    parser.add_argument("--build-only", action="store_true", help="install/build; no NPU acceptance")
    parser.add_argument("--probe-only", action="store_true", help="build and run independent NPU operators; no service")
    parser.add_argument("--calibrate-only", action="store_true", help="build, reuse or generate valid rotations.pt, then exit")
    parser.add_argument("--calibration-profile", type=Path, default=ROOT / "configs/calibration.json")
    parser.add_argument("--soc-version", default=os.environ.get("OSCAR_SOC_VERSION"))
    parser.add_argument("--cann", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--logs", type=Path)
    args = parser.parse_args()
    if sum((args.host_check, args.build_only, args.probe_only, args.calibrate_only)) > 1:
        parser.error("select only one mode")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"
    logdir = args.logs or ROOT / "logs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logdir.mkdir(parents=True, exist_ok=True)
    status = {"phase": "configuration", "status": "running", "npu_acceptance": "not_run",
              "graph_capture": "not_run", "graph_replay": "not_run", "log_directory": str(logdir)}
    report_path = logdir / "status.json"

    def phase(name, argv, timeout, extra_env=None):
        status["phase"] = name
        report_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[oscar-ascendc] {name}; log: {logdir / (name + '.log')}", flush=True)
        return run_phase(argv, logdir / (name + ".log"), timeout=timeout,
                         extra_env=extra_env)

    try:
        argv = target_argv(ROOT / "configs/target_service.json")
        if args.host_check:
            # Ordinary host contracts need no NPU or permission to inspect processes.
            status["phase"] = "host-contracts"
            env = dict(os.environ, OSCAR_ASCEND_ENABLED="0")
            with (logdir / "host-contracts.log").open("w") as log:
                result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            if result.returncode:
                raise RuntimeError(f"host tests failed; see {logdir / 'host-contracts.log'}")
            text = (logdir / "host-contracts.log").read_text()
            found = re.search(r"Ran (\d+) tests?", text)
            skipped = re.search(r"skipped=(\d+)", text)
            count, skips = int(found.group(1)) if found else None, int(skipped.group(1)) if skipped else 0
            status["tests"] = {"collected": count, "skipped": skips,
                               "passed": count - skips if count is not None else None}
            status["status"] = "host_checks_passed_with_skips" if skips else "host_checks_passed"
            return 0

        status["phase"] = "target-environment"
        environment = collect(Path(argv[2]), [Path("/vllm-workspace/vllm"), Path("/vllm-workspace/vllm-ascend")])
        (logdir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        if environment["preflight_missing"]:
            raise RuntimeError("; ".join(environment["preflight_missing"]))
        if not args.cann:
            args.cann = next((p for p in (Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/latest")) if p.is_dir()), None)
        if not args.cann or not args.cann.is_dir():
            raise RuntimeError("ASCEND_HOME_PATH/--cann must identify the installed CANN toolkit")
        if not args.soc_version:
            raw = (environment.get("npu_smi") or {}).get("stdout", "")
            models = set(re.findall(r"(?:Ascend)?(910B\d*|910_93\w*)\b", raw, re.IGNORECASE))
            if len(models) == 1:
                args.soc_version = "Ascend" + next(iter(models))
            else:
                raise RuntimeError("cannot determine a unique supported SoC from npu-smi; set OSCAR_SOC_VERSION/--soc-version")
        phase("install", [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(ROOT)], 300)
        build = ROOT / "build/ascendc"
        # The ascendc.cmake flow drives nested ExternalProject build trees whose
        # stamps and intermediates survive top-level cleans; start from an empty
        # directory every run so no artifact from an older source revision can
        # be reused or misread (for example as an ld.lld 'unknown file type').
        shutil.rmtree(build, ignore_errors=True)
        phase("configure", ["cmake", "-S", str(ROOT / "csrc"), "-B", str(build),
                            f"-DASCEND_HOME_PATH={args.cann}", f"-DSOC_VERSION={args.soc_version}",
                            f"-DPython3_EXECUTABLE={sys.executable}"], 180)
        phase("build", ["cmake", "--build", str(build), "--parallel", "4"], 2400)
        library = build / "liboscar_ascend_ops.so"
        if not library.is_file():
            raise RuntimeError(f"expected fresh operator library missing: {library}")
        if args.build_only:
            status["status"] = "built_not_validated"
            return 0
        from .plugin import load_config
        from .prepare_rotations import ensure_rotations
        from .npu_resources import parse_process_table, verify_release
        options = load_config()

        def generate_rotations(request_path, candidate, profile):
            # The calibration engine uses native BF16 FULL cache, native GDN,
            # native weight quantization and native MTP. OSCAR routing is off.
            smi = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=15, check=True)
            occupied = [row for row in parse_process_table(smi.stdout) if row["npu"] in (4, 5, 6, 7)]
            if occupied:
                raise RuntimeError(f"native calibration requires free task devices; existing NPU processes were preserved: {occupied}")
            try:
                cleanup = phase("native-calibration", [sys.executable, "-m", "oscar_ascend.calibrate",
                                 "--request", str(request_path), "--output", str(candidate)],
                                profile["phase_timeout_seconds"],
                                extra_env={"OSCAR_ASCEND_ENABLED": "0", "PYTHONUNBUFFERED": "1"})
            finally:
                record = logdir / "native-calibration.process.json"
                if record.is_file():
                    cleanup = json.loads(record.read_text())
                    if cleanup["status"] != "released":
                        raise RuntimeError("native calibration workers have not exited")
                    verify_release(set(cleanup["observed_pids"]), logdir / "native-calibration.npu-release.json")

        status["phase"] = "prepare-rotations"
        print("[oscar-ascendc] checking model-matched rotations.pt; missing/invalid auto-cache triggers native calibration", flush=True)
        rotation = ensure_rotations(service_config=ROOT / "configs/target_service.json", options=options,
                                    library=library, profile_path=args.calibration_profile,
                                    cache_root=ROOT / "artifacts/rotations", generate=generate_rotations)
        status["rotations"] = rotation
        resolved_options = asdict(options)
        resolved_options.update(library_path=str(library.resolve()), rotations_path=rotation["path"])
        resolved_path = logdir / "oscar_config.json"
        resolved_path.write_text(json.dumps(resolved_options, indent=2) + "\n")
        os.environ["OSCAR_ASCEND_CONFIG"] = str(resolved_path.resolve())
        print(f"[oscar-ascendc] rotations {rotation['status']}: {rotation['path']}", flush=True)
        if args.calibrate_only:
            status["status"] = "rotations_ready_service_not_started"
            return 0
        phase("operators-ranks4", [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                "--nnodes=1", "--nproc-per-node=4", "-m", "oscar_ascend.probe",
                                "--library", str(library), "--output", str(logdir / "operators"),
                                "--capture"], 300)
        phase("window-pipeline", [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                                  "-p", "test_npu_window_pipeline.py", "-v"], 300,
              extra_env={"OSCAR_ASCEND_TEST_LIBRARY": str(library)})
        status["npu_acceptance"] = "operator_probe_only"
        if args.probe_only:
            status["status"] = "operators_probed_service_not_started"
            return 0

        # A successful tile probe cannot authorize a service with incomplete
        # request/prefix/MTP/graph semantics. This check is kept separate from
        # build and independent operator validation.
        from .readiness import require_service_implementation
        status["phase"] = "service-integration"
        require_service_implementation()
        raise RuntimeError("full integration probe and NPU resource-release verification remain unimplemented; formal service was not started")
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="failed", error=str(exc) or type(exc).__name__)
        print(f"[oscar-ascendc] FAILED phase={status['phase']}: {exc}\nFull logs: {logdir}", file=sys.stderr)
        # Operators may work on a machine with no outbound access; print the
        # failing phase's error context directly so it can be pasted back.
        print_failure_details(logdir, status["phase"])
        return 1
    finally:
        report_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[oscar-ascendc] result: {report_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
