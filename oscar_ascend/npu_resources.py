"""Confirm owned calibration workers exit and task devices return their HBM.

Process-table release alone is not enough: a force-killed worker can leave
device memory allocated with no owning row, and the next engine then fails
its startup free-memory check. The helpers here parse both views of npu-smi.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time


def parse_process_table(output: str) -> list[dict]:
    rows, header, empty = [], False, False
    for line in output.splitlines():
        lower = line.lower()
        if "no running process" in lower:
            empty = True
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) >= 2 and "npu" in cells[0].lower() and "process id" in cells[1].lower():
            header = True
            continue
        if header and len(cells) >= 4 and re.fullmatch(r"\d+(?:\s+\d+)?", cells[0]) and cells[1].isdigit():
            identity = [int(v) for v in cells[0].split()]
            memory = re.search(r"\d+(?:\.\d+)?", cells[3])
            if memory is None:
                raise ValueError("unrecognized process memory field in npu-smi")
            rows.append({"npu": identity[0], "chip": identity[1] if len(identity) > 1 else None,
                         "pid": int(cells[1]), "name": cells[2], "memory_mb": float(memory.group())})
    if not header and not empty:
        raise ValueError("npu-smi output has no recognized process table or explicit empty-process marker")
    return rows


def verify_release(pids: set[int], output: Path, *, timeout: float = 60, interval: float = 2,
                   runner=subprocess.run) -> dict:
    if not pids:
        raise ValueError("cannot verify release without recorded task process IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    samples = []
    while True:
        result = runner(["npu-smi", "info"], text=True, capture_output=True, timeout=15, check=True)
        rows = parse_process_table(result.stdout)
        owned = [row for row in rows if row["pid"] in pids]
        samples.append({"owned_rows": owned, "stdout": result.stdout, "stderr": result.stderr})
        if not owned:
            report = {"status": "released", "owned_pids": sorted(pids), "samples": samples,
                      "method": "owned OS process group exited and no owned PID/memory row in npu-smi",
                      "scope": "owned process allocations; no reset or termination of unrelated tasks"}
            output.write_text(json.dumps(report, indent=2) + "\n")
            return report
        if time.monotonic() >= deadline:
            report = {"status": "failed", "owned_pids": sorted(pids), "samples": samples}
            output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(f"owned NPU process allocations remain; report: {output}")
        time.sleep(min(interval, max(0, deadline - time.monotonic())))


def parse_memory_usage(output: str) -> dict[int, dict]:
    """Per-device HBM usage from the npu-smi main table.

    Rows look like "| 4  910B4  ... |  13012 / 27648 MB | ..."; the first
    "used / total MB" field on a device row is the HBM line. Devices without
    a recognized pattern are reported as unparsed rather than silently free.
    """
    usage: dict[int, dict] = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if not cells or not cells[0].isdigit():
            continue
        device = int(cells[0])
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*M?B", " ".join(cells[1:]))
        if match:
            used, total = float(match.group(1)), float(match.group(2))
            usage[device] = {"used_mb": used, "total_mb": total,
                             "free_mb": total - used}
    return usage


def insufficient_devices(output: str, devices: tuple[int, ...], required_free_mb: float) -> list[dict]:
    """Devices that cannot satisfy the engine's startup free-memory request."""
    usage = parse_memory_usage(output)
    blocked = []
    for device in devices:
        entry = usage.get(device)
        if entry is None:
            blocked.append({"npu": device, "reason": "npu-smi memory field not recognized"})
        elif entry["free_mb"] < required_free_mb:
            blocked.append({"npu": device, **entry, "required_free_mb": required_free_mb})
    return blocked


def wait_for_memory(output: Path, devices: tuple[int, ...], required_free_mb: float,
                    *, timeout: float = 180, interval: float = 5,
                    runner=subprocess.run) -> dict:
    """Poll npu-smi until the task devices free the requested HBM.

    Device memory can lag process exit by tens of seconds after abnormal
    termination; report the final table instead of guessing at owners.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    samples = []
    while True:
        result = runner(["npu-smi", "info"], text=True, capture_output=True, timeout=15, check=True)
        blocked = insufficient_devices(result.stdout, devices, required_free_mb)
        samples.append({"blocked": blocked, "stdout": result.stdout})
        if not blocked:
            report = {"status": "released", "devices": list(devices),
                      "required_free_mb": required_free_mb, "samples": samples[-1]}
            output.write_text(json.dumps(report, indent=2) + "\n")
            return report
        if time.monotonic() >= deadline:
            report = {"status": "failed", "devices": list(devices),
                      "required_free_mb": required_free_mb, "blocked": blocked,
                      "npu_smi_stdout": result.stdout}
            output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(
                f"task NPU devices still hold memory below the free threshold; report: {output}")
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
