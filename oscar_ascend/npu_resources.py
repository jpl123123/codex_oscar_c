"""Confirm owned calibration workers no longer appear in the NPU process table."""
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
