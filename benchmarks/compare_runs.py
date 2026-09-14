#!/usr/bin/env python3
"""Compare paired per-workload measurements without averaging away slow cases.

Input JSON: manifest (hardware/model/software/input/launch/sampling identities),
cases: [{id, runs:[{ttft_ms,tpot_ms,e2e_ms,generation_tokens_per_s,
                   failed_requests,completed_requests}]}].
No numerical tolerance is silently introduced; noise policy is external and
must be declared before timing. A measured degradation is reported, not erased.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

IDENTITIES = ("hardware", "model", "software", "inputs", "launch", "sampling", "prefix_state", "warmup", "repetitions")
METRICS = {"ttft_ms": "lower", "tpot_ms": "lower", "e2e_ms": "lower", "generation_tokens_per_s": "higher"}


def compare(native, oscar):
    for key in IDENTITIES:
        if key not in native["manifest"] or native["manifest"][key] != oscar["manifest"].get(key):
            raise ValueError(f"unpaired or missing benchmark identity: {key}")
    a = {c["id"]: c for c in native["cases"]}
    b = {c["id"]: c for c in oscar["cases"]}
    if len(a) != len(native["cases"]) or len(b) != len(oscar["cases"]) or a.keys() != b.keys() or not a:
        raise ValueError("case IDs must be nonempty, unique, and identical")
    comparisons = []
    for key in a:
        left, right = a[key]["runs"], b[key]["runs"]
        repeats = native["manifest"]["repetitions"]
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2 or len(left) != repeats or len(right) != repeats:
            raise ValueError(f"{key}: expected >=2 declared repetitions on both sides")
        result = {"id": key, "metrics": {}, "failed": False}
        if any(run["failed_requests"] or run["completed_requests"] <= 0 for run in left + right):
            result["failed"] = True
            result["failure_reason"] = "request failures or empty timing sample"
        for metric, direction in METRICS.items():
            lv, rv = [r[metric] for r in left], [r[metric] for r in right]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and value > 0 for value in lv + rv):
                raise ValueError(f"{key}: {metric} measurements must be positive")
            lm, rm = statistics.median(lv), statistics.median(rv)
            degraded = rm > lm if direction == "lower" else rm < lm
            result["metrics"][metric] = {"native_median": lm, "oscar_median": rm,
                                         "oscar_over_native": rm / lm,
                                         "native_range": [min(lv), max(lv)], "oscar_range": [min(rv), max(rv)],
                                         "measured_degradation": degraded}
            result["failed"] |= degraded
        comparisons.append(result)
    return {"status": "measured_degradation_or_failure" if any(c["failed"] for c in comparisons) else "no_measured_median_degradation",
            "acceptance": "requires separate correctness, path, memory, percentile and noise-policy evidence",
            "cases": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oscar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.native.read_text()), json.loads(args.oscar.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])
    raise SystemExit(1 if any(c["failed"] for c in report["cases"]) else 0)
