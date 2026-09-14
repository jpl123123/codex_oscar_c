#!/usr/bin/env python3
"""Hash only the supplied reference trees, without modifying or importing them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oscar_ascend.environment import source_state

TREES = ("vllm", "vllm-ascend", "oscar-vllm-pr46774", "oscar-paper")


def inventory(root: Path) -> dict:
    result = {}
    for name in TREES:
        tree = root / name
        files = {}
        if tree.is_symlink():
            raise ValueError(f"Reference tree must not be a symlink: {tree}")
        if tree.is_dir():
            for path in sorted(tree.rglob("*")):
                relative = path.relative_to(tree)
                if ".git" in relative.parts or ".DS_Store" in relative.parts:
                    continue
                if path.is_symlink():
                    # Record the link text, never follow it out of the reference scope.
                    files[str(relative)] = {"symlink": str(path.readlink())}
                elif path.is_file():
                    files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        result[name] = {"source": source_state(tree), "file_count": len(files),
                        "inventory_sha256": hashlib.sha256(canonical).hexdigest(), "files": files}
    return {"schema_version": 1, "trees": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "references")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = inventory(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.compare:
        previous = json.loads(args.compare.read_text())
        differences = [name for name in TREES if report["trees"][name]["files"] != previous["trees"][name]["files"]]
        if differences:
            print("Reference contents changed: " + ", ".join(differences), file=sys.stderr)
            return 1
    print(json.dumps({name: {key: val for key, val in data.items() if key != "files"}
                      for name, data in report["trees"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
