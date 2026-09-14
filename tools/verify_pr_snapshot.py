#!/usr/bin/env python3
"""Compare the provided OSCAR snapshot with the exact declared upstream commit."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import ssl

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "57286d5d2cb08c3dcd8c17bb59e132d6985e6796"
SNAPSHOT = ROOT / "references/oscar-vllm-pr46774"


def check(path):
    relative = path.relative_to(SNAPSHOT).as_posix()
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{COMMIT}/{relative}"
    local_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        # macOS framework Python may lack its bundled CA file; use the system
        # trust bundle when available. Certificate/hostname verification stays on.
        context = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").is_file() else None)
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            upstream_sha = hashlib.sha256(response.read()).hexdigest()
        return {"path": relative, "url": url, "local_sha256": local_sha,
                "upstream_sha256": upstream_sha, "matches": local_sha == upstream_sha}
    except Exception as exc:
        return {"path": relative, "url": url, "local_sha256": local_sha,
                "matches": None, "error": str(exc)}


if __name__ == "__main__":
    files = [p for p in SNAPSHOT.rglob("*.py") if p.is_file() and not p.is_symlink()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, sorted(files)))
    report = {"commit": COMMIT, "snapshot_file_count": len(files),
              "status": "matched" if results and all(r["matches"] is True for r in results) else "unverified_or_mismatch",
              "files": results}
    output = ROOT / "reports/oscar_upstream_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "file_count": len(files), "report": str(output)}))
    raise SystemExit(0 if report["status"] == "matched" else 1)
