"""Own and reap only process groups created by this invocation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .service_config import PHYSICAL_DEVICES, task_environment


def live_group_members(pgid: int) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,pgid=,stat="], text=True,
                            capture_output=True, timeout=10, check=True)
    members = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and int(parts[1]) == pgid and not parts[2].startswith("Z"):
            members.append(int(parts[0]))
    return members


class OwnedProcess:
    def __init__(self, argv: list[str], log: Path, *, enabled: bool = False,
                 extra_env: dict[str, str] | None = None):
        self.argv, self.log = argv, log
        self.env = task_environment(enabled=enabled)
        self.env.update(extra_env or {})
        # Never allow a phase to broaden the physical device selection.
        self.env["ASCEND_RT_VISIBLE_DEVICES"] = PHYSICAL_DEVICES
        self.process: subprocess.Popen | None = None
        self.handle = None
        self.observed_pids: set[int] = set()

    def start(self):
        # Refuse to create workers if this host cannot inspect/verify their exit.
        live_group_members(os.getpgrp())
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("w")
        try:
            self.process = subprocess.Popen(self.argv, env=self.env, stdout=self.handle,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.handle.close()
            raise
        self.observed_pids.add(self.process.pid)
        return self

    def _members(self):
        if self.process is None:
            return []
        members = live_group_members(self.process.pid)
        self.observed_pids.update(members)
        return members

    def cleanup(self, grace: float = 15) -> dict:
        if self.process is None:
            return {"status": "not_started"}
        pgid = self.process.pid
        for sig, deadline in ((signal.SIGTERM, grace), (signal.SIGKILL, 5)):
            if not self._members():
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
            until = time.monotonic() + deadline
            while time.monotonic() < until:
                self.process.poll()  # reap the leader, including after normal phase exit
                if not self._members():
                    break
                time.sleep(0.1)
        self.process.poll()
        remaining = self._members()
        if self.handle:
            self.handle.close()
        report = {"status": "released" if not remaining else "failed",
                  "process_group": pgid, "observed_pids": sorted(self.observed_pids),
                  "remaining_pids": remaining, "returncode": self.process.returncode,
                  "log": str(self.log), "npu_memory_release": "not_verified_by_process_check"}
        self.log.with_suffix(".process.json").write_text(json.dumps(report, indent=2) + "\n")
        if remaining:
            raise RuntimeError(f"Task process group {pgid} still owns live processes {remaining}")
        return report

    def wait(self, timeout: float) -> int:
        if not self.process:
            raise RuntimeError("Process not started")
        deadline = time.monotonic() + timeout
        while self.process.poll() is None:
            self._members()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out after {timeout}s: {self.argv}; log: {self.log}")
            time.sleep(0.1)
        return self.process.returncode


def run_phase(argv: list[str], log: Path, *, timeout: float, enabled: bool = False,
              extra_env: dict[str, str] | None = None) -> dict:
    phase = OwnedProcess(argv, log, enabled=enabled, extra_env=extra_env).start()
    try:
        code = phase.wait(timeout)
    finally:
        cleanup = phase.cleanup()
    if code:
        raise RuntimeError(f"Phase failed (exit {code}): {argv}; full log: {log}")
    return cleanup
