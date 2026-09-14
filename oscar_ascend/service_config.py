"""Target CLI configuration and reproducible command construction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

PHYSICAL_DEVICES = "4,5,6,7"


def target_argv(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    argv = data["argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("target_service.argv must be a string array")
    if argv[:2] != ["vllm", "serve"]:
        raise ValueError("target_service.argv must start with vllm serve")
    required = {"--tensor-parallel-size": "4", "--data-parallel-size": "1",
                "--port": "8989", "--max-model-len": "262144",
                "--max-num-batched-tokens": "16384", "--max-num-seqs": "128",
                "--quantization": "ascend", "--mamba-cache-dtype": "bfloat16",
                "--mamba-ssm-cache-dtype": "bfloat16"}
    flags = [item for item in argv if item.startswith("--")]
    if len(flags) != len(set(flags)):
        raise ValueError("duplicate CLI options are ambiguous")
    for option, value in required.items():
        if option not in argv or argv[argv.index(option) + 1] != value:
            raise ValueError(f"required target option changed: {option}={value}")
    if "--async-scheduling" not in argv or "--enforce-eager" in argv:
        raise ValueError("target requires async scheduling without global enforce-eager")
    speculative = json.loads(argv[argv.index("--speculative_config") + 1])
    if speculative != {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": True}:
        raise ValueError("target MTP configuration changed")
    compilation = json.loads(argv[argv.index("--compilation-config") + 1])
    if compilation["cudagraph_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("target requires FULL_DECODE_ONLY")
    return argv


def task_environment(*, enabled: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["ASCEND_RT_VISIBLE_DEVICES"] = PHYSICAL_DEVICES
    env["OSCAR_ASCEND_ENABLED"] = "1" if enabled else "0"
    return env


def display_command(path: Path) -> str:
    return shlex.join(target_argv(path))
