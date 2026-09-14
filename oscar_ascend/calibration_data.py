"""Pinned GPQA prompt construction matching the provided paper dump recipe.

Only text/configuration is processed here, never model K/V or numerical stats.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import random
import ssl
import tempfile
import urllib.request
import os

TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def build_prompts(raw: bytes, count: int, seed: int) -> list[str]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(rows):
        raise ValueError("num_prompts must be positive and no larger than the pinned dataset")
    rng = random.Random(seed)
    if count < len(rows):
        rows = rng.sample(rows, count)
    prompts = []
    for row in rows:
        choices = [row[key] for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
        choices = [choices[i] for i in rng.sample(range(4), 4)]
        prompts.append(TEMPLATE.format(Question=row["Question"], **dict(zip("ABCD", choices))))
    return prompts


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    if profile.get("format") != "oscar-calibration-profile-v1":
        raise ValueError("unrecognized calibration profile")
    if profile.get("method") != "qqt_sst" or profile.get("composition") != "r_h_pbr":
        raise ValueError("this implementation requires the paper QQT/SST + U H Pbr method")
    for name in ("num_prompts", "token_budget", "max_tokens", "rpc_timeout_seconds", "phase_timeout_seconds"):
        value = profile.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid calibration {name}")
    if profile["max_tokens"] != 1:
        raise ValueError("calibration captures prefill with max_tokens=1, matching the paper dump recipe")
    if isinstance(profile.get("seed"), bool) or not isinstance(profile.get("seed"), int):
        raise ValueError("calibration seed must be an explicit integer")
    if profile.get("statistics_dtype") != "float32" or profile.get("solver") != "ascendc_symmetric_jacobi":
        raise ValueError("calibration profile does not describe the implemented NPU numerical method")
    if profile.get("prompt_recipe") != "oscar-gpqa-multichoice-seed0-v1":
        raise ValueError("unrecognized calibration prompt construction recipe")
    if len(profile.get("dataset_sha256", "")) != 64:
        raise ValueError("calibration corpus must have a pinned SHA256")
    return profile


def ensure_corpus(profile: dict, cache_directory: Path) -> bytes:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / (profile["dataset_sha256"] + ".csv")
    if path.is_file():
        raw = path.read_bytes()
    elif profile.get("dataset_path"):
        source = Path(profile["dataset_path"])
        if not source.is_file() and not source.is_absolute():
            # Air-gapped machines receive the vendored corpus through the
            # repository; resolve the configured relative path against the
            # project root so the working directory does not matter.
            source = Path(__file__).resolve().parents[1] / profile["dataset_path"]
        raw = source.read_bytes()
    else:
        context = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").is_file() else None)
        with urllib.request.urlopen(profile["dataset_url"], context=context, timeout=60) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("calibration CSV exceeds the declared small-corpus download limit")
    if hashlib.sha256(raw).hexdigest() != profile["dataset_sha256"]:
        raise ValueError("calibration dataset checksum mismatch; the source changed or cache is corrupt")
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=".corpus-", dir=cache_directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return raw
