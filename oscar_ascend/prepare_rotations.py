"""Find a valid model-matched .pt, or generate it with a native Ascend engine."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable

from .calibration_data import build_prompts, digest, ensure_corpus, load_profile
from .runtime import OSCAR_REFERENCE_COMMIT, ROTATION_OBJECTIVES
from .service_config import target_argv


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_identity(model: Path) -> dict:
    """Config/index contents plus local weight file metadata; no weight loading.

    Metadata catches ordinary replacement, not an adversarial same-size/mtime
    rewrite. The artifact documents that limitation instead of calling it a
    content checksum of all model weights.
    """
    config = model / "config.json"
    config_hash = hash_file(config)
    files = sorted(set(model.glob("*.safetensors")) | set(model.glob("*.bin")))
    weights = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
    indices = {p.name: hash_file(p) for p in sorted(model.glob("*.index.json"))}
    text_files = {model / name for name in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "chat_template.jinja", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt")}
    text_files.update(model.glob("*.jinja"))
    text_files.update(model.glob("*.tiktoken"))
    text_files.update(model.glob("*.py"))
    tokenizers = {p.name: hash_file(p) for p in sorted(text_files) if p.is_file()}
    return {"model_config_sha256": config_hash, "weight_files_metadata": weights,
            "weight_index_sha256": indices, "tokenizer_template_code_sha256": tokenizers,
            "weight_identity_method": "file names/sizes/mtimes plus index content SHA256"}


def make_identity(*, service_config: Path, options, library: Path, profile: dict) -> dict:
    argv = target_argv(service_config)
    identity = model_identity(Path(argv[2]))
    package = Path(__file__).resolve().parent
    producers = {name: hash_file(package / name) if (package / name).is_file() else "missing"
                 for name in ("calibrate.py", "calibration_worker.py", "calibration_data.py")}
    versions = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity.update({"format": "oscar-ascend-calibration-request-v1", "model": argv[2],
                     "reference_commit": OSCAR_REFERENCE_COMMIT, "tensor_parallel_size": 4,
                     "group_size": options.group_size,
                     "clip_ratios": {"k": options.k_clip, "v": options.v_clip},
                     "objectives": ROTATION_OBJECTIVES,
                     "target_argv_sha256": digest(argv), "target_argv": argv,
                     "library_sha256": hash_file(library), "producer_source_sha256": producers,
                     "native_package_versions": versions,
                     "calibration_profile": profile,
                     "corpus_sha256": profile["dataset_sha256"]})
    return identity


def validate_artifact(path: Path, identity: dict) -> dict:
    import torch
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != "oscar-ascend-rotations-v1":
        raise ValueError("invalid rotation artifact format")
    for field in ("model", "model_config_sha256", "reference_commit", "tensor_parallel_size",
                  "group_size", "clip_ratios", "objectives"):
        if artifact.get(field) != identity[field]:
            raise ValueError(f"rotation artifact {field} mismatch")
    if artifact.get("calibration_request_sha256") != digest(identity):
        raise ValueError("rotation artifact calibration identity mismatch")
    ranks = artifact.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != {"0", "1", "2", "3"}:
        raise ValueError("rotation artifact must cover all four TP ranks")
    layers = None
    for rank, entries in ranks.items():
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"empty rotation coverage at rank {rank}")
        if layers is not None and set(entries) != layers:
            raise ValueError("rotation layer coverage differs across TP ranks")
        layers = set(entries)
        for name, matrices in entries.items():
            if not isinstance(matrices, dict) or set(matrices) != {"k", "v"}:
                raise ValueError(f"missing K/V matrices: rank {rank} {name}")
            for matrix in matrices.values():
                if not isinstance(matrix, torch.Tensor) or matrix.dtype != torch.float32 or matrix.ndim != 2:
                    raise ValueError(f"invalid rotation tensor: rank {rank} {name}")
                if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] not in (64, 128, 256) or not matrix.is_contiguous():
                    raise ValueError(f"unsupported rotation geometry: rank {rank} {name}")
            if rank != "0":
                for side in ("k", "v"):
                    # Comparing saved rotation constants is an artifact check,
                    # not CPU calibration or processing of online K/V.
                    if not torch.equal(matrices[side], ranks["0"][name][side]):
                        raise ValueError(f"rank {rank} does not share the paper per-layer {side} rotation: {name}")
    calibration = artifact.get("calibration", {})
    if not calibration.get("validated_on_npu"):
        raise ValueError("rotation artifact lacks NPU convergence/orthogonality validation")
    diagnostics = calibration.get("diagnostics", {})
    for rank, entries in ranks.items():
        reports = diagnostics.get(rank, {}).get("diagnostics", {})
        if set(reports) != set(entries):
            raise ValueError(f"NPU calibration diagnostics do not cover rank {rank}")
        for name in entries:
            for side in ("k", "v"):
                report = reports[name].get(side, {})
                solver = report.get("solver", [])
                error = report.get("rotation_roundtrip_max_error", float("inf"))
                if (len(solver) != 8 or solver[0] != 1 or solver[7] != 0
                        or not 0 <= error <= 0.02
                        or any(not 0 <= solver[i] <= 0.02 for i in (4, 5, 6))):
                    raise ValueError(f"failed/missing NPU eigensolver validation: rank {rank} {name}/{side}")
    return artifact


@contextmanager
def cache_lock(path: Path, timeout: float = 7500):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another calibration still owns {path}")
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ensure_rotations(*, service_config: Path, options, library: Path, profile_path: Path,
                     cache_root: Path, generate: Callable[[Path, Path, dict], None],
                     validator=validate_artifact, corpus_loader=ensure_corpus) -> dict:
    profile = load_profile(profile_path)
    identity = make_identity(service_config=service_config, options=options, library=library, profile=profile)
    key = digest(identity)
    directory = cache_root / key
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(options.rotations_path).resolve() if options.rotations_path else directory / "rotations.pt"
    # Explicit valid artifacts may be reused, but an invalid explicit file is
    # never overwritten on behalf of the caller.
    lock_path = destination.with_suffix(destination.suffix + ".lock") if options.rotations_path else directory / "prepare.lock"
    with cache_lock(lock_path):
        if destination.is_file():
            try:
                manifest_path = directory / "manifest.json"
                if not options.rotations_path:
                    if not manifest_path.is_file():
                        raise ValueError("automatic rotation cache lacks its completed publication manifest")
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("identity") != key or manifest.get("sha256") != hash_file(destination):
                        raise ValueError("cached .pt bytes changed after validated publication")
                validator(destination, identity)
                return {"path": str(destination), "status": "reused", "identity": key}
            except Exception as exc:
                if options.rotations_path:
                    raise ValueError(f"explicit rotation file is invalid and was preserved: {destination}: {exc}") from exc
                quarantine = destination.with_name("invalid-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".pt")
                destination.rename(quarantine)
        elif options.rotations_path:
            # The user-selected absent file is still a legitimate output target.
            destination.parent.mkdir(parents=True, exist_ok=True)
        raw = corpus_loader(profile, cache_root.parent / "calibration_data")
        prompts = build_prompts(raw, profile["num_prompts"], profile["seed"])
        request = dict(identity, calibration_request_sha256=key, prompts=prompts,
                       prompt_sha256=digest(prompts), seed=profile["seed"], max_tokens=profile["max_tokens"],
                       token_budget=profile["token_budget"], rpc_timeout_seconds=profile["rpc_timeout_seconds"],
                       target_service_config=str(service_config.resolve()), library_path=str(library.resolve()))
        # Stage on the destination filesystem: final rename must be atomic.
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
            temporary = Path(temporary)
            request_path, candidate = temporary / "request.json", temporary / "rotations.pt"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
            (directory / "request_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
            generate(request_path, candidate, profile)
            if not candidate.is_file():
                raise RuntimeError("native calibration exited without producing rotations.pt")
            validator(candidate, identity)
            with candidate.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(candidate, destination)
            # Publish a sidecar only after a validated .pt is durable.
            summary = {"path": str(destination), "status": "generated", "identity": key,
                       "sha256": hash_file(destination), "prompt_sha256": request["prompt_sha256"]}
            (directory / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
            return summary
