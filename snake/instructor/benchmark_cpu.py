"""Compare reference, eager tensor, compiled tensor and Numba CPU collection.

Outputs contain timing, parity checks and optional trained policies. Keep them
in a private instructor directory outside the public activity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, help="A numeric actor export or instructor checkpoint.")
    parser.add_argument("--batches", type=int, default=16, help="Complete 32-game batches per timing repeat.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=("reference", "torch_eager", "torch_compile", "numba"),
                        default=("reference", "torch_eager", "torch_compile", "numba"))
    parser.add_argument("--train-episodes", type=int, choices=(0, *protocol.EPISODES), default=0,
                        help="Also compare complete reference and JIT CPU training runs.")
    parser.add_argument("--fresh-cache", action="store_true", help="Use a new compiler cache for a cold measurement.")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if source == output or source in output.parents:
        parser.error("Choose an output directory outside the public Snake activity.")
    if not 1 <= args.batches <= 1024 or not 1 <= args.repeats <= 20:
        parser.error("Use 1–1024 batches and 1–20 repeats.")
    if "reference" not in args.backends:
        parser.error("Include the reference backend for trajectory parity checks.")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if args.fresh_cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor-", dir=output)
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba-", dir=output)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import evaluator
    import model
    import protocol

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw_policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else model.initial_policy()
    policy = evaluator.validate_policy(raw_policy.get("policy", raw_policy))
    actor = model.PolicyNetwork().eval()
    with torch.no_grad():
        for index, parameters in enumerate(policy["layers"]):
            layer = actor.layers[2 * index]
            layer.weight.copy_(torch.as_tensor(parameters["weight"], dtype=torch.float32))
            layer.bias.copy_(torch.as_tensor(parameters["bias"], dtype=torch.float32))
    config = protocol.settings(episodes=256, learning_rate=.003, distance_reward=1)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "torch": torch.__version__, "cpu_threads": 1,
              "board_size": protocol.BOARD_SIZE, "max_steps": protocol.MAX_STEPS,
              "batch_episodes": model.BATCH_EPISODES, "batches": args.batches,
              "repeats": args.repeats, "fresh_cache": args.fresh_cache,
              "compiler_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
              "numba_cache": os.environ.get("NUMBA_CACHE_DIR"),
              "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in ("model.py", "torch_environment.py", "numba_environment.py")
                                if (source / name).exists()},
              "protocol_fingerprint": protocol.protocol_fingerprint(), "collection": [], "training": []}
    if "numba" in args.backends:
        try:
            report["numba"] = importlib.metadata.version("numba")
            report["numba_cache_policy"] = "in-process specializations; disk cache disabled"
        except importlib.metadata.PackageNotFoundError:
            parser.error("Install the optional numba package to benchmark its CPU backend.")

    def collect(backend, batches):
        rng = np.random.default_rng(np.random.SeedSequence([42, 1]))
        action_rng = torch.Generator().manual_seed(42 ^ 0x315A)
        digest = hashlib.sha256()
        elapsed, steps = 0.0, 0
        for _ in range(batches):
            start = time.perf_counter()
            if backend == "reference":
                batch = model.collect_episodes(actor, config, rng, action_rng)
            elif backend.startswith("torch_"):
                batch = model.collect_episodes_torch(actor, config, rng, action_rng, backend=backend)
            else:
                batch = model.collect_episodes_numba(actor, config, rng, action_rng)
            elapsed += time.perf_counter() - start
            steps += int(batch["lengths"].sum())
            # Hash outside the timed region, preserving full trajectory evidence.
            for key in ("observations", "actions", "returns", "discounts", "lengths"):
                digest.update(np.ascontiguousarray(batch[key]).tobytes())
        return elapsed, steps, digest.hexdigest()

    expected = None
    model.reset_torch_kernel_cache()
    model.reset_numba_kernel_cache()
    for backend in ("reference", *(name for name in args.backends if name != "reference")):
        cold, _, _ = collect(backend, 1)
        timings = []
        for _ in range(args.repeats):
            seconds, steps, fingerprint = collect(backend, args.batches)
            if expected is None:
                expected = fingerprint
            if fingerprint != expected:
                raise ValueError("Trajectory parity failed for " + backend)
            timings.append(seconds)
        row = {"backend": backend, "first_batch_seconds": cold,
               "warm_seconds": timings, "median_warm_seconds": float(np.median(timings)),
               "moves_per_repeat": steps, "moves_per_second": steps / float(np.median(timings)),
               "trajectory_sha256": fingerprint, "bitwise_trajectory_parity": True}
        if backend.startswith("torch_"):
            row["compile_seconds"] = model.get_torch_kernel(backend).compile_seconds
        elif backend == "numba":
            row["compile_seconds"] = model.get_numba_kernel().compile_seconds
        report["collection"].append(row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)

    if args.train_episodes:
        training_config = protocol.settings(episodes=args.train_episodes, learning_rate=.003, distance_reward=1)
        reference_policy = None
        for backend in ("reference", *(name for name in ("torch_compile", "numba") if name in args.backends)):
            checkpoint = model.train_policy(training_config, backend=backend)
            if reference_policy is None:
                reference_policy = checkpoint["policy"]
            if checkpoint["policy"] != reference_policy:
                raise ValueError("Trained policy differs between reference and compiled CPU execution.")
            write_json(output / (backend + "-policy.json"), checkpoint)
            row = {"backend": backend, "episodes": args.train_episodes,
                   "seconds": checkpoint["training_seconds"], "steps": checkpoint["training_steps"],
                   "compile_seconds": checkpoint.get("compile_seconds", 0),
                   "validation": checkpoint["validation"], "identical_trained_policy": True}
            report["training"].append(row)
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
    print("CPU benchmark report:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
