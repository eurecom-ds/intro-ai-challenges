"""Measure CPU learning and reward comparisons with disposable outputs."""

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluator
import model
import protocol


def rehearsal_configurations(episodes):
    reference = protocol.settings(episodes=episodes)
    return [("baseline", protocol.BASELINE), ("reference", reference),
            ("less_food", {**reference, "food_reward": 1.0}),
            ("no_collision_penalty", {**reference, "collision_reward": 0.0}),
            ("survival", {**reference, "move_reward": .1}),
            ("distance", {**reference, "distance_reward": 1.0})]


def run_rehearsal(output, episodes=8192, seeds=(42, 43, 44), profiles=None):
    output = Path(output).resolve()
    if Path("/private/tmp") not in output.parents:
        raise ValueError("Keep disposable rehearsal output below /private/tmp.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__, "torch": torch.__version__,
        "training_threads": 1, "batch_episodes": model.BATCH_EPISODES,
        "evaluation_games": 100, "protocol_fingerprint": protocol.protocol_fingerprint(),
    }, indent=2), encoding="utf-8")
    public_pairs = protocol.public_seed_pairs()
    # Separate instructor games test the learning result beyond public selection.
    instructor_rng = np.random.default_rng(985311)
    instructor_pairs = [[int(x), int(y)] for x, y in instructor_rng.integers(0, 2**32, (100, 2), dtype=np.uint64)]
    records = []
    for seed in seeds:
        untrained = evaluator.evaluate(model.initial_policy(seed), public_pairs)
        records.append({"profile": "untrained", "seed": seed, "public": untrained})
        print(json.dumps(records[-1]), flush=True)
        for name, config in rehearsal_configurations(episodes):
            if profiles is not None and name not in profiles:
                continue
            result = model.train_policy(config, seed=seed)
            start = time.perf_counter()
            separate = evaluator.evaluate(result["policy"], instructor_pairs)
            row = {"profile": name, "seed": seed, "config": result["config"],
                   "seconds": result["training_seconds"], "steps": result["training_steps"],
                   "public": result["validation"], "instructor": separate,
                   "evaluation_seconds": time.perf_counter() - start,
                   "final_entropy": result["history"][-1]["entropy"],
                   "final_training_return": result["history"][-1]["training_return"]}
            records.append(row)
            (output / f"{name}-{seed}.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
            (output / "report.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(row), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--profiles", default="", help="Comma-separated profile names; empty runs all comparisons.")
    args = parser.parse_args()
    run_rehearsal(args.output, args.episodes, tuple(int(value) for value in args.seeds.split(",")),
                  set(args.profiles.split(",")) if args.profiles else None)


if __name__ == "__main__":
    main()
