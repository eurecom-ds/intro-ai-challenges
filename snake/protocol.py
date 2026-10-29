"""Versioned rules and classroom controls for the Snake activity."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CHALLENGE_VERSION = "snake-v1"
BOARD_SIZE = 6
BOARD_CELLS = BOARD_SIZE ** 2
INITIAL_LENGTH = 3
MAX_STEPS = 300
OBSERVATION_SIZE = 118
ARCHITECTURE = "mlp-118-64-64-3-relu-v1"
ACTION_NAMES = ("straight", "left", "right")
LAYER_SHAPES = ((64, 118), (64, 64), (3, 64))
MAX_PARAMETER = 1_000_000.0
MAX_JSON_BYTES = 512 * 1024
TRAINING_SEED = 42
BATCH_SIZE = 32
EPISODES = (256, 4096, 8192, 16384)
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003, 0.01)
GAMMAS = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
PUBLIC_EPISODES = 30
PRIVATE_EPISODES = 100
DEMO_SEED_PAIR = (670_001, 670_002)
MODEL_FIELDS = frozenset({"episodes", "learning_rate", "gamma", "food_reward",
                          "collision_reward", "move_reward", "distance_reward", "seed"})


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Set {name} to a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Set {name} to a finite number.") from None
    if not math.isfinite(result):
        raise ValueError(f"Set {name} to a finite number.")
    return 0.0 if result == 0 else result


def settings(episodes=8192, learning_rate=0.001, gamma=0.99, food_reward=10.0,
             collision_reward=-10.0, move_reward=-0.01, distance_reward=0.0,
             seed=TRAINING_SEED, *, instructor=False):
    """Validate controls; instructor rehearsals can use another training seed."""
    if isinstance(episodes, (bool, np.bool_)) or not isinstance(episodes, (int, np.integer)) or episodes not in EPISODES:
        raise ValueError(f"Choose episodes from {list(EPISODES)}.")
    lr = _number(learning_rate, "learning_rate")
    discount = _number(gamma, "gamma")
    if lr not in LEARNING_RATES:
        raise ValueError(f"Choose learning_rate from {list(LEARNING_RATES)}.")
    if discount not in GAMMAS:
        raise ValueError(f"Choose gamma from {list(GAMMAS)}.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
        raise ValueError("Set seed to an unsigned 32-bit integer.")
    if not instructor and seed != TRAINING_SEED:
        raise ValueError(f"Classroom comparisons use seed {TRAINING_SEED}.")
    rewards = {}
    for name, value, bound in (("food_reward", food_reward, 100),
                                ("collision_reward", collision_reward, 100),
                                ("move_reward", move_reward, 10),
                                ("distance_reward", distance_reward, 10)):
        number = _number(value, name)
        if not -bound <= number <= bound:
            raise ValueError(f"Set {name} between {-bound} and {bound}.")
        rewards[name] = number
    return {"episodes": int(episodes), "learning_rate": lr, "gamma": discount,
            **rewards, "seed": int(seed)}


BASELINE = settings(episodes=256)


def public_seed_pairs():
    """The fixed, public environment/action seeds used for model selection."""
    return [(630_000 + i, 730_000 + i) for i in range(PUBLIC_EPISODES)]


def validate_seed_pairs(value, *, count=None):
    if not isinstance(value, (list, tuple)) or not value or (count is not None and len(value) != count):
        raise ValueError("Use the prescribed nonempty list of evaluation seed pairs.")
    pairs = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each evaluation game needs an environment seed and an action seed.")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or not 0 <= v < 2 ** 32 for v in pair):
            raise ValueError("Evaluation seeds must be unsigned 32-bit integers.")
        pairs.append((int(pair[0]), int(pair[1])))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Evaluation seed pairs must be distinct.")
    return pairs


def protocol_definition():
    return {"challenge_version": CHALLENGE_VERSION, "board_size": BOARD_SIZE,
            "initial_length": INITIAL_LENGTH, "max_steps": MAX_STEPS,
            "actions": list(ACTION_NAMES), "heading": ["up", "right", "down", "left"],
            "observation_size": OBSERVATION_SIZE,
            "observation_order": ["body_age/36", "head", "food", "heading_one_hot",
                                  "remaining_steps/300", "food_row_offset/5", "food_col_offset/5",
                                  "collision_straight", "collision_left", "collision_right"],
            "architecture": ARCHITECTURE, "parameter_dtype": "float32",
            "inference_dtype": "float64", "sampling": "PCG64-uniform-cdf",
            "ranking": ["total_apples:descending", "sum_food_steps:ascending", "acceptance_order:ascending"],
            "public_episodes": PUBLIC_EPISODES, "private_episodes": PRIVATE_EPISODES,
            "public_seed_pairs": public_seed_pairs(), "demo_seed_pair": list(DEMO_SEED_PAIR)}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_fingerprint():
    """Bind rules, observation order and trusted inference implementation."""
    directory = Path(__file__).resolve().parent
    definition = {"definition": protocol_definition(), "sources": {
        name: sha256_file(directory / name)
        for name in ("environment.py", "evaluator.py", "protocol.py")}}
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()
