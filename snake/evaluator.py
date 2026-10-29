"""Canonical CPU policy inference and food-first Snake evaluation."""
from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from . import environment, protocol
except ImportError:
    import environment
    import protocol


def validate_policy(value):
    """Validate fixed actor arrays and return float32-quantized float64 arrays."""
    if not isinstance(value, dict) or set(value) != {"architecture", "dtype", "layers"}:
        raise ValueError("policy must contain architecture, dtype and layers.")
    if value["architecture"] != protocol.ARCHITECTURE or value["dtype"] != "float32":
        raise ValueError("Use the fixed float32 Snake policy architecture.")
    if not isinstance(value["layers"], (list, tuple)) or len(value["layers"]) != 3:
        raise ValueError("The policy must contain exactly three layers.")
    layers = []
    for index, (layer, shape) in enumerate(zip(value["layers"], protocol.LAYER_SHAPES), 1):
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias"}:
            raise ValueError(f"Policy layer {index} must contain weight and bias.")
        result = {}
        for name, expected in (("weight", shape), ("bias", (shape[0],))):
            try:
                raw = np.asarray(layer[name])
                if raw.dtype.kind not in "iuf" or raw.shape != expected:
                    raise ValueError
                # Mixed JSON booleans and numbers otherwise coerce to floats.
                if any(isinstance(item, (bool, np.bool_))
                       for item in np.asarray(layer[name], dtype=object).flat):
                    raise ValueError
                array = np.asarray(raw, dtype=np.float64)
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f"Policy layer {index} {name} must be a numeric array with shape {list(expected)}.") from None
            if not np.isfinite(array).all() or np.any(np.abs(array) > protocol.MAX_PARAMETER):
                raise ValueError(f"Policy layer {index} {name} must be finite with absolute values no greater than {protocol.MAX_PARAMETER:g}.")
            result[name] = array.astype(np.float32).astype(np.float64)
        layers.append(result)
    return {"architecture": protocol.ARCHITECTURE, "dtype": "float32", "layers": layers}


def zero_policy():
    return {"architecture": protocol.ARCHITECTURE, "dtype": "float32", "layers": [
        {"weight": np.zeros(shape, dtype=np.float32).tolist(),
         "bias": np.zeros(shape[0], dtype=np.float32).tolist()} for shape in protocol.LAYER_SHAPES]}


def policy_probabilities(policy, observations, *, validated=False):
    policy = policy if validated else validate_policy(policy)
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != protocol.OBSERVATION_SIZE or not np.isfinite(values).all():
        raise ValueError("Policy observations must be a finite [N,118] matrix.")
    for index, layer in enumerate(policy["layers"]):
        values = values @ layer["weight"].T + layer["bias"]
        if index < 2:
            np.maximum(values, 0.0, out=values)
    values -= values.max(axis=1, keepdims=True)
    np.exp(values, out=values)
    values /= values.sum(axis=1, keepdims=True)
    return values


def sample_actions(probabilities, rngs):
    if len(probabilities) != len(rngs):
        raise ValueError("Provide one independent action generator per game.")
    uniform = np.fromiter((rng.random() for rng in rngs), dtype=np.float64, count=len(rngs))
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return (uniform[:, None] >= cumulative).sum(axis=1).astype(np.int64)


def ranking_key(result):
    return (-int(result["total_apples"]), int(result["sum_food_steps"]))


def evaluate(policy, seed_pairs, *, keep_replay=False):
    """Evaluate sampled actions on fixed independent environment/action seeds.

    A retained replay contains the first listed game only. Hidden evaluators
    should generate their public replay with a separate demonstration seed.
    """
    policy = validate_policy(policy)
    pairs = protocol.validate_seed_pairs(seed_pairs)
    env_rngs = [np.random.Generator(np.random.PCG64(pair[0])) for pair in pairs]
    action_rngs = [np.random.Generator(np.random.PCG64(pair[1])) for pair in pairs]
    states = [environment.reset(rng) for rng in env_rngs]
    active = list(range(len(pairs)))
    apples = np.zeros(len(pairs), dtype=np.int64)
    timestamps = np.zeros(len(pairs), dtype=np.int64)
    frames = [environment.frame(states[0])] if keep_replay else None
    while active:
        observations = environment.observe_batch([states[index] for index in active])
        probabilities = policy_probabilities(policy, observations, validated=True)
        actions = sample_actions(probabilities, [action_rngs[index] for index in active])
        remaining = []
        for index, action in zip(active, actions):
            state, events, done, _ = environment.step(states[index], action, env_rngs[index])
            states[index] = state
            apples[index] += events["food"]
            timestamps[index] += events["food"] * state.step_count
            if keep_replay and index == 0:
                frames.append(environment.frame(state))
            if not done:
                remaining.append(index)
        active = remaining
    total = int(apples.sum())
    time_sum = int(timestamps.sum())
    reasons = Counter(state.reason for state in states)
    count = len(pairs)
    result = {"episodes": count, "total_apples": total, "sum_food_steps": time_sum,
              "mean_apples": total / count, "mean_food_step": time_sum / total if total else None,
              "diagnostics": {"mean_steps": sum(state.step_count for state in states) / count,
                              "collision_rate": (reasons["wall_collision"] + reasons["body_collision"]) / count,
                              "terminal_reasons": dict(reasons)}}
    if keep_replay:
        result["replay"] = {"board_size": protocol.BOARD_SIZE, "max_steps": protocol.MAX_STEPS,
                            "frames": frames, "label": "Demonstration game"}
    return result
