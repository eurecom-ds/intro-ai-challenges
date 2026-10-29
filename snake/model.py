"""CPU episodic policy gradients for the Snake classroom activity.

The actor is updated once after each batch of complete games. A separate
network fits Monte Carlo reward-to-go and supplies the next batch's baseline.
"""

from contextlib import contextmanager
from copy import deepcopy
import importlib
import math
import time

import numpy as np
import torch
from torch import nn

try:
    from . import environment, evaluator, protocol
except ImportError:
    import environment
    import evaluator
    import protocol

BATCH_EPISODES = 32
BASELINE_LEARNING_RATE = 0.001
MAX_GRAD_NORM = 1.0
VALIDATION_BATCH_INTERVAL = 16
ARCHITECTURE = protocol.ARCHITECTURE
BACKENDS = ("reference", "torch_eager", "torch_compile", "numba")
_TORCH_KERNELS = {}
_NUMBA_KERNELS = {}


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(protocol.OBSERVATION_SIZE, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.layers[-1].weight, 0.01)

    def forward(self, observations):
        return self.layers(observations)


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(protocol.OBSERVATION_SIZE, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, observations):
        return self.layers(observations).squeeze(-1)


def export_policy(actor):
    """Freeze the final actor as the numeric data read by the evaluator."""
    return {"architecture": ARCHITECTURE, "dtype": "float32", "layers": [
        {"weight": layer.weight.detach().cpu().numpy().astype(np.float32).tolist(),
         "bias": layer.bias.detach().cpu().numpy().astype(np.float32).tolist()}
        for layer in actor.layers if isinstance(layer, nn.Linear)]}


@contextmanager
def _cpu_randomness(seed):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def initial_policy(seed=42):
    """Return the same near-uniform initialization used for every experiment."""
    with _cpu_randomness(seed):
        return export_policy(PolicyNetwork())


def discounted_returns(rewards, gamma):
    result = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def actor_loss(log_probabilities, advantages, discounts, batch_episodes):
    """Average complete-episode sums, retaining their relative lengths."""
    return -(log_probabilities * advantages.detach() * discounts).sum() / batch_episodes


def _sample_actions(logits, generator):
    probabilities = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(-1)
    return actions.numpy(), entropy.numpy()


def collect_episodes(actor, config, environment_rng, action_generator, count=BATCH_EPISODES):
    """Collect complete games with one frozen actor and independent game RNGs."""
    rngs = [np.random.default_rng(int(environment_rng.integers(0, 2**63)))
            for _ in range(count)]
    states = [environment.reset(rng) for rng in rngs]
    active = list(range(count))
    observations = [[] for _ in range(count)]
    actions = [[] for _ in range(count)]
    rewards = [[] for _ in range(count)]
    components = np.zeros((count, 4), dtype=np.float64)
    apples = np.zeros(count, dtype=np.int64)
    reasons = [None] * count
    entropy_sum = 0.0
    with torch.no_grad():
        while active:
            batch = environment.observe_batch([states[index] for index in active])
            chosen, entropies = _sample_actions(actor(torch.from_numpy(batch)), action_generator)
            entropy_sum += float(entropies.sum())
            next_active = []
            for row, index in enumerate(active):
                observations[index].append(batch[row].copy())
                actions[index].append(int(chosen[row]))
                state, events, done, reason = environment.step(states[index], int(chosen[row]), rngs[index])
                states[index] = state
                rewards[index].append(environment.training_reward(events, config))
                contributions = (config["food_reward"] * events["food"],
                                 config["collision_reward"] * events["collision"],
                                 config["move_reward"] * events["move"],
                                 config["distance_reward"] * events["distance"])
                components[index] += contributions
                apples[index] += int(events["food"])
                if done:
                    reasons[index] = reason
                else:
                    next_active.append(index)
            active = next_active
    lengths = np.asarray([len(items) for items in rewards], dtype=np.int64)
    return {"observations": np.concatenate([np.asarray(items, dtype=np.float32) for items in observations]),
            "actions": np.concatenate([np.asarray(items, dtype=np.int64) for items in actions]),
            "returns": np.concatenate([discounted_returns(items, config["gamma"]) for items in rewards]),
            "discounts": np.concatenate([np.power(config["gamma"], np.arange(length), dtype=np.float64)
                                          for length in lengths]).astype(np.float32),
            "lengths": lengths, "mean_return": float(np.mean([sum(items) for items in rewards])),
            "reward_components": dict(zip(("food", "collision", "move", "distance"), components.mean(0).tolist())),
            "mean_apples": float(apples.mean()), "mean_steps": float(lengths.mean()),
            "collision_fraction": float(np.mean([reason in ("wall_collision", "body_collision") for reason in reasons])),
            "entropy": entropy_sum / int(lengths.sum()), "terminal_reasons": reasons}


def _tensor_environment():
    """Load the optional training-only environment when a tensor backend is used."""
    name = (__package__ + ".torch_environment") if __package__ else "torch_environment"
    return importlib.import_module(name)


class _TorchKernel:
    def __init__(self, backend, count):
        self.backend = backend
        self.count = count
        self.compile_seconds = 0.0
        self._function = None if backend == "torch_compile" else _tensor_environment().step

    def __call__(self, state, actions, food_ranks):
        if self._function is None:
            start = time.perf_counter()
            function = torch.compile(_tensor_environment().step, fullgraph=True, dynamic=False)
            result = function(state, actions, food_ranks)
            self.compile_seconds += time.perf_counter() - start
            self._function = function
            return result
        return self._function(state, actions, food_ranks)


def get_torch_kernel(backend="torch_eager", count=BATCH_EPISODES):
    """Return one cached CPU transition kernel per backend and batch shape.

    Compilation is lazy on the first step. ``compile_seconds`` records that
    first compiled call, including compiler setup and its first execution.
    """
    if backend not in ("torch_eager", "torch_compile"):
        raise ValueError("Choose torch_eager or torch_compile for a tensor kernel.")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Use a positive integer number of games.")
    key = (backend, count)
    if key not in _TORCH_KERNELS:
        _TORCH_KERNELS[key] = _TorchKernel(backend, count)
    return _TORCH_KERNELS[key]


def reset_torch_kernel_cache():
    """Discard cached kernel wrappers for an instructor cold-start benchmark."""
    _TORCH_KERNELS.clear()


def _numba_environment():
    """Load the optional NumPy/Numba training environment on demand."""
    name = (__package__ + ".numba_environment") if __package__ else "numba_environment"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == "numba":
            raise RuntimeError("Install the optional numba package before using the numba backend.") from error
        raise


class _NumbaKernel:
    def __init__(self, count):
        self.count = count
        self.compile_seconds = 0.0
        self._environment = _numba_environment()
        self._observed = False
        self._stepped = False

    def observe(self, state):
        if not self._observed:
            start = time.perf_counter()
            result = self._environment.observe(state)
            self.compile_seconds += time.perf_counter() - start
            self._observed = True
            return result
        return self._environment.observe(state)

    def __call__(self, state, actions, food_ranks):
        if not self._stepped:
            start = time.perf_counter()
            result = self._environment.step(state, actions, food_ranks)
            self.compile_seconds += time.perf_counter() - start
            self._stepped = True
            return result
        return self._environment.step(state, actions, food_ranks)


def get_numba_kernel(count=BATCH_EPISODES):
    """Return a cached Numba kernel, measuring its first observe and step calls."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Use a positive integer number of games.")
    if count not in _NUMBA_KERNELS:
        _NUMBA_KERNELS[count] = _NumbaKernel(count)
    return _NUMBA_KERNELS[count]


def reset_numba_kernel_cache():
    """Discard Numba wrappers; a fresh process measures compiler startup."""
    _NUMBA_KERNELS.clear()


def _recorded_batch(config, observations, actions, events, valid, lengths, reasons, entropy_sum):
    """Finish either buffered CPU collector using reference reward precision."""
    raw_events = events.astype(np.float64)
    weights = np.asarray([config[name] for name in ("food_reward", "collision_reward",
                                                   "move_reward", "distance_reward")], dtype=np.float64)
    contributions = raw_events * weights
    rewards = contributions[..., 0] + contributions[..., 1]
    rewards = rewards + contributions[..., 2]
    rewards = rewards + contributions[..., 3]
    component_totals = np.cumsum(contributions, axis=1)[:, -1]
    returns = [discounted_returns(rewards[index, :length], config["gamma"])
               for index, length in enumerate(lengths)]
    reward_totals = np.asarray([sum(float(value) for value in rewards[index, :length])
                                for index, length in enumerate(lengths)], dtype=np.float64)
    apples = raw_events[..., 0].sum(axis=1).astype(np.int64)
    return {"observations": observations[valid], "actions": actions[valid],
            "returns": np.concatenate(returns),
            "discounts": np.concatenate([np.power(config["gamma"], np.arange(length), dtype=np.float64)
                                          for length in lengths]).astype(np.float32),
            "lengths": lengths, "mean_return": float(reward_totals.mean()),
            "reward_components": dict(zip(("food", "collision", "move", "distance"),
                                          component_totals.mean(0).tolist())),
            "mean_apples": float(apples.mean()), "mean_steps": float(lengths.mean()),
            "collision_fraction": float(np.mean([reason in ("wall_collision", "body_collision") for reason in reasons])),
            "entropy": entropy_sum / int(lengths.sum()), "terminal_reasons": reasons}


def collect_episodes_torch(actor, config, environment_rng, action_generator,
                           count=BATCH_EPISODES, *, backend="torch_eager", kernel=None):
    """Collect the reference trajectories using fixed-shape CPU game tensors.

    Only active rows enter the actor and action sampler. Their order matches
    the reference collector, preserving its action-generator stream. Game
    physics advances all rows with completed rows masked by the kernel.
    """
    tensor_environment = _tensor_environment()
    if kernel is None:
        kernel = get_torch_kernel(backend, count)
    rngs = [np.random.default_rng(int(environment_rng.integers(0, 2**63)))
            for _ in range(count)]
    state, food_ranks = tensor_environment.initialize(rngs)
    observation = tensor_environment.observe(state)
    horizon = protocol.MAX_STEPS
    observations = torch.empty((count, horizon, protocol.OBSERVATION_SIZE), dtype=torch.float32)
    actions = torch.empty((count, horizon), dtype=torch.int64)
    events = torch.zeros((count, horizon, 4), dtype=torch.float32)
    valid = torch.zeros((count, horizon), dtype=torch.bool)
    full_actions = torch.zeros(count, dtype=torch.int64)
    entropy_sum = 0.0
    with torch.no_grad():
        for move in range(horizon):
            active = torch.nonzero(state.reason == 0, as_tuple=False).flatten()
            if active.numel() == 0:
                break
            batch = observation[active]
            chosen, action_entropies = _sample_actions(actor(batch), action_generator)
            # Sum each step's active rows in the same float32 order as NumPy.
            entropy_sum += float(action_entropies.sum())
            observations[active, move] = batch
            actions[active, move] = torch.from_numpy(chosen)
            valid[active, move] = True
            full_actions[active] = torch.from_numpy(chosen)
            state, move_events, observation = kernel(state, full_actions, food_ranks)
            events[:, move] = move_events
    reasons = [tensor_environment.REASONS[int(code)] for code in state.reason]
    return _recorded_batch(config, observations.numpy(), actions.numpy(), events.numpy(),
                           valid.numpy(), state.steps.numpy().copy(), reasons, entropy_sum)


def collect_episodes_numba(actor, config, environment_rng, action_generator,
                           count=BATCH_EPISODES, *, kernel=None):
    """Collect canonical CPU trajectories with a NumPy/Numba game kernel."""
    numba_environment = _numba_environment()
    if kernel is None:
        kernel = get_numba_kernel(count)
    rngs = [np.random.default_rng(int(environment_rng.integers(0, 2**63)))
            for _ in range(count)]
    state, food_ranks = numba_environment.initialize(rngs)
    observation = kernel.observe(state)
    horizon = protocol.MAX_STEPS
    observations = np.empty((count, horizon, protocol.OBSERVATION_SIZE), dtype=np.float32)
    actions = np.empty((count, horizon), dtype=np.int64)
    events = np.zeros((count, horizon, 4), dtype=np.float32)
    valid = np.zeros((count, horizon), dtype=bool)
    full_actions = np.zeros(count, dtype=np.int64)
    entropy_sum = 0.0
    with torch.no_grad():
        for move in range(horizon):
            active = np.flatnonzero(state.reason == 0)
            if active.size == 0:
                break
            batch = observation[active]
            chosen, action_entropies = _sample_actions(actor(torch.from_numpy(batch)), action_generator)
            entropy_sum += float(action_entropies.sum())
            observations[active, move] = batch
            actions[active, move] = chosen
            valid[active, move] = True
            full_actions[active] = chosen
            state, move_events, observation = kernel(state, full_actions, food_ranks)
            events[:, move] = move_events
    reasons = [numba_environment.REASONS[int(code)] for code in state.reason]
    return _recorded_batch(config, observations, actions, events, valid,
                           state.steps.copy(), reasons, entropy_sum)


def _check_finite(value, message):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(message)


def train_policy(config=None, progress=None, seed=None, evaluation_seed_pairs=None, backend="reference"):
    """Train from a fresh initialization and retain the final CPU actor.

    ``seed`` is an instructor-only override. Progress receives immutable numeric
    snapshots after initial validation and every completed training batch.
    """
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise ValueError(f"Choose backend from {list(BACKENDS)}.")
    config = protocol.settings(**(config or {}))
    training_seed = config.get("seed", 42) if seed is None else seed
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or not 0 <= training_seed < 2**32:
        raise ValueError("Use an integer instructor seed from 0 to 2**32 - 1.")
    if config["episodes"] % BATCH_EPISODES:
        raise ValueError("Training episodes must contain complete batches of 32 games.")
    evaluation_seed_pairs = (protocol.public_seed_pairs() if evaluation_seed_pairs is None
                             else evaluation_seed_pairs)
    start = time.perf_counter()
    history = []
    total_steps = 0
    kernel = None
    compile_before = 0.0
    with _cpu_randomness(training_seed):
        if backend == "numba":
            kernel = get_numba_kernel(BATCH_EPISODES)
            compile_before = kernel.compile_seconds
        elif backend != "reference":
            kernel = get_torch_kernel(backend, BATCH_EPISODES)
            compile_before = kernel.compile_seconds
        actor = PolicyNetwork()
        torch.manual_seed(training_seed ^ 0x5A17)
        baseline = ValueNetwork()
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"])
        value_optimizer = torch.optim.Adam(baseline.parameters(), lr=BASELINE_LEARNING_RATE)
        environment_rng = np.random.default_rng(np.random.SeedSequence([training_seed, 1]))
        action_generator = torch.Generator().manual_seed(training_seed ^ 0x315A)
        initial = {"batch": 0, "episodes": 0, "steps": 0,
                   "validation": evaluator.evaluate(export_policy(actor), evaluation_seed_pairs),
                   "seconds": time.perf_counter() - start}
        history.append(initial)
        if progress is not None:
            progress(deepcopy(initial))
        batch_count = config["episodes"] // BATCH_EPISODES
        for batch_number in range(1, batch_count + 1):
            if backend == "reference":
                batch = collect_episodes(actor, config, environment_rng, action_generator)
            elif backend == "numba":
                batch = collect_episodes_numba(actor, config, environment_rng, action_generator,
                                               kernel=kernel)
            else:
                batch = collect_episodes_torch(actor, config, environment_rng, action_generator,
                                               backend=backend, kernel=kernel)
            x = torch.from_numpy(batch["observations"])
            chosen = torch.from_numpy(batch["actions"])
            returns = torch.from_numpy(batch["returns"])
            discounts = torch.from_numpy(batch["discounts"])
            with torch.no_grad():
                advantages = returns - baseline(x)
            log_probabilities = torch.log_softmax(actor(x), dim=-1).gather(1, chosen[:, None]).squeeze(1)
            policy_objective = actor_loss(log_probabilities, advantages, discounts, BATCH_EPISODES)
            _check_finite(policy_objective, "The policy loss became non-finite. Reduce the learning rate or reward magnitudes.")
            actor_optimizer.zero_grad(set_to_none=True)
            policy_objective.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            actor_optimizer.step()
            value_objective = (baseline(x) - returns).square().mean()
            _check_finite(value_objective, "The baseline loss became non-finite. Reduce the reward magnitudes.")
            value_optimizer.zero_grad(set_to_none=True)
            value_objective.backward()
            value_grad_norm = nn.utils.clip_grad_norm_(baseline.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            value_optimizer.step()
            for parameter in (*actor.parameters(), *baseline.parameters()):
                _check_finite(parameter, "Training produced non-finite weights. Reduce the learning rate or reward magnitudes.")
            total_steps += int(batch["lengths"].sum())
            row = {"batch": batch_number, "episodes": batch_number * BATCH_EPISODES,
                   "steps": total_steps, "training_return": batch["mean_return"],
                   "reward_components": batch["reward_components"], "training_apples": batch["mean_apples"],
                   "mean_episode_steps": batch["mean_steps"], "collision_fraction": batch["collision_fraction"],
                   "entropy": batch["entropy"], "policy_loss": float(policy_objective.detach()),
                   "value_loss": float(value_objective.detach()), "actor_grad_norm": float(actor_grad_norm),
                   "value_grad_norm": float(value_grad_norm), "seconds": time.perf_counter() - start}
            if batch_number % VALIDATION_BATCH_INTERVAL == 0 or batch_number == batch_count:
                row["validation"] = evaluator.evaluate(export_policy(actor), evaluation_seed_pairs)
            history.append(row)
            if progress is not None:
                progress(deepcopy(row))
        final_policy = export_policy(actor)
        evaluator.validate_policy(final_policy)
    return {"config": config, "policy": final_policy, "history": history,
            "training_seed": training_seed, "training_episodes": config["episodes"],
            "training_steps": total_steps, "training_seconds": time.perf_counter() - start,
            "backend": backend, "compile_seconds": (kernel.compile_seconds - compile_before) if kernel else 0.0,
            "validation": history[-1]["validation"]}
