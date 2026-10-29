"""Fixed-shape CPU tensor Snake transitions and canonical observations.

Initialization uses each game's NumPy generator outside the compiled kernel.
``step`` and ``observe`` contain tensor operations only. Finished rows retain
their state and emit zero reward events.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

try:
    from . import environment, protocol
except ImportError:
    import environment
    import protocol

SIZE = protocol.BOARD_SIZE
CELLS = protocol.BOARD_CELLS
HORIZON = protocol.MAX_STEPS
REASONS = (None, "wall_collision", "body_collision", "board_complete", "move_limit")
EVENT_NAMES = ("food", "collision", "move", "distance")

_POSITIONS = torch.arange(CELLS, dtype=torch.int64)[None, :]
_HEADINGS = torch.arange(4, dtype=torch.int64)[None, :]
_TURNS = torch.tensor((0, -1, 1), dtype=torch.int64)
_DIRECTIONS = torch.tensor(environment.DIRECTIONS, dtype=torch.int64)
# Lookups preserve the reference's float32 rounding under kernel fusion.
_AGES = torch.from_numpy(np.arange(CELLS + 1, dtype=np.float32) / CELLS)
_REMAINING = torch.from_numpy(np.asarray(
    [(HORIZON - move) / HORIZON for move in range(HORIZON + 1)], dtype=np.float32))
_OFFSETS = torch.from_numpy(np.asarray(
    [offset / (SIZE - 1) for offset in range(1 - SIZE, SIZE)], dtype=np.float32))


class TensorState(NamedTuple):
    age: torch.Tensor
    head: torch.Tensor
    food: torch.Tensor
    heading: torch.Tensor
    length: torch.Tensor
    steps: torch.Tensor
    reason: torch.Tensor


def from_states(states, rngs):
    """Encode reference states and precompute their remaining food draws.

    ``rngs`` must have the states they would have immediately before the next
    reference move. Food ranks have shape [B,225], indexed by new length minus
    one. A growth to length four uses column three, whose bound is 221.
    Prefetched unused draws stay within each game's discarded generator.
    """
    if not states or len(states) != len(rngs):
        raise ValueError("Provide one independent food generator per game.")
    age = np.zeros((len(states), CELLS), dtype=np.int64)
    ranks = np.zeros_like(age)
    for row, (state, rng) in enumerate(zip(states, rngs)):
        if (not state.body or len(set(state.body)) != len(state.body)
                or any(not 0 <= cell < CELLS for cell in state.body)
                or not 0 <= state.heading < 4 or not 0 <= state.step_count <= HORIZON
                or state.reason not in REASONS
                or (state.food is not None and not 0 <= state.food < CELLS)):
            raise ValueError("Use a valid reference Snake state.")
        age[row, list(state.body)] = np.arange(1, len(state.body) + 1)
        if state.reason is None:
            bounds = np.arange(CELLS - len(state.body) - 1, 0, -1, dtype=np.int64)
            if bounds.size:
                ranks[row, len(state.body):CELLS - 1] = rng.integers(0, bounds, dtype=np.int64)
    long = lambda values: torch.tensor(values, dtype=torch.int64)
    state = TensorState(torch.from_numpy(age), long([s.body[-1] for s in states]),
                        long([-1 if s.food is None else s.food for s in states]),
                        long([s.heading for s in states]), long([len(s.body) for s in states]),
                        long([s.step_count for s in states]), long([REASONS.index(s.reason) for s in states]))
    return state, torch.from_numpy(ranks)


def initialize(rngs):
    """Reset each game canonically, then prefetch future free-cell ranks."""
    return from_states([environment.reset(rng) for rng in rngs], rngs)


def _destinations(state, headings):
    directions = _DIRECTIONS[headings]
    row = torch.div(state.head, SIZE, rounding_mode="floor")[:, None] + directions[..., 0]
    col = torch.remainder(state.head, SIZE)[:, None] + directions[..., 1]
    wall = (row < 0) | (row >= SIZE) | (col < 0) | (col >= SIZE)
    positions = (row * SIZE + col).clamp(0, CELLS - 1)
    age = state.age.gather(1, positions)
    growth = positions == state.food[:, None]
    collision = wall | (age > 1) | ((age > 0) & growth)
    return positions, collision, wall, row, col


def observe(state):
    """Return the reference's float32 [B,685] observation without host reads."""
    head_row = torch.div(state.head, SIZE, rounding_mode="floor")
    head_col = torch.remainder(state.head, SIZE)
    safe_food = state.food.clamp_min(0)
    food_row = torch.div(safe_food, SIZE, rounding_mode="floor")
    food_col = torch.remainder(safe_food, SIZE)
    present = state.food >= 0
    row_offset = torch.where(present, _OFFSETS[food_row - head_row + SIZE - 1], 0)
    col_offset = torch.where(present, _OFFSETS[food_col - head_col + SIZE - 1], 0)
    headings = torch.remainder(state.heading[:, None] + _TURNS[None, :], 4)
    _, collision, _, _, _ = _destinations(state, headings)
    return torch.cat((_AGES[state.age], (_POSITIONS == state.head[:, None]).float(),
                      (_POSITIONS == state.food[:, None]).float(),
                      (_HEADINGS == state.heading[:, None]).float(),
                      _REMAINING[state.steps.clamp(0, HORIZON)][:, None],
                      row_offset[:, None], col_offset[:, None], collision.float()), dim=1)


def step(state, actions, food_ranks):
    """Return (next state, float32 events [food,collision,move,distance], obs).

    Actions are int64 [B] values 0, 1 or 2. All state and rank tensors remain
    on CPU. The fixed shapes support ``torch.compile(..., fullgraph=True)``.
    """
    alive = state.reason == 0
    headings = torch.remainder(state.heading + _TURNS[actions], 4)
    positions, hit, wall, row, col = _destinations(state, headings[:, None])
    destination, hit, wall = positions[:, 0], hit[:, 0], wall[:, 0]
    legal = alive & ~hit
    eaten = legal & (destination == state.food)
    length = state.length + eaten.long()
    steps = state.steps + alive.long()
    shifted = torch.where(eaten[:, None], state.age, (state.age - 1).clamp_min(0))
    proposed = shifted.scatter(1, destination[:, None], length[:, None])
    age = torch.where(legal[:, None], proposed, state.age)
    head = torch.where(legal, destination, state.head)
    heading = torch.where(alive, headings, state.heading)

    free = age == 0
    rank = food_ranks.gather(1, (length - 1).clamp(0, CELLS - 1)[:, None])[:, 0]
    selected = (free & (free.long().cumsum(dim=1) == rank[:, None] + 1)).long().argmax(dim=1)
    spawned = torch.where(length < CELLS, selected, -1)
    food = torch.where(eaten, spawned, state.food)

    reason = torch.where(wall, 1, torch.where(hit, 2, torch.where(
        length == CELLS, 3, torch.where(steps >= HORIZON, 4, 0))))
    reason = torch.where(alive, reason, state.reason)
    next_state = TensorState(age, head, food, heading, length, steps, reason)

    safe_food = state.food.clamp_min(0)
    food_row = torch.div(safe_food, SIZE, rounding_mode="floor")
    food_col = torch.remainder(safe_food, SIZE)
    old_row = torch.div(state.head, SIZE, rounding_mode="floor")
    old_col = torch.remainder(state.head, SIZE)
    progress = ((old_row - food_row).abs() + (old_col - food_col).abs()
                - (row[:, 0] - food_row).abs() - (col[:, 0] - food_col).abs())
    progress = torch.where(legal, progress, 0)
    events = torch.stack((eaten, alive & hit, alive, progress), dim=1).float()
    return next_state, events, observe(next_state)
