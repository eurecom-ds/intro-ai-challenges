"""Fixed-shape NumPy Snake state with native CPU transition loops.

Public wrappers pass the board size, horizon and observation lookup tables
to native kernels explicitly. Compiled specializations are retained within
the process. Disk caching is disabled because notebooks load helpers under
release-specific module names, while Numba's disk cache retains import names.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

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
_AGES = np.arange(CELLS + 1, dtype=np.float32) / CELLS
_REMAINING = np.asarray([(HORIZON - move) / HORIZON for move in range(HORIZON + 1)], dtype=np.float32)
_OFFSETS = np.asarray([offset / (SIZE - 1) for offset in range(1 - SIZE, SIZE)], dtype=np.float32)
_NEIGHBORS = np.full((CELLS, 4), -1, dtype=np.int32)
for _cell in range(CELLS):
    _row, _column = divmod(_cell, SIZE)
    if _row:
        _NEIGHBORS[_cell, 0] = _cell - SIZE
    if _column + 1 < SIZE:
        _NEIGHBORS[_cell, 1] = _cell + 1
    if _row + 1 < SIZE:
        _NEIGHBORS[_cell, 2] = _cell + SIZE
    if _column:
        _NEIGHBORS[_cell, 3] = _cell - 1


class ArrayState(NamedTuple):
    age: np.ndarray
    head: np.ndarray
    food: np.ndarray
    heading: np.ndarray
    length: np.ndarray
    steps: np.ndarray
    reason: np.ndarray


@njit(cache=False)
def _tail_connected_impl(observations, neighbors, tail_value):
    count = observations.shape[0]
    connected = np.zeros(count, dtype=np.uint8)
    queue = np.empty(CELLS, dtype=np.int32)
    visited = np.zeros(CELLS, dtype=np.int64)
    for index in range(count):
        head = -1
        tail = -1
        for cell in range(CELLS):
            if observations[index, CELLS + cell] == 1:
                head = cell
            if observations[index, cell] == tail_value:
                tail = cell
            if head >= 0 and tail >= 0:
                break
        if head < 0 or tail < 0:
            raise ValueError("Each observation must contain a head and an age-one tail cell.")
        if head == tail:
            connected[index] = 1
            continue
        stamp = index + 1
        visited[head] = stamp
        queue[0] = head
        first = 0
        last = 1
        reached = False
        while first < last and not reached:
            cell = queue[first]
            first += 1
            for direction in range(4):
                following = neighbors[cell, direction]
                if following < 0 or visited[following] == stamp:
                    continue
                visited[following] = stamp
                if following == tail:
                    connected[index] = 1
                    reached = True
                    break
                if observations[index, following] == 0:
                    queue[last] = following
                    last += 1
    return connected


def tail_connected_batch(observations):
    """Find static head-to-tail paths through free cells for each observation.

    The head and current tail are traversable; other body cells are blocked.
    The result is a uint8 flag per observation, with input arrays preserved.
    """
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != protocol.OBSERVATION_SIZE:
        raise ValueError(f"Provide an N-by-{protocol.OBSERVATION_SIZE} Snake observation array.")
    if observations.dtype != np.float32:
        observations = observations.astype(np.float32)
    return _tail_connected_impl(observations, _NEIGHBORS, _AGES[1])


def from_states(states, rngs):
    """Encode reference states and their exact future free-cell rank draws."""
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
    integer = lambda values: np.asarray(values, dtype=np.int64)
    state = ArrayState(age, integer([s.body[-1] for s in states]),
                       integer([-1 if s.food is None else s.food for s in states]),
                       integer([s.heading for s in states]), integer([len(s.body) for s in states]),
                       integer([s.step_count for s in states]), integer([REASONS.index(s.reason) for s in states]))
    return state, ranks


def initialize(rngs):
    return from_states([environment.reset(rng) for rng in rngs], rngs)


@njit(cache=False, inline="always")
def _destination(age, head, food, heading, action, size):
    if action == 1:
        heading = (heading - 1) % 4
    elif action == 2:
        heading = (heading + 1) % 4
    row, col = head // size, head % size
    if heading == 0:
        row -= 1
    elif heading == 1:
        col += 1
    elif heading == 2:
        row += 1
    else:
        col -= 1
    if row < 0 or row >= size or col < 0 or col >= size:
        return heading, -1, 1, row, col
    cell = row * size + col
    body_collision = age[cell] > 1 or (age[cell] > 0 and cell == food)
    return heading, cell, 2 if body_collision else 0, row, col


@njit(cache=False, nogil=True)
def _observe_impl(state, size, horizon, age_values, time_values, offset_values):
    age, heads, foods, headings, lengths, moves, reasons = state
    batch, cells = age.shape
    auxiliary = 3 * cells
    observations = np.zeros((batch, auxiliary + 10), dtype=np.float32)
    for row in range(batch):
        for cell in range(cells):
            observations[row, cell] = age_values[age[row, cell]]
        head, food, heading = heads[row], foods[row], headings[row]
        observations[row, cells + head] = 1
        observations[row, auxiliary + heading] = 1
        observations[row, auxiliary + 4] = time_values[min(max(moves[row], 0), horizon)]
        if food >= 0:
            observations[row, 2 * cells + food] = 1
            observations[row, auxiliary + 5] = offset_values[food // size - head // size + size - 1]
            observations[row, auxiliary + 6] = offset_values[food % size - head % size + size - 1]
        for action in range(3):
            _, _, collision, _, _ = _destination(age[row], head, food, heading, action, size)
            observations[row, auxiliary + 7 + action] = collision != 0
    return observations


@njit(cache=False, nogil=True)
def _step_impl(state, actions, food_ranks, size, horizon, age_values, time_values, offset_values):
    old_age, old_heads, old_foods, old_headings, old_lengths, old_steps, old_reasons = state
    age = old_age.copy()
    head, food, heading = old_heads.copy(), old_foods.copy(), old_headings.copy()
    length, steps, reason = old_lengths.copy(), old_steps.copy(), old_reasons.copy()
    batch, cells = age.shape
    events = np.zeros((batch, 4), dtype=np.float32)
    for row in range(batch):
        if reason[row] != 0:
            continue
        old_head, old_food = head[row], food[row]
        next_heading, cell, collision, next_row, next_col = _destination(
            old_age[row], old_head, old_food, heading[row], actions[row], size)
        heading[row] = next_heading
        steps[row] += 1
        events[row, 2] = 1
        if collision:
            reason[row] = collision
            events[row, 1] = 1
            continue
        events[row, 3] = (abs(old_head // size - old_food // size)
                          + abs(old_head % size - old_food % size)
                          - abs(next_row - old_food // size) - abs(next_col - old_food % size))
        eaten = cell == old_food
        if eaten:
            length[row] += 1
            events[row, 0] = 1
        else:
            for position in range(cells):
                if age[row, position] > 0:
                    age[row, position] -= 1
        age[row, cell] = length[row]
        head[row] = cell
        if eaten:
            if length[row] == cells:
                food[row] = -1
            else:
                target = food_ranks[row, length[row] - 1]
                seen = 0
                for position in range(cells):
                    if age[row, position] == 0:
                        if seen == target:
                            food[row] = position
                            break
                        seen += 1
        if length[row] == cells:
            reason[row] = 3
        elif steps[row] >= horizon:
            reason[row] = 4
    result = (age, head, food, heading, length, steps, reason)
    observations = _observe_impl(result, size, horizon, age_values, time_values, offset_values)
    return result, events, observations


def observe(state):
    """Return bitwise-reference float32 observations through a native loop."""
    return _observe_impl(tuple(state), SIZE, HORIZON, _AGES, _REMAINING, _OFFSETS)


def step(state, actions, food_ranks):
    """Return new state, float32 [food,collision,move,distance] events and obs."""
    result, events, observations = _step_impl(
        tuple(state), actions, food_ranks, SIZE, HORIZON, _AGES, _REMAINING, _OFFSETS)
    return ArrayState(*result), events, observations
