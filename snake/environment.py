"""The 15 × 15 Snake game and its full numeric observation."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from . import protocol
except ImportError:
    import protocol

BOARD_SIZE = protocol.BOARD_SIZE
BOARD_CELLS = protocol.BOARD_CELLS
MAX_STEPS = protocol.MAX_STEPS
OBSERVATION_SIZE = protocol.OBSERVATION_SIZE
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TURNS = (0, -1, 1)


class State(NamedTuple):
    body: tuple[int, ...]
    heading: int
    food: int | None
    step_count: int = 0
    reason: str | None = None


def _placements():
    result = []
    for heading, (dr, dc) in enumerate(DIRECTIONS):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cells = [(row - j * dr, col - j * dc) for j in reversed(range(protocol.INITIAL_LENGTH))]
                if all(0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE for r, c in cells):
                    result.append((tuple(r * BOARD_SIZE + c for r, c in cells), heading))
    return tuple(result)


INITIAL_PLACEMENTS = _placements()


def _food(body, rng):
    occupied = set(body)
    free = [cell for cell in range(BOARD_CELLS) if cell not in occupied]
    return int(free[int(rng.integers(len(free)))]) if free else None


def reset(rng):
    body, heading = INITIAL_PLACEMENTS[int(rng.integers(len(INITIAL_PLACEMENTS)))]
    return State(body, heading, _food(body, rng))


def _destination(state, action):
    heading = (state.heading + TURNS[action]) % 4
    dr, dc = DIRECTIONS[heading]
    row, col = divmod(state.body[-1], BOARD_SIZE)
    row, col = row + dr, col + dc
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return heading, None, "wall_collision"
    cell = row * BOARD_SIZE + col
    # The tail leaves on a non-growing move, so its old cell is available.
    occupied = state.body if cell == state.food else state.body[1:]
    return heading, cell, "body_collision" if cell in occupied else None


def step(state, action, rng):
    """Return (next state, reward events, done, reason) for one attempted move."""
    if state.reason is not None or state.step_count >= MAX_STEPS:
        raise ValueError("Reset a completed game before taking another action.")
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < 3:
        raise ValueError("Choose action 0 (straight), 1 (left) or 2 (right).")
    heading, cell, reason = _destination(state, int(action))
    moves = state.step_count + 1
    if reason:
        events = {"food": 0, "collision": 1, "move": 1, "distance": 0}
        return State(state.body, heading, state.food, moves, reason), events, True, reason
    old_row, old_col = divmod(state.body[-1], BOARD_SIZE)
    row, col = divmod(cell, BOARD_SIZE)
    food_row, food_col = divmod(state.food, BOARD_SIZE)
    progress = (abs(old_row - food_row) + abs(old_col - food_col)
                - abs(row - food_row) - abs(col - food_col))
    eaten = int(cell == state.food)
    body = state.body + (cell,) if eaten else state.body[1:] + (cell,)
    food = _food(body, rng) if eaten else state.food
    if len(body) == BOARD_CELLS:
        reason = "board_complete"
    elif moves >= MAX_STEPS:
        reason = "move_limit"
    events = {"food": eaten, "collision": 0, "move": 1, "distance": int(progress)}
    return State(body, heading, food, moves, reason), events, reason is not None, reason


def training_reward(events, settings):
    return (settings["food_reward"] * events["food"]
            + settings["collision_reward"] * events["collision"]
            + settings["move_reward"] * events["move"]
            + settings["distance_reward"] * events["distance"])


def observe_batch(states):
    """Encode the documented planes and features in row-major order."""
    result = np.zeros((len(states), OBSERVATION_SIZE), dtype=np.float32)
    auxiliary = 3 * BOARD_CELLS
    for index, state in enumerate(states):
        result[index, list(state.body)] = np.arange(1, len(state.body) + 1, dtype=np.float32) / BOARD_CELLS
        result[index, BOARD_CELLS + state.body[-1]] = 1
        if state.food is not None:
            result[index, 2 * BOARD_CELLS + state.food] = 1
        result[index, auxiliary + state.heading] = 1
        result[index, auxiliary + 4] = max(0, MAX_STEPS - state.step_count) / MAX_STEPS
        if state.food is not None:
            head_row, head_col = divmod(state.body[-1], BOARD_SIZE)
            food_row, food_col = divmod(state.food, BOARD_SIZE)
            result[index, auxiliary + 5] = (food_row - head_row) / (BOARD_SIZE - 1)
            result[index, auxiliary + 6] = (food_col - head_col) / (BOARD_SIZE - 1)
        for action in range(3):
            result[index, auxiliary + 7 + action] = int(_destination(state, action)[2] is not None)
    return result


def observe(state):
    return observe_batch([state])[0]


def frame(state, *, apples=None):
    """A compact, public replay frame; body cells run from tail to head."""
    return {"body": list(state.body), "heading": state.heading, "food": state.food,
            "step": state.step_count, "apples": len(state.body) - protocol.INITIAL_LENGTH if apples is None else apples,
            "reason": state.reason}
