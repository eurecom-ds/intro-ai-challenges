"""Validate the optional NumPy/Numba simulator and retained-policy training."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from snake import environment, model, protocol

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
if NUMBA_AVAILABLE:
    from snake import numba_environment
else:
    numba_environment = None


def snake_path():
    return tuple(row * protocol.BOARD_SIZE + col
                 for row in range(protocol.BOARD_SIZE)
                 for col in (range(protocol.BOARD_SIZE) if row % 2 == 0
                             else range(protocol.BOARD_SIZE - 1, -1, -1)))


def reference_state(arrays, index):
    age = arrays.age[index]
    cells = np.flatnonzero(age)
    body = tuple(int(cell) for cell in cells[np.argsort(age[cells])])
    food = int(arrays.food[index])
    return environment.State(body, int(arrays.heading[index]), None if food < 0 else food,
                             int(arrays.steps[index]), numba_environment.REASONS[int(arrays.reason[index])])


@unittest.skipUnless(NUMBA_AVAILABLE, "The optional numba package is not installed.")
class NumbaGameTests(unittest.TestCase):
    def test_import_aliases_work_across_fresh_processes(self):
        # Notebook helpers use a release-scoped module name. Reusing native
        # disk caches from another import name must leave both imports usable.
        source = """import importlib.util, json, sys, numpy as np
from pathlib import Path
path=Path(sys.argv[1])
sys.path.insert(0,str(path.parent))
name=sys.argv[2]
spec=importlib.util.spec_from_file_location(name,path)
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
state,ranks=module.initialize([np.random.default_rng(20)])
module.observe(state)
state,events,observation=module.step(state,np.zeros(1,dtype=np.int64),ranks)
print(json.dumps([observation.tolist(),events.tolist(),state.reason.tolist()]))
"""
        with tempfile.TemporaryDirectory(prefix="snake-numba-alias-") as directory:
            cache_environment = {**os.environ, "NUMBA_CACHE_DIR": directory}
            results = []
            for name in ("snake_first_alias", "snake_second_alias"):
                completed = subprocess.run([sys.executable, "-c", source,
                                            str(Path(numba_environment.__file__)), name],
                                           env=cache_environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                results.append(completed.stdout)
            self.assertEqual(results[0], results[1])

    def test_seeded_initialization_observations_and_rng_prefix(self):
        seeds = range(100)
        reference_rngs = [np.random.default_rng(seed) for seed in seeds]
        actual_rngs = [np.random.default_rng(seed) for seed in seeds]
        expected = [environment.reset(rng) for rng in reference_rngs]
        actual, ranks = numba_environment.initialize(actual_rngs)
        np.testing.assert_array_equal(numba_environment.observe(actual), environment.observe_batch(expected))
        for index, (state, rng) in enumerate(zip(expected, reference_rngs)):
            self.assertEqual(reference_state(actual, index), state)
            for length in range(len(state.body) + 1, protocol.BOARD_CELLS):
                self.assertEqual(int(ranks[index, length - 1]), int(rng.integers(protocol.BOARD_CELLS - length)))
            self.assertEqual(rng.bit_generator.state, actual_rngs[index].bit_generator.state)
        self.assertTrue(all(field.dtype == np.int64 for field in actual))

    def test_fixed_transitions_and_completed_rows_match_without_mutating_inputs(self):
        side, path = protocol.BOARD_SIZE, snake_path()
        fixtures = [
            (environment.State((0, 1, 2), 0, 2 * side), 0),
            (environment.State((0, 1, 2, side + 2, side + 3, 2 * side + 3,
                                2 * side + 2, 2 * side + 1, side + 1), 0, protocol.BOARD_CELLS - 1), 0),
            (environment.State((0, 1, side + 1, side), 3, protocol.BOARD_CELLS - 1), 2),
            (environment.State((side, side + 1, side + 2), 1, side + 3), 0),
            (environment.State((side, side + 1, side + 2), 1, side + 3, protocol.MAX_STEPS - 1), 0),
            (environment.State(path[:-1], 1, path[-1]), 0),
            (environment.State((side, side + 1, side + 2), 1, side + 3,
                               protocol.MAX_STEPS, "move_limit"), 2),
            (environment.State(path, 1, None, 222, "board_complete"), 0),
        ]
        states = [state for state, _ in fixtures]
        rngs = [np.random.default_rng(1800 + index) for index in range(len(states))]
        actual, ranks = numba_environment.from_states(states, [deepcopy(rng) for rng in rngs])
        before = [field.copy() for field in actual]
        next_state, events, observation = numba_environment.step(
            actual, np.asarray([action for _, action in fixtures], dtype=np.int64), ranks)
        expected_states = []
        for index, ((state, action), rng) in enumerate(zip(fixtures, rngs)):
            if state.reason is None:
                expected, event, _, _ = environment.step(state, action, rng)
                expected_events = [event[name] for name in numba_environment.EVENT_NAMES]
            else:
                expected, expected_events = state, [0, 0, 0, 0]
            self.assertEqual(reference_state(next_state, index), expected)
            np.testing.assert_array_equal(events[index], expected_events)
            expected_states.append(expected)
        for old, field in zip(before, actual):
            np.testing.assert_array_equal(old, field)
        np.testing.assert_array_equal(observation, environment.observe_batch(expected_states))
        self.assertEqual(events.dtype, np.float32)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(numba_environment._step_impl.nopython_signatures)
        self.assertTrue(numba_environment._observe_impl.nopython_signatures)

    def test_all_food_draws_and_board_completion_match(self):
        path = snake_path()
        state = environment.State(path[:3], 1, path[3])
        rng = np.random.default_rng(816)
        actual, ranks = numba_environment.from_states([state], [deepcopy(rng)])
        for next_cell in path[3:]:
            state = state._replace(food=next_cell)
            actual = actual._replace(food=np.asarray([next_cell], dtype=np.int64))
            row, col = divmod(state.body[-1], protocol.BOARD_SIZE)
            next_row, next_col = divmod(next_cell, protocol.BOARD_SIZE)
            heading = environment.DIRECTIONS.index((next_row - row, next_col - col))
            action = {0: 0, 3: 1, 1: 2}[(heading - state.heading) % 4]
            state, event, _, _ = environment.step(state, action, rng)
            actual, events, observation = numba_environment.step(actual, np.asarray([action], dtype=np.int64), ranks)
            self.assertEqual(reference_state(actual, 0), state)
            np.testing.assert_array_equal(events[0], [event[name] for name in numba_environment.EVENT_NAMES])
            np.testing.assert_array_equal(observation[0], environment.observe(state))
        self.assertEqual(state.reason, "board_complete")


@unittest.skipUnless(NUMBA_AVAILABLE, "The optional numba package is not installed.")
class NumbaCollectorTests(unittest.TestCase):
    def tearDown(self):
        model.reset_numba_kernel_cache()

    def test_collector_preserves_the_entire_reference_batch_and_random_streams(self):
        config = protocol.settings(episodes=256, distance_reward=1)
        expected_rng, actual_rng = np.random.default_rng(801), np.random.default_rng(801)
        expected_actions = torch.Generator().manual_seed(802)
        actual_actions = torch.Generator().manual_seed(802)
        with model._cpu_randomness(42):
            actor = model.PolicyNetwork()
            expected = model.collect_episodes(actor, config, expected_rng, expected_actions)
            actual = model.collect_episodes_numba(actor, config, actual_rng, actual_actions)
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            with self.subTest(field=name):
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(actual[name], value)
                    self.assertEqual(actual[name].dtype, value.dtype)
                else:
                    self.assertEqual(actual[name], value)
        self.assertEqual(expected_rng.bit_generator.state, actual_rng.bit_generator.state)
        self.assertTrue(torch.equal(expected_actions.get_state(), actual_actions.get_state()))

    def test_training_final_actor_and_zero_rewards_preserve_reference_results(self):
        pairs = protocol.public_seed_pairs()[:2]
        config = protocol.settings(episodes=256, distance_reward=1)
        expected = model.train_policy(config, evaluation_seed_pairs=pairs)
        actual = model.train_policy(config, evaluation_seed_pairs=pairs, backend="numba")
        self.assertEqual(expected["policy"], actual["policy"])
        self.assertEqual(expected["training_steps"], actual["training_steps"])
        self.assertEqual(expected["validation"], actual["validation"])
        self.assertEqual(actual["backend"], "numba")
        self.assertEqual(actual["config"], config)
        self.assertNotIn("backend", actual["config"])
        self.assertGreater(actual["compile_seconds"], 0)
        zero = protocol.settings(episodes=256, food_reward=0, collision_reward=0,
                                 move_reward=0, distance_reward=0)
        result = model.train_policy(zero, evaluation_seed_pairs=pairs, backend="numba")
        self.assertEqual(result["policy"], model.initial_policy())
        self.assertEqual(result["compile_seconds"], 0)

    def test_kernel_cache_reuses_its_observation_and_step_wrappers(self):
        first = model.get_numba_kernel(1)
        self.assertIs(first, model.get_numba_kernel(1))
        self.assertEqual(first.compile_seconds, 0)
        state, ranks = numba_environment.initialize([np.random.default_rng(4)])
        first.observe(state)
        state, _, _ = first(state, np.zeros(1, dtype=np.int64), ranks)
        elapsed = first.compile_seconds
        first.observe(state)
        first(state, np.zeros(1, dtype=np.int64), ranks)
        self.assertEqual(first.compile_seconds, elapsed)
        self.assertGreater(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
