"""Compare the optional CPU tensor game and collector with canonical Snake."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from snake import environment, model, protocol, torch_environment as tensor_environment


def snake_path():
    return tuple(row * protocol.BOARD_SIZE + col
                 for row in range(protocol.BOARD_SIZE)
                 for col in (range(protocol.BOARD_SIZE) if row % 2 == 0
                             else range(protocol.BOARD_SIZE - 1, -1, -1)))


def reference_state(tensors, index):
    age = tensors.age[index].numpy()
    cells = np.flatnonzero(age)
    body = tuple(int(cell) for cell in cells[np.argsort(age[cells])])
    food = int(tensors.food[index])
    return environment.State(body, int(tensors.heading[index]), None if food < 0 else food,
                             int(tensors.steps[index]), tensor_environment.REASONS[int(tensors.reason[index])])


class TensorGameTests(unittest.TestCase):
    def assert_observations_equal(self, state, expected):
        actual = tensor_environment.observe(state).numpy()
        np.testing.assert_array_equal(actual, environment.observe_batch(expected))
        self.assertEqual(actual.dtype, np.float32)

    def test_seeded_initialization_and_food_rank_prefix(self):
        seeds = tuple(range(100))
        reference_rngs = [np.random.default_rng(seed) for seed in seeds]
        tensor_rngs = [np.random.default_rng(seed) for seed in seeds]
        expected = [environment.reset(rng) for rng in reference_rngs]
        actual, ranks = tensor_environment.initialize(tensor_rngs)
        self.assert_observations_equal(actual, expected)
        for index, (state, rng) in enumerate(zip(expected, reference_rngs)):
            self.assertEqual(reference_state(actual, index), state)
            for length in range(len(state.body) + 1, protocol.BOARD_CELLS):
                self.assertEqual(int(ranks[index, length - 1]), int(rng.integers(protocol.BOARD_CELLS - length)))
            self.assertEqual(rng.bit_generator.state, tensor_rngs[index].bit_generator.state)
        self.assertTrue(all(field.device.type == "cpu" for field in actual))

    def test_prefetch_with_no_future_spawn_leaves_rng_unchanged(self):
        path = snake_path()
        states = [environment.State(path[:-1], 1, path[-1]),
                  environment.State(path[:3], 1, path[3], protocol.MAX_STEPS, "move_limit")]
        rngs = [np.random.default_rng(318), np.random.default_rng(819)]
        before = [deepcopy(rng.bit_generator.state) for rng in rngs]
        _, ranks = tensor_environment.from_states(states, rngs)
        self.assertTrue(torch.equal(ranks, torch.zeros_like(ranks)))
        self.assertEqual(before, [rng.bit_generator.state for rng in rngs])

    def test_transitions_growth_tail_collisions_full_board_and_last_move(self):
        side = protocol.BOARD_SIZE
        path = snake_path()
        fixtures = [
            (environment.State((0, 1, 2), 0, 2 * side), 0),
            (environment.State((0, 1, 2, side + 2, side + 3, 2 * side + 3,
                                2 * side + 2, 2 * side + 1, side + 1), 0, protocol.BOARD_CELLS - 1), 0),
            (environment.State((0, 1, side + 1, side), 3, protocol.BOARD_CELLS - 1), 2),
            (environment.State((side, side + 1, side + 2), 1, side + 3), 0),
            (environment.State((2 * side, 2 * side + 1, 2 * side + 2), 1, protocol.BOARD_CELLS - 1), 1),
            (environment.State((side, side + 1, side + 2), 1, side + 3, protocol.MAX_STEPS - 1), 0),
            (environment.State(path[:-1], 1, path[-1]), 0),
            (environment.State((side, side + 1, side + 2), 1, side + 3,
                               protocol.MAX_STEPS, "move_limit"), 2),
            (environment.State(path, 1, None, 222, "board_complete"), 0),
        ]
        states = [state for state, _ in fixtures]
        rngs = [np.random.default_rng(1400 + index) for index in range(len(states))]
        actual, ranks = tensor_environment.from_states(states, [deepcopy(rng) for rng in rngs])
        self.assert_observations_equal(actual, states)
        next_state, events, observation = tensor_environment.step(
            actual, torch.tensor([action for _, action in fixtures]), ranks)
        expected_states = []
        for index, ((state, action), rng) in enumerate(zip(fixtures, rngs)):
            if state.reason is None:
                expected, event, _, _ = environment.step(state, action, rng)
                expected_events = [event[name] for name in tensor_environment.EVENT_NAMES]
            else:
                expected, expected_events = state, [0, 0, 0, 0]
            self.assertEqual(reference_state(next_state, index), expected)
            np.testing.assert_array_equal(events[index].numpy(), expected_events)
            expected_states.append(expected)
        np.testing.assert_array_equal(observation.numpy(), environment.observe_batch(expected_states))
        frozen, frozen_events, _ = tensor_environment.step(next_state, torch.zeros(len(states), dtype=torch.int64), ranks)
        for index, expected in enumerate(expected_states):
            if expected.reason is not None:
                self.assertEqual(reference_state(frozen, index), expected)
                np.testing.assert_array_equal(frozen_events[index].numpy(), [0, 0, 0, 0])

    def test_every_future_food_draw_matches_the_canonical_free_cell_order(self):
        path = snake_path()
        state = environment.State(path[:3], 1, path[3])
        rng = np.random.default_rng(816)
        actual, ranks = tensor_environment.from_states([state], [deepcopy(rng)])
        for next_cell in path[3:]:
            state = state._replace(food=next_cell)
            actual = actual._replace(food=torch.tensor([next_cell], dtype=torch.int64))
            row, col = divmod(state.body[-1], protocol.BOARD_SIZE)
            next_row, next_col = divmod(next_cell, protocol.BOARD_SIZE)
            heading = environment.DIRECTIONS.index((next_row - row, next_col - col))
            action = {0: 0, 3: 1, 1: 2}[(heading - state.heading) % 4]
            state, event, _, _ = environment.step(state, action, rng)
            actual, events, observation = tensor_environment.step(actual, torch.tensor([action]), ranks)
            self.assertEqual(reference_state(actual, 0), state)
            np.testing.assert_array_equal(events[0].numpy(), [event[name] for name in tensor_environment.EVENT_NAMES])
            np.testing.assert_array_equal(observation[0].numpy(), environment.observe(state))
        self.assertEqual(state.reason, "board_complete")
        self.assertIsNone(state.food)


class TensorCollectorTests(unittest.TestCase):
    def tearDown(self):
        model.reset_torch_kernel_cache()

    def test_collector_preserves_actions_observations_returns_statistics_and_rngs(self):
        config = protocol.settings(episodes=256, distance_reward=1)
        reference_rng = np.random.default_rng(801)
        tensor_rng = np.random.default_rng(801)
        reference_actions = torch.Generator().manual_seed(802)
        tensor_actions = torch.Generator().manual_seed(802)
        with model._cpu_randomness(42):
            actor = model.PolicyNetwork()
            expected = model.collect_episodes(actor, config, reference_rng, reference_actions)
            actual = model.collect_episodes_torch(actor, config, tensor_rng, tensor_actions)
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            with self.subTest(field=name):
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(actual[name], value)
                    self.assertEqual(actual[name].dtype, value.dtype)
                else:
                    self.assertEqual(actual[name], value)
        self.assertEqual(reference_rng.bit_generator.state, tensor_rng.bit_generator.state)
        self.assertTrue(torch.equal(reference_actions.get_state(), tensor_actions.get_state()))

    def test_tensor_training_preserves_the_final_reference_actor(self):
        config = protocol.settings(episodes=256, distance_reward=1)
        pairs = protocol.public_seed_pairs()[:2]
        expected = model.train_policy(config, evaluation_seed_pairs=pairs)
        actual = model.train_policy(config, evaluation_seed_pairs=pairs, backend="torch_eager")
        self.assertEqual(expected["policy"], actual["policy"])
        self.assertEqual(expected["training_steps"], actual["training_steps"])
        self.assertEqual(expected["validation"], actual["validation"])
        self.assertEqual(actual["config"], config)
        self.assertNotIn("backend", actual["config"])
        self.assertEqual(actual["backend"], "torch_eager")
        self.assertEqual(actual["compile_seconds"], 0)

    def test_zero_reward_tensor_training_preserves_the_initial_actor(self):
        config = protocol.settings(episodes=256, food_reward=0, collision_reward=0,
                                   move_reward=0, distance_reward=0)
        result = model.train_policy(config, evaluation_seed_pairs=protocol.public_seed_pairs()[:1],
                                    backend="torch_eager")
        self.assertEqual(result["policy"], model.initial_policy())
        self.assertTrue(all(row["policy_loss"] == 0 for row in result["history"][1:]))

    def test_compiled_kernel_is_lazy_and_cached_without_cold_compilation_in_unit_tests(self):
        # Exercise cache behaviour using the eager function as the compiler stand-in.
        rngs = [np.random.default_rng(4)]
        state, ranks = tensor_environment.initialize(rngs)
        with patch.object(torch, "compile", return_value=tensor_environment.step) as compiler:
            first = model.get_torch_kernel("torch_compile", 1)
            self.assertIs(first, model.get_torch_kernel("torch_compile", 1))
            compiler.assert_not_called()
            state, _, _ = first(state, torch.zeros(1, dtype=torch.int64), ranks)
            first(state, torch.zeros(1, dtype=torch.int64), ranks)
            compiler.assert_called_once_with(tensor_environment.step, fullgraph=True, dynamic=False)
        self.assertGreater(first.compile_seconds, 0)
        self.assertIsNot(first, model.get_torch_kernel("torch_compile", 2))

    def test_invalid_backend_fails_before_collecting_games(self):
        with patch.object(model, "collect_episodes"), patch.object(model, "collect_episodes_torch") as collector:
            with self.assertRaisesRegex(ValueError, "backend"):
                model.train_policy(backend="cuda")
            collector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
