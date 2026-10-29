"""Rule, reward, observation and canonical evaluation checks."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import environment as env
import evaluator
import protocol


class GameRules(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_all_initial_placements(self):
        self.assertEqual(len(env.INITIAL_PLACEMENTS), 96)
        self.assertEqual(len(set(env.INITIAL_PLACEMENTS)), 96)
        for body, heading in env.INITIAL_PLACEMENTS:
            self.assertEqual(len(set(body)), 3)
            dr, dc = env.DIRECTIONS[heading]
            for first, second in zip(body, body[1:]):
                r1, c1 = divmod(first, 6)
                r2, c2 = divmod(second, 6)
                self.assertEqual((r2 - r1, c2 - c1), (dr, dc))
        for _ in range(500):
            state = env.reset(self.rng)
            self.assertNotIn(state.food, state.body)

    def test_straight_left_right(self):
        initial = env.State((12, 13, 14), 1, 0)
        expected = [(15, 1), (8, 0), (20, 2)]
        for action, (head, heading) in enumerate(expected):
            state, event, done, reason = env.step(initial, action, self.rng)
            self.assertEqual(state.body, (13, 14, head))
            self.assertEqual(state.heading, heading)
            self.assertFalse(done)
            self.assertIsNone(reason)
            self.assertEqual(event["move"], 1)

    def test_food_growth_uses_old_food(self):
        state, event, done, _ = env.step(env.State((12, 13, 14), 1, 15), 0, self.rng)
        self.assertEqual(state.body, (12, 13, 14, 15))
        self.assertEqual(event, {"food": 1, "collision": 0, "move": 1, "distance": 1})
        self.assertNotIn(state.food, state.body)
        self.assertFalse(done)

    def test_wall_and_body_collision(self):
        cases = [(env.State((0, 1, 2), 0, 35), "wall_collision"),
                 (env.State((0, 1, 2, 8, 14, 13, 7), 0, 35), "body_collision")]
        for initial, expected in cases:
            state, event, done, reason = env.step(initial, 0, self.rng)
            self.assertTrue(done)
            self.assertEqual(reason, expected)
            self.assertEqual(state.body, initial.body)
            self.assertEqual(state.step_count, 1)
            self.assertEqual(event, {"food": 0, "collision": 1, "move": 1, "distance": 0})
            with self.assertRaises(ValueError):
                env.step(state, 0, self.rng)

    def test_departing_tail_is_available(self):
        initial = env.State((7, 8, 14, 13), 3, 35)
        state, event, done, _ = env.step(initial, 2, self.rng)
        self.assertFalse(done)
        self.assertEqual(state.body, (8, 14, 13, 7))
        self.assertEqual(event["collision"], 0)
        self.assertEqual(env.observe(initial)[117], 0)

    def test_board_completion(self):
        # A snake following a Hamiltonian path reaches the sole empty cell.
        path = []
        for row in range(6):
            path.extend(row * 6 + col for col in (range(6) if row % 2 == 0 else range(5, -1, -1)))
        initial = env.State(tuple(path[:-1]), 3, path[-1])
        state, event, done, reason = env.step(initial, 0, self.rng)
        self.assertTrue(done)
        self.assertEqual(reason, "board_complete")
        self.assertEqual(len(state.body), 36)
        self.assertIsNone(state.food)
        self.assertEqual(event["food"], 1)

    def test_food_on_last_move_counts(self):
        state, event, done, reason = env.step(env.State((12, 13, 14), 1, 15, 299), 0, self.rng)
        self.assertTrue(done)
        self.assertEqual(reason, "move_limit")
        self.assertEqual(state.step_count, 300)
        self.assertEqual(event["food"], 1)
        self.assertEqual(env.observe(state)[112], 0)

    def test_observation_order_and_body_order(self):
        state = env.State((12, 13, 14), 1, 15, 75)
        observation = env.observe(state)
        self.assertEqual(observation.shape, (118,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation[[12, 13, 14]], np.array([1, 2, 3]) / 36)
        self.assertEqual(observation[36 + 14], 1)
        self.assertEqual(observation[72 + 15], 1)
        np.testing.assert_array_equal(observation[108:112], [0, 1, 0, 0])
        self.assertEqual(observation[112], 0.75)
        np.testing.assert_allclose(observation[113:115], [0, 0.2])
        np.testing.assert_array_equal(env.observe_batch([state, state]), np.stack([observation, observation]))

    def test_signed_rewards(self):
        config = protocol.settings(food_reward=-4, collision_reward=3, move_reward=2, distance_reward=-1)
        self.assertEqual(env.training_reward({"food": 1, "collision": 0, "move": 1, "distance": 1}, config), -3)
        self.assertEqual(env.training_reward({"food": 0, "collision": 1, "move": 1, "distance": 0}, config), 5)
        self.assertEqual(env.training_reward({"food": 0, "collision": 0, "move": 1, "distance": -1}, config), 3)


class Evaluation(unittest.TestCase):
    def test_stable_uniform_policy_and_seeded_results(self):
        policy = evaluator.zero_policy()
        observations = env.observe_batch([env.reset(np.random.default_rng(9))])
        np.testing.assert_allclose(evaluator.policy_probabilities(policy, observations), [[1 / 3] * 3])
        first = evaluator.evaluate(policy, protocol.public_seed_pairs(), keep_replay=True)
        second = evaluator.evaluate(policy, protocol.public_seed_pairs(), keep_replay=True)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 30)
        self.assertEqual(first["mean_apples"], first["total_apples"] / 30)
        frames = first["replay"]["frames"]
        self.assertEqual(frames[0]["step"], 0)
        self.assertIsNotNone(frames[-1]["reason"])
        self.assertEqual([f["step"] for f in frames], list(range(len(frames))))
        json.dumps(first, allow_nan=False)

    def test_policy_validation(self):
        for mutation in ("shape", "nan", "large", "string", "bool"):
            policy = evaluator.zero_policy()
            if mutation == "shape":
                policy["layers"][0]["bias"].pop()
            else:
                policy["layers"][0]["bias"][0] = {"nan": float("nan"), "large": 1e7,
                                                     "string": "1", "bool": True}[mutation]
            with self.assertRaises(ValueError, msg=mutation):
                evaluator.validate_policy(policy)

    def test_ranking(self):
        better_food = {"total_apples": 4, "sum_food_steps": 1000}
        less_food = {"total_apples": 3, "sum_food_steps": 3}
        faster = {"total_apples": 4, "sum_food_steps": 999}
        self.assertLess(evaluator.ranking_key(better_food), evaluator.ranking_key(less_food))
        self.assertLess(evaluator.ranking_key(faster), evaluator.ranking_key(better_food))
        zero = evaluator.evaluate(evaluator.zero_policy(), [(1, 2)])
        self.assertEqual(zero["total_apples"], 0)
        self.assertIsNone(zero["mean_food_step"])

    def test_controls(self):
        for options in ({"food_reward": float("inf")}, {"collision_reward": 101},
                        {"episodes": True}, {"gamma": 0.8}, {"seed": 43}):
            with self.assertRaises(ValueError):
                protocol.settings(**options)
        self.assertEqual(protocol.settings(seed=43, instructor=True)["seed"], 43)
        self.assertEqual(len(protocol.public_seed_pairs()), 30)
        self.assertNotIn(protocol.DEMO_SEED_PAIR, protocol.public_seed_pairs())
        self.assertEqual(len(protocol.protocol_fingerprint()), 64)

    def test_mirrored_core(self):
        server = ROOT.parents[1] / "intro-ai-leaderboard" / "challenges" / "snake"
        for filename in ("protocol.py", "environment.py", "evaluator.py"):
            if (server / filename).exists():
                self.assertEqual((ROOT / filename).read_bytes(), (server / filename).read_bytes(), filename)


if __name__ == "__main__":
    unittest.main()
