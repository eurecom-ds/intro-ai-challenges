"""Rule, reward, observation and canonical evaluation checks."""
import json
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import environment as env
import evaluator
import protocol


def cell(row, col):
    return row * protocol.BOARD_SIZE + col


def horizontal_body():
    return tuple(cell(7, col) for col in (5, 6, 7))


class GameRules(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_all_initial_placements(self):
        expected = 4 * protocol.BOARD_SIZE * (protocol.BOARD_SIZE - protocol.INITIAL_LENGTH + 1)
        self.assertEqual(len(env.INITIAL_PLACEMENTS), expected)
        self.assertEqual(len(set(env.INITIAL_PLACEMENTS)), expected)
        for body, heading in env.INITIAL_PLACEMENTS:
            self.assertEqual(len(set(body)), 3)
            dr, dc = env.DIRECTIONS[heading]
            for first, second in zip(body, body[1:]):
                r1, c1 = divmod(first, protocol.BOARD_SIZE)
                r2, c2 = divmod(second, protocol.BOARD_SIZE)
                self.assertEqual((r2 - r1, c2 - c1), (dr, dc))
        for _ in range(500):
            state = env.reset(self.rng)
            self.assertNotIn(state.food, state.body)

    def test_straight_left_right(self):
        initial = env.State(horizontal_body(), 1, 0)
        expected = [(cell(7, 8), 1), (cell(6, 7), 0), (cell(8, 7), 2)]
        for action, (head, heading) in enumerate(expected):
            state, event, done, reason = env.step(initial, action, self.rng)
            self.assertEqual(state.body, (*horizontal_body()[1:], head))
            self.assertEqual(state.heading, heading)
            self.assertFalse(done)
            self.assertIsNone(reason)
            self.assertEqual(event["move"], 1)

    def test_food_growth_uses_old_food(self):
        food = cell(7, 8)
        state, event, done, _ = env.step(env.State(horizontal_body(), 1, food), 0, self.rng)
        self.assertEqual(state.body, (*horizontal_body(), food))
        self.assertEqual(event, {"food": 1, "collision": 0, "move": 1, "distance": 1})
        self.assertNotIn(state.food, state.body)
        self.assertFalse(done)

    def test_wall_and_body_collision(self):
        wall_body = tuple(cell(row, 0) for row in (2, 1, 0))
        loop = tuple(cell(row, col) for row, col in
                     ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1)))
        cases = [(env.State(wall_body, 0, protocol.BOARD_CELLS - 1), "wall_collision"),
                 (env.State(loop, 0, protocol.BOARD_CELLS - 1), "body_collision")]
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
        body = tuple(cell(row, col) for row, col in ((1, 1), (1, 2), (2, 2), (2, 1)))
        initial = env.State(body, 3, protocol.BOARD_CELLS - 1)
        state, event, done, _ = env.step(initial, 2, self.rng)
        self.assertFalse(done)
        self.assertEqual(state.body, (*body[1:], body[0]))
        self.assertEqual(event["collision"], 0)
        self.assertEqual(env.observe(initial)[3 * protocol.BOARD_CELLS + 9], 0)

    def test_board_completion(self):
        # A snake following a Hamiltonian path reaches the sole empty cell.
        path = []
        for row in range(protocol.BOARD_SIZE):
            columns = range(protocol.BOARD_SIZE) if row % 2 == 0 else reversed(range(protocol.BOARD_SIZE))
            path.extend(cell(row, col) for col in columns)
        initial = env.State(tuple(path[:-1]), 1 if protocol.BOARD_SIZE % 2 else 3, path[-1])
        state, event, done, reason = env.step(initial, 0, self.rng)
        self.assertTrue(done)
        self.assertEqual(reason, "board_complete")
        self.assertEqual(len(state.body), protocol.BOARD_CELLS)
        self.assertIsNone(state.food)
        self.assertEqual(event["food"], 1)

    def test_food_on_last_move_counts(self):
        state, event, done, reason = env.step(
            env.State(horizontal_body(), 1, cell(7, 8), protocol.MAX_STEPS - 1), 0, self.rng)
        self.assertTrue(done)
        self.assertEqual(reason, "move_limit")
        self.assertEqual(state.step_count, protocol.MAX_STEPS)
        self.assertEqual(event["food"], 1)
        self.assertEqual(env.observe(state)[3 * protocol.BOARD_CELLS + 4], 0)

    def test_observation_order_and_body_order(self):
        body = horizontal_body()
        state = env.State(body, 1, cell(10, 12), protocol.MAX_STEPS // 4)
        observation = env.observe(state)
        self.assertEqual(observation.shape, (685,))
        self.assertEqual(observation.dtype, np.float32)
        cells = protocol.BOARD_CELLS
        auxiliary = 3 * cells
        np.testing.assert_allclose(observation[list(body)], np.array([1, 2, 3]) / cells)
        self.assertEqual(observation[cells + body[-1]], 1)
        self.assertEqual(observation[2 * cells + state.food], 1)
        np.testing.assert_array_equal(observation[auxiliary:auxiliary + 4], [0, 1, 0, 0])
        self.assertEqual(observation[auxiliary + 4], 0.75)
        np.testing.assert_allclose(observation[auxiliary + 5:auxiliary + 7], [3 / 14, 5 / 14])
        np.testing.assert_array_equal(env.observe_batch([state, state]), np.stack([observation, observation]))

    def test_far_board_cells_remain_distinct_from_auxiliary_features(self):
        body = tuple(cell(14, col) for col in (12, 13, 14))
        state = env.State(body, 1, cell(13, 14))
        observation = env.observe(state)
        self.assertAlmostEqual(float(observation[224]), 3 / 225)
        self.assertEqual(observation[225 + 224], 1)
        self.assertEqual(observation[450 + cell(13, 14)], 1)
        np.testing.assert_array_equal(observation[682:685], [1, 0, 1])

    def test_signed_rewards(self):
        config = protocol.settings(food_reward=-4, collision_reward=3, move_reward=2, distance_reward=-1)
        self.assertEqual(env.training_reward({"food": 1, "collision": 0, "move": 1, "distance": 1}, config), -3)
        self.assertEqual(env.training_reward({"food": 0, "collision": 1, "move": 1, "distance": 0}, config), 5)
        self.assertEqual(env.training_reward({"food": 0, "collision": 0, "move": 1, "distance": -1}, config), 3)


class Evaluation(unittest.TestCase):
    def test_surviving_policy_uses_the_full_extended_move_budget(self):
        policy = evaluator.zero_policy()
        policy["layers"][-1]["bias"][1] = 100
        result = evaluator.evaluate(policy, [(10000, 20000)], keep_replay=True)
        self.assertEqual(result["diagnostics"]["mean_steps"], protocol.MAX_STEPS)
        self.assertEqual(result["diagnostics"]["terminal_reasons"], {"move_limit": 1})
        frames = result["replay"]["frames"]
        self.assertEqual(len(frames), protocol.MAX_STEPS + 1)
        self.assertEqual(frames[-1]["step"], protocol.MAX_STEPS)

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

    def test_larger_policy_export_fits_and_six_by_six_weights_reject(self):
        policy = evaluator.zero_policy()
        rng = np.random.default_rng(8)
        for layer, shape in zip(policy["layers"], protocol.LAYER_SHAPES):
            layer["weight"] = rng.normal(0, 0.1, shape).astype(np.float32).tolist()
            layer["bias"] = rng.normal(0, 0.1, shape[0]).astype(np.float32).tolist()
        payload = {"challenge_version": protocol.CHALLENGE_VERSION,
                   "protocol_fingerprint": protocol.protocol_fingerprint(),
                   "student_name": "Ada", "student_surname": "Lovelace", "phase": "final",
                   "selected": protocol.settings(),
                   "validation": {"episodes": 30, "total_apples": 3, "sum_food_steps": 30},
                   "policy": policy}
        stamp = "2026-10-29T12:00:00.000001Z"
        canonical = json.dumps({"created_at": stamp, "submission": payload},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        payload["fingerprint"] = {"created_at": stamp, "algorithm": "sha256",
                                  "value": hashlib.sha256(canonical.encode()).hexdigest()}
        raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.assertGreater(len(raw), 512 * 1024)
        self.assertLessEqual(len(raw), protocol.MAX_JSON_BYTES)
        validated = evaluator.validate_policy(json.loads(raw)["policy"])
        self.assertEqual(validated["layers"][0]["weight"].shape, (64, 685))
        legacy = evaluator.zero_policy()
        legacy["architecture"] = "mlp-118-64-64-3-relu-v1"
        with self.assertRaises(ValueError):
            evaluator.validate_policy(legacy)
        legacy["architecture"] = protocol.ARCHITECTURE
        legacy["layers"][0]["weight"] = np.zeros((64, 118)).tolist()
        with self.assertRaises(ValueError):
            evaluator.validate_policy(legacy)
        with self.assertRaises(ValueError):
            evaluator.policy_probabilities(policy, np.zeros((1, 118)))

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
