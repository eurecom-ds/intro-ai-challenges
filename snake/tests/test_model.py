"""Checks for the episodic estimator and retained CPU actor."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from snake import environment, evaluator, model, protocol


class EstimatorTests(unittest.TestCase):
    def test_discounted_returns_and_terminal_boundary(self):
        np.testing.assert_allclose(model.discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3])
        np.testing.assert_allclose(model.discounted_returns([1, 2, 3], 1), [6, 5, 3])
        np.testing.assert_allclose(model.discounted_returns([1, 2, 3], 0), [1, 2, 3])
        np.testing.assert_array_equal(model.discounted_returns([4], .99), [4])

    def test_actor_loss_uses_episode_sums_and_detaches_baseline(self):
        logits = torch.tensor([.2, -.4, .6, -.8], requires_grad=True)
        advantage = torch.tensor([1., 2., 3., 4.], requires_grad=True)
        discounts = torch.tensor([1., 1., .5, .25])
        objective = model.actor_loss(logits, advantage, discounts, 2)
        self.assertAlmostEqual(float(objective.detach()), .25, places=6)
        objective.backward()
        np.testing.assert_allclose(logits.grad.numpy(), [-.5, -1, -.75, -.5])
        self.assertIsNone(advantage.grad)

    def test_actor_and_baseline_are_separate_and_baseline_starts_at_zero(self):
        with model._cpu_randomness(42):
            actor = model.PolicyNetwork()
            baseline = model.ValueNetwork()
        inputs = torch.ones(3, protocol.OBSERVATION_SIZE)
        np.testing.assert_array_equal(baseline(inputs).detach().numpy(), [0, 0, 0])
        actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
        optimizer = torch.optim.Adam(baseline.parameters(), lr=.001)
        (baseline(inputs) - 1).square().mean().backward()
        optimizer.step()
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(actor_before, actor.parameters())))

    def test_gamma_zero_discount_keeps_first_action_only(self):
        log_probabilities = torch.ones(3, requires_grad=True)
        model.actor_loss(log_probabilities, torch.ones(3), torch.tensor([1., 0, 0]), 1).backward()
        np.testing.assert_array_equal(log_probabilities.grad.numpy(), [-1, 0, 0])

    def test_exported_actor_matches_numpy_probabilities(self):
        with model._cpu_randomness(42):
            actor = model.PolicyNetwork()
        rng = np.random.default_rng(15)
        observations = environment.observe_batch([environment.reset(rng) for _ in range(4)])
        expected = torch.softmax(actor(torch.from_numpy(observations)), -1).detach().numpy()
        actual = evaluator.policy_probabilities(model.export_policy(actor), observations)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


class TrainingTests(unittest.TestCase):
    def test_zero_rewards_are_stable_and_final_actor_is_reproducible(self):
        config = protocol.settings(episodes=256, food_reward=0, collision_reward=0,
                                   move_reward=0, distance_reward=0)
        pairs = protocol.public_seed_pairs()[:2]
        result = model.train_policy(config, evaluation_seed_pairs=pairs)
        self.assertEqual(result["training_episodes"], 256)
        self.assertGreater(result["training_steps"], 256)
        self.assertEqual(result["history"][-1]["episodes"], 256)
        self.assertEqual(result["policy"], model.initial_policy())
        self.assertEqual(result["validation"], result["history"][-1]["validation"])
        again = model.train_policy(config, evaluation_seed_pairs=pairs)
        self.assertEqual(result["policy"], again["policy"])
        self.assertEqual(result["training_steps"], again["training_steps"])

    def test_retained_actor_matches_final_evaluation_and_progress(self):
        config = protocol.settings(episodes=256)
        rows = []
        result = model.train_policy(config, progress=rows.append,
                                    evaluation_seed_pairs=protocol.public_seed_pairs()[:2])
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["episodes"], 0)
        self.assertEqual(rows[-1]["episodes"], 256)
        self.assertEqual(rows[-1]["validation"], result["validation"])
        self.assertTrue(all(np.isfinite(row["policy_loss"]) for row in rows[1:]))

    def test_bad_settings_fail_before_collection(self):
        with patch.object(model, "collect_episodes") as collection:
            with self.assertRaises(ValueError):
                model.train_policy({"food_reward": float("inf")})
            with self.assertRaises(ValueError):
                model.train_policy(seed=True)
            collection.assert_not_called()

    def test_signed_extreme_rewards_and_gamma_boundaries_stay_finite(self):
        for gamma in (0, 1):
            config = protocol.settings(episodes=256, learning_rate=.01, gamma=gamma,
                                       food_reward=-100, collision_reward=100,
                                       move_reward=10, distance_reward=-10)
            result = model.train_policy(config, evaluation_seed_pairs=protocol.public_seed_pairs()[:2])
            evaluator.validate_policy(result["policy"])
            self.assertTrue(all(np.isfinite(row["value_loss"]) for row in result["history"][1:]))


if __name__ == "__main__":
    unittest.main()
