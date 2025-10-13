import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as np
import jax.random as random
from ddt import data, ddt, unpack
from utils import TestCase

from dist_alg_common import sample_env_actions, sample_env_policies
from dist_env import env_step


@ddt
class EnvDistancingTestCase(TestCase):
    @data(
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), np.array([4.0, 3, 2, 1])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]), np.array([8.0, 8, 6, 6])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), np.array([8.0, 8, 6, 6, 2])),
    )
    @unpack
    def test_env_step__without_congestion__gives_positive_reward(self, actions, expected_reward):
        initial_state = np.array(0)

        state, reward = env_step(initial_state, actions)

        self.assertEqual(state.item(), 0)
        self.assertEqualJax(reward, expected_reward)

    @data(
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), np.array([4.0, 3, 2, 1])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]), np.array([8.0, 8, 6, 6, 4, 4, 2, 2])),
    )
    @unpack
    def test_env_step__without_congestion_in_bad_state__transitions_and_gives_positive_reward(self, actions, expected_reward):
        initial_state = np.array(1)

        state, reward = env_step(initial_state, actions)

        self.assertEqual(state.item(), 0)
        self.assertEqualJax(reward, expected_reward)

    @data(
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]), np.array([-80.0, -80, -80, -80, -80])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), np.array([-92.0, -92])),
    )
    @unpack
    def test_env_step__with_congestion__transitions_and_gives_negative_reward(self, actions, expected_reward):
        initial_state = np.array(0)

        state, reward = env_step(initial_state, actions)

        self.assertEqual(state.item(), 1)
        self.assertEqualJax(reward, expected_reward)

    @data(
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]), np.array([-80.0, -80, -80, -80, -80])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]), np.array([-92.0, -92, -94, -94])),
        (np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), np.array([-92.0, -92])),
    )
    @unpack
    def test_env_step__with_congestion_in_bad_state__gives_negative_reward(self, actions, expected_reward):
        initial_state = np.array(1)

        state, reward = env_step(initial_state, actions)

        self.assertEqual(state.item(), 1)
        self.assertEqualJax(reward, expected_reward)

    def test_sample_env_actions__returns_correct_shape(self):
        key = random.PRNGKey(2)
        key = random.split(key, 8)
        policy = np.zeros((8, 4))

        result = sample_env_actions(policy, key)

        self.assertShape(result, (8,))

    def test_sample_env_policies__returns_correct_shape(self):
        key = random.PRNGKey(2)
        key = random.split(key, 8)
        policy = np.zeros((8, 4))

        result = sample_env_policies(policy, key)

        self.assertShape(result, (8, 4))

    def test_sample_actions__returns_approximately_sampled_actions(self):
        policy = np.array(
            [
                [[0.9, 0, 0.1, 0], [1, 0, 0, 0]],
                [[0.9, 0, 0.1, 0], [1, 0, 0, 0]],
                [[0.9, 0.1, 0, 0], [0, 1, 0, 0]],
                [[0.9, 0.1, 0, 0], [0, 1, 0, 0]],
                [[0.1, 0.9, 0, 0], [0, 0, 1, 0]],
                [[0.1, 0.9, 0, 0], [0, 0, 1, 0]],
                [[0.1, 0.9, 0, 0], [0, 0, 0, 1]],
                [[0.1, 0.9, 0, 0], [0, 0, 0, 1]],
            ]
        )
        key = random.key(0)
        n_repetitions = 1000
        n_agents = 8
        n_actions = 4
        count = np.zeros((n_agents, n_actions))

        for _ in range(n_repetitions):
            key, subkey = random.split(key)
            keys = random.split(subkey, n_agents)
            state = np.array(0)

            actions = sample_env_actions(policy[:, state, :], keys)
            for agent in range(n_agents):
                count = count.at[agent, actions[agent].item()].add(1)

        self.assertAlmostEqualJax(
            count,
            np.array(
                [
                    [900, 0, 100, 0.0],
                    [900, 0, 100, 0],
                    [900, 100, 0, 0],
                    [900, 100, 0, 0],
                    [100, 900, 0, 0],
                    [100, 900, 0, 0],
                    [100, 900, 0, 0],
                    [100, 900, 0, 0],
                ]
            ),
            delta=30,
        )
