import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as np
from ddt import data, ddt, unpack
from utils import TestCase

from old.env_congestion import env_step


@ddt
class EnvCongestionTestCase(TestCase):
    @data(
        (np.array([0, 0, 0, 0]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), np.array([2, 2, 2, 2]), np.array([1, 1, 1, 1])),
        (np.array([0, 0, 0, 0]), np.array([[1, 0], [0, 1], [0, 1], [0, 1]]), np.array([1, 2, 2, 2]), np.array([50, 5, 5, 5])),
        (np.array([0, 0, 0, 0]), np.array([[1, 0], [1, 0], [0, 1], [0, 1]]), np.array([1, 1, 2, 2]), np.array([15, 15, 15, 15])),
        (np.array([0, 0, 0, 0]), np.array([[1, 0], [1, 0], [1, 0], [0, 1]]), np.array([1, 1, 1, 2]), np.array([5, 5, 5, 50])),
        (np.array([0, 0, 0, 0]), np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        #
        (np.array([1, 1, 2, 2]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), np.array([4, 4, 3, 3]), np.array([15, 15, 15, 15])),
        (np.array([1, 1, 2, 2]), np.array([[1, 0], [0, 1], [0, 1], [0, 1]]), np.array([3, 4, 3, 3]), np.array([50, 50, 15, 15])),
        (np.array([1, 1, 2, 2]), np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), np.array([3, 4, 4, 3]), np.array([50, 50, 50, 50])),
        (np.array([1, 1, 2, 2]), np.array([[1, 0], [1, 0], [0, 1], [0, 1]]), np.array([3, 3, 3, 3]), np.array([15, 15, 15, 15])),
        (np.array([1, 1, 2, 2]), np.array([[1, 0], [1, 0], [1, 0], [0, 1]]), np.array([3, 3, 4, 3]), np.array([15, 15, 50, 50])),
        (np.array([1, 1, 2, 2]), np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), np.array([3, 3, 4, 4]), np.array([15, 15, 15, 15])),
        #
        (np.array([1, 1, 1, 2]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]]), np.array([4, 4, 4, 3]), np.array([5, 5, 5, 50])),
        (np.array([1, 1, 1, 2]), np.array([[1, 0], [0, 1], [0, 1], [0, 1]]), np.array([3, 4, 4, 3]), np.array([50, 15, 15, 50])),
        (np.array([1, 1, 1, 2]), np.array([[1, 0], [1, 0], [0, 1], [0, 1]]), np.array([3, 3, 4, 3]), np.array([15, 15, 50, 50])),
        (np.array([1, 1, 1, 2]), np.array([[1, 0], [1, 0], [1, 0], [0, 1]]), np.array([3, 3, 3, 3]), np.array([5, 5, 5, 50])),
        (np.array([1, 1, 1, 2]), np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), np.array([3, 3, 3, 4]), np.array([5, 5, 5, 50])),
        #
        (np.array([3, 3, 3, 3]), np.array([[1, 0], [1, 0], [0, 1], [0, 1]]), np.array([0, 0, 0, 0]), np.array([15, 15, 15, 15])),
        (np.array([3, 3, 4, 4]), np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), np.array([0, 0, 0, 0]), np.array([50, 50, 50, 50])),
    )
    @unpack
    def test_env_step__transitions_and_gives_correct_reward(self, initial_state, actions, expected_next_state, expected_reward):
        state, reward = env_step(initial_state, actions)

        self.assertEqualJax(state, expected_next_state)
        self.assertEqualJax(reward, expected_reward)
        self.assertDtypesEqualJax(state, expected_next_state)

    @data(
        (np.array([0, 1, 0, 0]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
        (np.array([1, 1, 1, 3]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
        (np.array([2, 3, 2, 3]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
        (np.array([0, 3, 0, 0]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
        (np.array([0, 0, 0, 4]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
        (np.array([0, 0, 2, 0]), np.array([[0, 1], [0, 1], [0, 1], [0, 1]])),
    )
    @unpack
    def test_env_step__invalid_state_gives_zero_reward(self, initial_state, actions):
        _, reward = env_step(initial_state, actions)

        self.assertEqualJax(reward, 0)
