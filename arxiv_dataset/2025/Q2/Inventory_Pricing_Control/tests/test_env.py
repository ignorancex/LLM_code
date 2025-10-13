# tests/test_env.py

import unittest
from envs import InvPricingManagementMasterEnv, InvPricingManagementMasterEnvPPO
import numpy as np


class TestInvManagementEnv(unittest.TestCase):

    def test_environment_initialization(self):
        env = InvPricingManagementMasterEnv()
        self.assertEqual(env.init_inv, 15)
        self.assertEqual(env.num_periods, 50)

    def test_step_function(self):
        env = InvPricingManagementMasterEnv()
        state = env.reset()
        action = env.sample_action()
        next_state, reward, done, info = env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


class TestInvManagementEnvPPO(unittest.TestCase):

    def test_environment_initialization(self):
        env = InvPricingManagementMasterEnvPPO()
        self.assertEqual(env.init_inv, 15)
        self.assertEqual(env.num_periods, 50)

    def test_step_function(self):
        env = InvPricingManagementMasterEnvPPO()
        state = env.reset()
        action = env.sample_action()
        next_state, reward, done, info = env.step(action)
        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


if __name__ == '__main__':
    unittest.main()
