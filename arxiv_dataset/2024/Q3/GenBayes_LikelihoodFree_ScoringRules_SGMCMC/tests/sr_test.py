import unittest

import jax.numpy as jnp
import numpy as np
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform

from src.scoring_rules.scoring_rules import EnergyScore, KernelScore
import torch



class EnergyScoreTests(unittest.TestCase):

    def setUp(self):
        torch.set_default_dtype(torch.float32)
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.scoring_rule = EnergyScore(beta=2)
        self.scoring_rule_beta1 = EnergyScore(beta=1)
        self.scoring_rule_jax = EnergyScore(beta=2, use_jax=True)
        self.scoring_rule_beta1_jax = EnergyScore(beta=1, use_jax=True)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100,
                                                                                rng=np.random.RandomState(1))
        # create observed data
        self.y_obs = [1.8]
        self.y_obs_double = [1.8, 0.9]

    def test_score(self):
        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        expected_score = 0.400940132262833
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score * 2)

    def test_score_additive(self):
        for score in [self.scoring_rule, self.scoring_rule_beta1]:
            comp_loglikelihood_a = score.score([self.y_obs_double[0]], self.y_sim)
            comp_loglikelihood_b = score.score([self.y_obs_double[1]], self.y_sim)
            comp_loglikelihood_two = score.score(self.y_obs_double, self.y_sim)

            self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

    def test_jax_numpy(self):
        # compute the score using numpy
        numpy_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        # compute the score using jax
        jax_score = self.scoring_rule_jax.score(self.y_obs, self.y_sim)
        # check they are identical; notice jax uses reduced precision, so need to change a bit the tolerance
        self.assertTrue(np.allclose(numpy_score, jax_score, atol=1e-5, rtol=1e-5))

        # compute the score using numpy
        numpy_score = self.scoring_rule_beta1.score(self.y_obs, self.y_sim)
        # compute the score using jax
        jax_score = self.scoring_rule_beta1_jax.score(self.y_obs, self.y_sim)
        # check they are identical; notice jax uses reduced precision, so need to change a bit the tolerance
        self.assertTrue(np.allclose(numpy_score, jax_score, atol=1e-5, rtol=1e-5))


class KernelScoreTests(unittest.TestCase):

    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.scoring_rule = KernelScore()
        self.scoring_rule_jax = KernelScore(use_jax=True)
        self.scoring_rule_biased = KernelScore(biased_estimator=True)
        self.scoring_rule_biased_jax = KernelScore(use_jax=True, biased_estimator=True)

        def def_negative_Euclidean_distance(beta=1.0):
            if beta <= 0 or beta > 2:
                raise RuntimeError("'beta' not in the right range (0,2]")

            if beta == 1:
                def Euclidean_distance(x, y):
                    return - np.linalg.norm(x - y)
            else:
                def Euclidean_distance(x, y):
                    return - np.linalg.norm(x - y) ** beta

            return Euclidean_distance

        def def_negative_Euclidean_distance_jax(beta=1.0):
            if beta <= 0 or beta > 2:
                raise RuntimeError("'beta' not in the right range (0,2]")

            if beta == 1:
                def Euclidean_distance(x, y):
                    return - jnp.linalg.norm(x - y)
            else:
                def Euclidean_distance(x, y):
                    return - jnp.linalg.norm(x - y) ** beta

            return Euclidean_distance

        self.kernel_energy_SR = KernelScore(kernel=def_negative_Euclidean_distance(beta=1.4))
        self.energy_SR = EnergyScore(beta=1.4)
        self.kernel_energy_SR_jax = KernelScore(
            kernel=def_negative_Euclidean_distance_jax(beta=1.4), use_jax=True)
        self.energy_SR_jax = EnergyScore(beta=1.4, use_jax=True)

        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100,
                                                                                rng=np.random.RandomState(1))

        def def_gaussian_kernel_numpy(sigma=1):
            sigma_2 = 2 * sigma ** 2

            def Gaussian_kernel(x, y):
                xy = x - y
                return np.exp(- np.dot(xy, xy) / sigma_2)

            return Gaussian_kernel

        def def_gaussian_kernel_jax(sigma=1):
            sigma_2 = 2 * sigma ** 2

            def Gaussian_kernel(x, y):
                xy = x - y
                return jnp.exp(- jnp.dot(xy, xy) / sigma_2)

            return Gaussian_kernel

        # try providing the gaussian kernel in jax as an external function:
        self.kernel_SR_external = KernelScore(
            kernel=def_gaussian_kernel_numpy(), use_jax=False)
        self.kernel_SR_external_biased = KernelScore(biased_estimator=True,
                                                     kernel=def_gaussian_kernel_numpy(), use_jax=False)
        self.kernel_SR_external_jax = KernelScore(
            kernel=def_gaussian_kernel_jax(), use_jax=True)
        self.kernel_SR_external_jax_biased = KernelScore(biased_estimator=True,
                                                         kernel=def_gaussian_kernel_jax(), use_jax=True)

        # create observed data
        self.y_obs = [1.8]
        self.y_obs_double = [1.8, 0.9]

    def test_error_init(self):
        # test if it raises RuntimeError when kernel is a list:
        self.assertRaises(RuntimeError, KernelScore, kernel=[])

        # test if it raises NotImplementedError when kernel is "cauchy":
        self.assertRaises(NotImplementedError, KernelScore, kernel="cauchy")

    def test_score(self):
        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        expected_score = -0.7045988787568286
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score)

    def test_match_energy_score(self):
        comp_score1 = self.kernel_energy_SR.score(self.y_obs_double, self.y_sim)
        comp_score2 = self.energy_SR.score(self.y_obs_double, self.y_sim)
        self.assertAlmostEqual(comp_score1, comp_score2)

        comp_score1 = self.kernel_energy_SR_jax.score(self.y_obs_double, self.y_sim)
        comp_score2 = self.energy_SR_jax.score(self.y_obs_double, self.y_sim)
        self.assertAlmostEqual(comp_score1, comp_score2, places=5)  # reduced precision with jax

    def test_match_external_gaussian_kernel(self):
        sr_external_list = [self.kernel_SR_external, self.kernel_SR_external_biased, self.kernel_SR_external_jax,
                            self.kernel_SR_external_jax_biased]
        sr_interal_list = [self.scoring_rule, self.scoring_rule_biased, self.scoring_rule_jax,
                           self.scoring_rule_biased_jax]

        for sr_external, sr_internal in zip(sr_external_list, sr_interal_list):
            comp_score = sr_external.score(self.y_obs_double, self.y_sim)
            expected_score = sr_internal.score(self.y_obs_double, self.y_sim)
            self.assertAlmostEqual(comp_score, expected_score, places=5)

    def test_score_additive(self):
        comp_loglikelihood_a = self.scoring_rule.score([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.scoring_rule.score([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.scoring_rule.score(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

    def test_jax_numpy(self):
        # compute the score using numpy
        numpy_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        # compute the score using jax
        jax_score = self.scoring_rule_jax.score(self.y_obs, self.y_sim)
        # check they are identical; notice jax uses reduced precision, so need to change a bit the tolerance
        self.assertTrue(np.allclose(numpy_score, jax_score, atol=1e-5, rtol=1e-5))

        # compute the score using numpy
        numpy_score = self.scoring_rule_biased.score(self.y_obs, self.y_sim)
        # compute the score using jax
        jax_score = self.scoring_rule_biased_jax.score(self.y_obs, self.y_sim)
        # check they are identical; notice jax uses reduced precision, so need to change a bit the tolerance
        self.assertTrue(np.allclose(numpy_score, jax_score, atol=1e-5, rtol=1e-5))

        # compute the score using numpy
        numpy_score = self.kernel_energy_SR.score(self.y_obs, self.y_sim)
        # compute the score using jax
        jax_score = self.kernel_energy_SR_jax.score(self.y_obs, self.y_sim)
        # check they are identical; notice jax uses reduced precision, so need to change a bit the tolerance
        self.assertTrue(np.allclose(numpy_score, jax_score, atol=1e-5, rtol=1e-5))


class EnergyScoreTorchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(2)
        self.simulations = self.rng.randn(5, 3).astype("float32")
        self.observations = self.rng.randn(2, 3).astype("float32")
        self.simulations_torch = torch.from_numpy(self.simulations)
        self.observations_torch = torch.from_numpy(self.observations)
        self.sr = EnergyScore(beta=1.7)

    def test_numpy_torch_match(self):
        # you can test their accordance only in case of 1 single observation (ie batch element) due to the
        # different way they are computed

        numpy_value = self.sr.score(self.observations, self.simulations)
        torch_value = self.sr._estimate_score_torch(self.simulations_torch, self.observations_torch)
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

    def test_additive_batch_torch(self):
        score_1 = self.sr._estimate_score_torch(self.simulations_torch, self.observations_torch[0].unsqueeze(0))
        score_2 = self.sr._estimate_score_torch(self.simulations_torch, self.observations_torch[1].unsqueeze(0))
        score_joint = self.sr._estimate_score_torch(self.simulations_torch, self.observations_torch)

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))

    def test_fast_torch(self):
        score = self.sr._estimate_score_torch(simulations=self.simulations_torch, observations=self.observations_torch)
        score_f = self.sr.score(simulations=self.simulations_torch, observations=self.observations_torch, use_torch=True)
        torch.testing.assert_close(score, score_f)


class KernelScoreTorchTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(3)
        self.simulations = self.rng.randn(5, 3).astype("float32")
        self.observations = self.rng.randn(2, 3).astype("float32")
        self.simulations_torch = torch.from_numpy(self.simulations)
        self.observations_torch = torch.from_numpy(self.observations)
        self.sr_unbiased = KernelScore(sigma=1.5)
        self.sr_biased = KernelScore(biased_estimator=True, sigma=1.5)

        def def_negative_Euclidean_distance(beta=1.0):
            if beta <= 0 or beta > 2:
                raise RuntimeError("'beta' not in the right range (0,2]")

            if beta == 1:
                def Euclidean_distance(x, y):
                    return - torch.norm(x - y)
            else:
                def Euclidean_distance(x, y):
                    return - torch.norm(x - y) ** beta

            return Euclidean_distance

        self.sr_unbiased_kernel_energy = KernelScore(kernel=def_negative_Euclidean_distance(beta=1.4))
        self.sr_energy = EnergyScore(beta=1.4)

    def test_numpy_torch_match(self):
        # you can test their accordance only in case of 1 single observation (ie batch element) due to the
        # different way they are computed

        # unbiased:
        numpy_value = self.sr_unbiased.score(self.observations, self.simulations)
        torch_value = self.sr_unbiased._estimate_score_torch(self.simulations_torch, self.observations_torch)
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

        # biased:
        numpy_value = self.sr_biased.score(self.observations, self.simulations)
        torch_value = self.sr_biased._estimate_score_torch(self.simulations_torch, self.observations_torch)
        self.assertTrue(np.allclose(torch_value.numpy(), numpy_value))

    def test_additive_batch_torch(self):
        # unbiased:
        score_1 = self.sr_unbiased._estimate_score_torch(self.simulations_torch,
                                                         self.observations_torch[0].unsqueeze(0))
        score_2 = self.sr_unbiased._estimate_score_torch(self.simulations_torch,
                                                         self.observations_torch[1].unsqueeze(0))
        score_joint = self.sr_unbiased._estimate_score_torch(self.simulations_torch,
                                                             self.observations_torch)

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))

        # biased:
        score_1 = self.sr_biased._estimate_score_torch(self.simulations_torch,
                                                       self.observations_torch[0].unsqueeze(0))
        score_2 = self.sr_biased._estimate_score_torch(self.simulations_torch,
                                                       self.observations_torch[1].unsqueeze(0))
        score_joint = self.sr_biased._estimate_score_torch(self.simulations_torch,
                                                           self.observations_torch)

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))

        # hand defined kernel:
        score_1 = self.sr_unbiased_kernel_energy._estimate_score_torch(self.simulations_torch,
                                                                       self.observations_torch[0].unsqueeze(0))
        score_2 = self.sr_unbiased_kernel_energy._estimate_score_torch(self.simulations_torch,
                                                                       self.observations_torch[1].unsqueeze(0))
        score_joint = self.sr_unbiased_kernel_energy._estimate_score_torch(self.simulations_torch,
                                                                           self.observations_torch)

        self.assertTrue(torch.allclose(score_joint, score_2 + score_1))

    def test_match_energy_score(self):
        score_1 = self.sr_unbiased_kernel_energy._estimate_score_torch(self.simulations_torch,
                                                                       self.observations_torch)
        score_2 = self.sr_energy._estimate_score_torch(self.simulations_torch, self.observations_torch)

        self.assertTrue(torch.allclose(score_2, score_1))

    def test_fast_torch(self):
        score = self.sr_unbiased._estimate_score_torch(simulations=self.simulations_torch, observations=self.observations_torch)
        score_f = self.sr_unbiased.score(simulations=self.simulations_torch, observations=self.observations_torch, use_torch=True)
        torch.testing.assert_close(score, score_f)

if __name__ == '__main__':
    unittest.main()
