import unittest

import numpy as np

from src.sampler.gradient_estimation import build_gradient_estimation_fn
from src.scoring_rules.scoring_rules import EnergyScore, KernelScore
from src.models.g_and_k_model import uni_g_and_k, mv_g_and_k
from src.models.lorenz_96_model import Lorenz96SDE
from src.models.neural_lorenz_96_model import NeuralLorenz96SDE
from src.utils import transform_neural_lorenz_parameter

import functools

import torch


class UniGkModelTests(unittest.TestCase):
    def setUp(self) -> None:
        sim_gk = uni_g_and_k()
        sim_parameters = torch.tensor([2.,2.,2.,2.])
        self.sims = sim_gk.torch_forward_simulate(sim_parameters, 10)

        return super().setUp()

    def test_randomness(self):
        gk = uni_g_and_k()
        sims = gk.torch_forward_simulate(torch.tensor([2.,2.,2.,2.]), 10)
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(self.sims, sims)

class MultiGkModelTests(unittest.TestCase):
    def setUp(self) -> None:
        sim_gk = mv_g_and_k()
        sim_parameters = torch.tensor([2.,2.,2.,2.,0.1])
        self.sims = sim_gk.torch_forward_simulate(sim_parameters, 10)

        return super().setUp()

    def test_randomness(self):
        gk = mv_g_and_k()
        sims = gk.torch_forward_simulate(torch.tensor([2.,2.,2.,2.,0.1]), 10)
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(self.sims, sims)

class L96ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        sim_l96 = Lorenz96SDE()
        sim_parameters = torch.tensor([2.,0.5,2.])
        self.sims = sim_l96.torch_forward_simulate(sim_parameters, 10)

        return super().setUp()

    def test_randomness(self):
        l96 = Lorenz96SDE()
        sims = l96.torch_forward_simulate(torch.tensor([2.,0.5,2.]), 10)
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(self.sims, sims)

class L96ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        sim_l96 = Lorenz96SDE()
        self.sims = sim_l96.torch_forward_simulate(np.array([1.5,0.5,1.0]), 10).detach()

        return super().setUp()

    def test_randomness(self):
        torch.set_default_dtype(torch.float64)
        l96 = Lorenz96SDE()
        sims = l96.torch_forward_simulate(np.array([1.5,0.5,1.0]), 10).detach()
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(self.sims, sims)

class NeuralL96ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        sim_l96 = NeuralLorenz96SDE()
        default_params = sim_l96.get_default_mlp_params(seed=0)
        init_params = torch.from_numpy(np.concatenate([np.array([0.5])] + [p.detach().flatten().numpy() for p in default_params]))
        init_params = transform_neural_lorenz_parameter(init_params)
        self.sims = sim_l96.torch_forward_simulate(init_params, 10).detach()

        return super().setUp()

    def test_randomness(self):
        torch.set_default_dtype(torch.float64)
        l96 = NeuralLorenz96SDE()
        default_params = l96.get_default_mlp_params(seed=0)
        init_params = torch.from_numpy(np.concatenate([np.array([0.5])] + [p.detach().flatten().numpy() for p in default_params]))
        init_params = transform_neural_lorenz_parameter(init_params)
        sims = l96.torch_forward_simulate(init_params, 10).detach()
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        assert_equal(self.sims, sims)