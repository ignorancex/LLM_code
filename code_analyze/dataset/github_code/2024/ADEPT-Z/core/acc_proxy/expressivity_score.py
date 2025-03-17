"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-19 11:14:46
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:33:31
"""

from copy import deepcopy

import einops
import torch
from pyutils.config import configs

# from pyutils.general import logger
from pyutils.general import logger as lg
from pyutils.torch_train import (
    load_model,
)

from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear

__all__ = ["ExpressivityScoreEvaluator"]


class ExpressivityScoreEvaluator:
    _conv = (SuperBlockConv2d,)
    _linear = (SuperBlockLinear,)

    # prepare target weights in a wrapped class ExpressivityScoreEvaluator
    def __init__(
        self,
        checkpoint_path="./checkpoint/mnist/cnn/train_8_MZI/SuperOCNN__acc-98.38_epoch-90.pt",  # used to load checkpoint
        solution_path="./configs/mnist/genes/MZI_solution_8.txt",  # used to fix arch solution
        device=torch.device("cuda:0"),
        dataset="mnist",  # specify the dataset
        model="cnn",  # specify the model
        num_samples=100,  # specify the number of samples
    ):
        from core.builder import make_model

        # load config file
        config_file = f"configs/{dataset}/{model}/train_baseline_16.yml"
        configs.load(config_file, recursive=True)
        model_cfg = configs.model

        self.device = device
        self.num_samples = num_samples

        # build target model(default)
        self.target_model = make_model(
            device=self.device, model_cfg=model_cfg, random_state=42
        ).to(device)

        # load checkpoint
        load_model(model=self.target_model, path=checkpoint_path)

        with open(solution_path, "r") as file:
            solution = file.read()

        if isinstance(solution, str):
            solution = eval(solution)

        # fix arch solution with MZI_solution.txt
        self.target_model.fix_arch_solution(solution)

        # prepare weights from target model
        self.U_t, self.Vh_t = self._prepare_weights()

    # prepare weights from target model
    def _prepare_weights(self):
        with torch.no_grad():
            W_list = []
            for m in self.target_model.modules():
                if isinstance(m, self._conv) or isinstance(m, self._linear):
                    W = m.super_layer.get_weight_matrix(
                        m.super_ps_layers, m.weight.data
                    ).data
                    W_list.append(W)

            W = torch.cat(W_list, dim=1)
            W_target = W[:, : self.num_samples, :, :]  # W_target:[1, num_samples, k, k]

            # U_t:[1,num_samples,k,k], V_t:[1,num_samples,k,k]
            U_t, _, Vh_t = torch.linalg.svd(W_target)
            return U_t, Vh_t

    # compute expressivity score
    def compute_expressivity_score(
        self,
        model,
        num_samples: int = 400,
        num_steps: int = 100,
        verbose: bool = False,
    ) -> float:
        """
        Compute the expressivity score for the given model.
        The method largely follows the implementation provided earlier,
        but it uses the target model's weights prepared during the class initialization.
        """
        num_samples = min(num_samples, self.U_t.size(1))
        # print(num_samples)
        super_layer = model.super_layer

        super_ps_layers = super_layer.build_ps_layers(num_samples, 1)

        def objective():
            U, V = super_layer.get_UV(
                super_ps_layers, grid_dim_x=num_samples, grid_dim_y=1
            )

            expressivity_score = (
                (
                    einops.einsum(self.U_t, U.conj(), "p q i j, p q i j -> p q").real
                    * einops.einsum(self.Vh_t, V.conj(), "p q i j, p q i j -> p q").real
                )
                .mean()
                .div(super_layer.n_waveguides**2)
            )
            return -(expressivity_score)
        optimizer = torch.optim.Adam(params=super_ps_layers.parameters(), lr=1e-1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=8e-2
        )

        for i in range(num_steps):
            optimizer.zero_grad()
            expressivity_score = objective()
            expressivity_score.backward()
            optimizer.step()
            scheduler.step()
            if verbose and (i % 100 == 0 or i == num_steps - 1):
                lg.info(f"Step {i}: {-expressivity_score.item()}")

        return -expressivity_score.item()


class ParallelExpressivityScoreEvaluator:
    _conv = (SuperBlockConv2d,)
    _linear = (SuperBlockLinear,)

    # prepare target weights in a wrapped class ExpressivityScoreEvaluator
    def __init__(
        self,
        checkpoint_path="./checkpoint/mnist/cnn/train_8_MZI/SuperOCNN__acc-98.38_epoch-90.pt",
        solution_path="./configs/mnist/genes/MZI_solution_8.txt",
        device=torch.device("cuda:0"),
        dataset="mnist",
        model="cnn",
        num_samples=100,
    ):
        from core.builder import make_model

        configs_copy = deepcopy(configs)
        config_file = f"configs/{dataset}/{model}/search_16.yml"
        configs_copy.load(config_file, recursive=True)
        model_cfg = configs_copy.model

        self.device = device
        self.num_samples = num_samples

        # build target model(default)
        self.target_model = make_model(
            device=self.device, model_cfg=model_cfg, random_state=42
        ).to(device)

        # load checkpoint
        load_model(model=self.target_model, path=checkpoint_path)

        with open(solution_path, "r") as file:
            solution = file.read()

        if isinstance(solution, str):
            solution = eval(solution)

        # fix arch solution with MZI_solution.txt
        self.target_model.fix_arch_solution(solution)
        self.super_layers = []

        # prepare weights from target model
        self.U_t, self.Vh_t = self._prepare_weights()

    def _build_super_layers(self, num_population: int = 1, num_samples: int = 100):
        self.super_layers = [
            deepcopy(self.target_model.super_layer) for i in range(num_population)
        ]
        self.super_ps_layers = [
            l.build_ps_layers(num_samples, 1) for l in self.super_layers
        ]

    # prepare weights from target model
    def _prepare_weights(self):
        with torch.no_grad():
            W_list = []
            for m in self.target_model.modules():
                if isinstance(m, self._conv) or isinstance(m, self._linear):
                    W = m.super_layer.get_weight_matrix(
                        m.super_ps_layers, m.weight.data
                    ).data
                    W_list.append(W)

            W = torch.cat(W_list, dim=1)
            W_target = W[:, : self.num_samples, :, :]  # W_target:[1, num_samples, k, k]

            # U_t:[1,num_samples,k,k], V_t:[1,num_samples,k,k]
            U_t, _, Vh_t = torch.linalg.svd(W_target)

            return U_t, Vh_t

    # compute expressivity score
    def compute_expressivity_score(
        self,
        arch_sols,
        num_samples: int = 400,
        num_steps: int = 100,
        verbose: bool = True,
    ) -> float:
        """
        Compute the expressivity score for the given model.
        The method largely follows the implementation provided earlier,
        but it uses the target model's weights prepared during the class initialization.
        """
        lg.info("compute_expressivity_score function called.")

        num_samples = min(num_samples, self.U_t.size(1))
        # print(num_samples)
        if len(self.super_layers) != len(arch_sols):
            self._build_super_layers(len(arch_sols), num_samples)

        # now need to fix layer solutions inside this function
        # because we need to calculate expressivity scores for different solutions
        for i in range(len(self.super_layers)):
            super_layer = self.super_layers[i]
            solution = arch_sols[i]
            super_layer.fix_layer_solution(solution)

        # Now different super_layers are fixed with different solutions

        def objective():
            Us, Vs = [], []
            for super_layer, super_ps_layer in zip(
                self.super_layers, self.super_ps_layers
            ):
                U, V = super_layer.get_UV(
                    super_ps_layer, grid_dim_x=num_samples, grid_dim_y=1
                )
                Us.append(U)
                Vs.append(V)
            Us = torch.stack(Us, 0)
            Vs = torch.stack(Vs, 0)

            # lg.info(f"Shape of Us and Vs: {Us.shape}, {Vs.shape}")

            expressivity_score = (
                (
                    einops.einsum(
                        self.U_t, Us.conj(), "p q i j, b p q i j -> b p q"
                    ).real
                    * einops.einsum(
                        self.Vh_t, Vs.conj(), "p q i j, b p q i j -> b p q"
                    ).real
                )
                .mean(dim=[-2, -1])
                .div(super_layer.n_waveguides**2)
            )
            # lg.info(f"Shape of expressivity_score: {expressivity_score.shape}")
            # lg.info(f"value of expressivity_scores: {expressivity_score.tolist()}")
            return -(expressivity_score)  # [b]

        total_parameters = []
        for super_ps_layer in self.super_ps_layers:
            total_parameters.extend(list(super_ps_layer.parameters()))

        optimizer = torch.optim.Adam(params=total_parameters, lr=1e-1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=8e-2
        )

        for i in range(num_steps):
            optimizer.zero_grad()
            expressivity_score = objective()  # [b]
            expressivity_score_sum = expressivity_score.sum()
            expressivity_score_sum.backward()
            optimizer.step()
            scheduler.step()
            if verbose and (i % 10 == 0 or i == num_steps - 1):
                lg.info(
                    f"value of expressivity_scores: {[-x for x in expressivity_score.tolist()]}"
                )
                if i % 100 == 0 or i == num_steps - 1:
                    lg.info(f"Step {i}: {-expressivity_score_sum.item()}")

        return (-expressivity_score).detach().cpu().numpy().tolist()
