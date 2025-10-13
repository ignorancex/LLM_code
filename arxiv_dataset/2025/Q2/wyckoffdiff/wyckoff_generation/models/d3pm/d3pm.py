# coding=utf-8
#
# This file is a heavily modified version of source code made available under the Apache 2.0 license by Google Research made available in their repository at https://github.com/google-research/google-research/blob/8f818e8fc802c31571a89084d4c61922bf26483e/d3pm/images/diffusion_categorical.py
#
# Modifications Copyright (c) 2025 The High-Throughput Toolkit
#
# The original copyright header from the file from Google Research follows below:
#
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from wyckoff_generation.common.registry import registry
from wyckoff_generation.datasets import lookup_tables
from wyckoff_generation.models import model_utils
from wyckoff_generation.models.d3pm import utils


@registry.register_d3pm_transition("base")
class D3PMTransition:
    # Base class for a transition in D3PM
    def __init__(self, t_max, variable_dim, device):
        self.device = device
        self.t_max = t_max
        self.variable_dim = variable_dim

        self.betas = self.get_betas()

        self.Qt = torch.stack(
            [self.get_Qt_single(t).to(self.device) for t in range(self.t_max)]
        )
        assert self.Qt.shape == (self.t_max, self.variable_dim, self.variable_dim)

        Qt_bar = [self.Qt[0].unsqueeze(0)]
        for t in range(1, self.t_max):
            Qt_bar.append(Qt_bar[-1] @ self.Qt[t])
        self.Qt_bar = torch.vstack(Qt_bar)
        assert self.Qt_bar.shape == (
            self.t_max,
            self.variable_dim,
            self.variable_dim,
        ), self.Qt_bar.shape

    def get_Qt_single(self, t):
        """
        t: a single time-step (integer)
        returns: Qt: (d, d), with d being the number of categories
        """
        raise NotImplementedError

    def get_Qt_batch(self, t):
        """
        t: tensor of size (bs) with time steps
        returns: Qt: (bs, d, d) with d being the number of categories
        """
        return self.Qt[t]

    def get_Qt(self, t):
        return self.get_Qt_batch(t)

    def get_Qt_bar(self, t):
        """
        t: tensor of size (bs) with time steps
        returns: Qt_bar: (bs, d, d) with d being the number of categories
        """
        return self.Qt_bar[t]

    def sample_q_t_0(self, x0, t):
        """
        x0: tensor of size (num_variables) or (num_variables, 1)
        t: tensor of size (num_variables) with time-steps
        return xt: tensor of size (num_variables) with samples from q(xt|x0)
        """
        bs = x0.shape[0]
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(1)
        assert len(x0.shape) == 2 and x0.shape[1] == 1
        Qt_bar = self.get_Qt_bar(t)
        x0_onehot = F.one_hot(x0.long(), num_classes=self.variable_dim).float()
        qt_probs = (x0_onehot @ Qt_bar).squeeze(
            1
        )  # Result of batch product is bs x 1 x variable_dim, squeeze to bs x variable_dim
        assert qt_probs.shape == (bs, self.variable_dim), qt_probs.shape
        xt = utils.sample_from_distr(qt_probs)
        return xt

    def q_t_posterior_over0(self, x_t, t):
        """
        x_t: tensor of size BxN, where N=1 is a possibility (useful for the zero dof nodes to just stack them and add 1 as dimension)
        t: tensor of size B
        """
        assert len(x_t.shape) == 2 and x_t.shape[0] == t.shape[0], (x_t.shape, t.shape)
        x_t = F.one_hot(x_t, num_classes=self.variable_dim).float()
        Qt_T = self.get_Qt(t).transpose(-1, -2)
        Qt_bar = self.get_Qt_bar(t)
        Qnext_bar = self.get_Qt_bar(t - 1)
        left = x_t @ Qt_T
        left = left.unsqueeze(dim=2)
        right = Qnext_bar.unsqueeze(1)
        numerator = left * right

        x_t_T = x_t.transpose(-1, -2)

        denominator = Qt_bar @ x_t_T
        denominator = denominator.transpose(-1, -2)
        denominator = denominator.unsqueeze(-1)
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator
        return out

    def backward_probs(self, x_t, t, p_x0, q_t_posterior_over0=None):
        if q_t_posterior_over0 is None:
            q_t_posterior_over0 = self.q_t_posterior_over0(x_t, t)
        term = p_x0.unsqueeze(-1) * q_t_posterior_over0
        unnormalized_probs = term.sum(dim=2)
        unnormalized_probs[torch.sum(unnormalized_probs, dim=-1) == 0] = 1e-5
        probs = F.normalize(unnormalized_probs, p=1, dim=-1)
        return probs

    def backward_step(self, x_t, t, p_x0):
        """
        Input x_t and p_x0 should either be of size B and BxD, or BxN and BxNxD
        """
        x_t = x_t.long()
        input_shape = x_t.shape
        assert (len(p_x0.shape) == len(x_t.shape) + 1) and (len(x_t.shape) in [1, 2])
        assert (
            t.shape[0] == x_t.shape[0]
        ), "number of timesteps should be equal to number of input variables"
        if len(p_x0.shape) == 2:
            # in case input is B and BxD, implicitly set N=1 so that input is of size Bx1 and Bx1xD
            p_x0 = p_x0.unsqueeze(1)
            x_t = x_t.unsqueeze(1)
        probs = self.backward_probs(x_t, t, p_x0)
        # flatten to sample
        probs = probs.flatten(0, 1)
        x_t_minus_one = utils.sample_from_distr(probs)
        # reshape to original shape
        x_t_minus_one = x_t_minus_one.reshape(input_shape)
        return x_t_minus_one

    def vb_loss(self, x_0_pred_logits, x_0, x_t, t, batch, num_samples):
        x_0 = x_0.long()
        # obtain q(x_t-1|x_t, x_0) for all valus of x_0
        q_posterior_over_0 = self.q_t_posterior_over0(
            x_t, t
        )  # Bx D x variable_dim_0 x variable_dim_{t-1} (in this case x0 and x_{t-1} have same dimension)
        # now gather the distribution q(x_t-1|x_t, x_0) for the actual x_0
        # TODO: is gather really necessary, or is it better you use torch.arange?
        q_posterior_true = q_posterior_over_0.gather(
            2, x_0[..., None, None].repeat_interleave(self.variable_dim, -1)
        ).squeeze(
            -2
        )  # B x D x variable_dim
        # Reuse all q(x_t-1|x_t, x_0) for computing p(x_t-1|x_t)
        x_0_pred_logits = x_0_pred_logits.unsqueeze(1)
        p_posterior = self.backward_probs(
            x_t, t, x_0_pred_logits.softmax(-1), q_posterior_over_0
        )
        # compute KL(q(x_t-1|x_t, x_0)||p(x_t-1|x_t))
        kl = utils.categorical_kl_probs(
            q_posterior_true, p_posterior
        ).squeeze()  # TODO: here they divide with log(2) in D3PM code, Why?

        nll = -utils.categorical_log_likelihood(
            x_0, x_0_pred_logits
        ).squeeze()  # TODO: They divide here as well with log(2)
        vb_loss = torch.where(t < 2.0, nll, kl)
        vb_loss = pyg.utils.scatter(
            vb_loss, batch, -1, dim_size=num_samples, reduce="mean"
        )  # mean over all variables
        assert len(vb_loss.shape) == 1 and vb_loss.shape[0] == num_samples
        return vb_loss.mean()


@registry.register_d3pm_transition("marginal")
class MarginalD3PMTransition(D3PMTransition):
    def __init__(self, t_max, variable_dim, device, marginal):
        self.init_marginal(marginal, variable_dim, device)
        assert self.m.shape == (1, variable_dim), self.m.shape
        super().__init__(t_max, variable_dim, device)

    def init_marginal(self, marginal, variable_dim, device):
        marginal = marginal.flatten()
        if marginal.shape[0] > variable_dim:
            print(
                f"Please notice: Marginal distribution has been computed for a variable of size {marginal.shape[-1]}, but desired variable dim is {variable_dim}. Will use the first {variable_dim} entries of marginal and renormalize"
            )
            marginal = F.normalize(marginal[:variable_dim], p=1, dim=-1)
        elif marginal.shape[0] < variable_dim:
            print(
                f"Please notice: Marginal distribution has been computed for a variable of size {marginal.shape[-1]}, but desired variable dim is {variable_dim}. Will add additional zeros to end of marginal distribution"
            )

            marginal = F.pad(marginal, (0, variable_dim - marginal.shape[0]), value=0)
        self.m = marginal.reshape(1, variable_dim).to(device)
        return

    def get_betas(self):
        steps = torch.arange(self.t_max + 1).float() / self.t_max
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas

    def get_Qt_single(self, t):
        beta_t = self.betas[t]
        Q_t = (1 - beta_t) * torch.eye(
            self.variable_dim, device=self.device
        ) + beta_t * self.m.repeat(self.variable_dim, 1)
        return Q_t

    def sample_from_prior(self, num_vars):
        # num_vars: total number of variables x_i to be sampled (batch_size x D where D is the dimension of the
        p_T = self.m.repeat(num_vars, 1)
        x_T = utils.sample_from_distr(p_T)
        return x_T


@registry.register_d3pm_transition("uniform")
class UniformD3PMTransition(MarginalD3PMTransition):
    def init_marginal(self, marginal, variable_dim, device):
        self.m = 1 / variable_dim * torch.ones((1, variable_dim), device=device)
        return


@registry.register_d3pm_transition("zeros_init")
class ZerosInitD3PMTransition(MarginalD3PMTransition):
    def init_marginal(self, marginal, variable_dim, device):
        self.m = torch.zeros((variable_dim), device=device)
        self.m[0] = 1.0
        self.m = self.m.unsqueeze(0)
        return


@registry.register_model("d3pm")
class D3PM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.t_max = config["t_max"]
        self.num_elements = config["num_elements"]
        self.max_num_atoms = config["max_num_atoms"]
        self.dataset_info = registry.get_datainfo_class(config["dataset"])()
        self.transition_model_zero = registry.get_d3pm_transition_class(
            config["d3pm_transition"]
        )(
            self.t_max,
            config["num_elements"] + 1,
            self.device,
            self.dataset_info.zero_dof_distr,
        )  # + 1 to include 0
        self.transition_model_inf = registry.get_d3pm_transition_class(
            config["d3pm_transition"]
        )(
            self.t_max,
            config["max_num_atoms"] + 1,
            self.device,
            self.dataset_info.inf_dof_distr,
        )  # max_num_atoms=i means we can have 0, 1, ..., i number of atoms, so +1 to account for 0
        self.model = registry.get_gnn_class(config["gnn"])(config)
        self.spg_distr = torch.distributions.Categorical(
            probs=self.dataset_info.spg_distr
        )
        self.verbose_losses = config["verbose_losses"]

        loss_fn = config["loss_fn"]
        if loss_fn is not None:
            if loss_fn == "kl":
                self.training_losses = self.vb_loss
            elif loss_fn == "ce":
                self.training_losses = self.ce_loss
            elif loss_fn == "hybrid":
                self.hybrid_lambda = config["hybrid_lambda"]
                assert self.hybrid_lambda is not None, "hybrid lambda is not set"
                self.training_losses = self.hybrid_loss
            else:
                raise ValueError(f"Loss function {loss_fn} not found")
        else:
            self.training_losses = self.no_loss_defined

    # TODO: Where should there be a sum, and where should there be a mean over variables? Or should mean also be used, but sometimes upweight explicitly?
    # TODO: Maybe it should only matter when comparing the 0 dof loss vs inf dof loss?
    def vb_loss(self, x0_0dof_pred_logits, x0_infdof_pred_logits, x_0, x_t, t):
        zero_dof_loss = self.transition_model_zero.vb_loss(
            x0_0dof_pred_logits,
            x_0.x_0_dof.unsqueeze(1),
            x_t.x_0_dof.unsqueeze(1),
            torch.repeat_interleave(t, x_t.num_0_dof),
            x_t.batch[x_t.zero_dof],
            len(x_t.num_0_dof),
        )
        inf_dof_loss = self.transition_model_inf.vb_loss(
            x0_infdof_pred_logits.flatten(0, 1),
            x_0.x_inf_dof.reshape(-1, 1),
            x_t.x_inf_dof.reshape(-1, 1),
            torch.repeat_interleave(t, self.num_elements * x_t.num_inf_dof),
            torch.repeat_interleave(x_t.batch[~x_t.zero_dof], self.num_elements),
            len(x_t.num_inf_dof),
        )

        if self.verbose_losses:
            # TODO: implement verbose of vb loss
            verbosed_losses = None
        else:
            verbosed_losses = None

        return zero_dof_loss, inf_dof_loss, verbosed_losses

    def ce_loss(self, x0_0dof_pred_logits, x0_infdof_pred_logits, x_0, x_t, t):
        # TODO: should there be sum, mean, or a mixture between the two?
        # Currently:
        # For inf dof, we compute the sum over each atom type, then sum all the different positions
        # For 0 dof, we sum over each position
        # This gives one loss term per position, and this is divided by the number of **variables**
        # Finally, take the mean over the batch
        # Essentially, this treats all variables equally

        # Mathematically H(p,q) cross-entropy
        nll_0dof = -utils.categorical_log_likelihood(
            x_0.x_0_dof.long(), x0_0dof_pred_logits
        )
        # Sum along all variables, 1 variable for 0-DoF and num_elements variables for inf-DoF.
        nll_0dof = pyg.utils.scatter(
            nll_0dof,
            x_t.batch[x_t.zero_dof],
            dim=0,
            dim_size=len(x_t.num_0_dof),
            reduce="sum",
        )
        # Ensure there is a loss for each material. There is one material for each time step t, so the shapes should be equal.
        assert nll_0dof.shape == t.shape
        # Average the sum of the losses with the amount of variables for 0-Dof and inf-Dof.
        nll_0dof = nll_0dof / (x_t.num_0_dof + x_t.num_inf_dof * self.num_elements)

        # Mathematically H(p,q) cross-entropy
        nll_infdof = -utils.categorical_log_likelihood(
            x_0.x_inf_dof, x0_infdof_pred_logits
        ).sum(
            dim=1
        )  # sum over the different elements
        nll_infdof = pyg.utils.scatter(
            nll_infdof,
            x_t.batch[~x_t.zero_dof],
            dim=0,
            dim_size=len(x_t.num_inf_dof),
            reduce="sum",
        )
        assert nll_infdof.shape == t.shape
        nll_infdof = nll_infdof / (x_t.num_0_dof + x_t.num_inf_dof * self.num_elements)

        if self.verbose_losses:

            # Pure guess of the q probability distribution. Size of the logits tensor.
            # print(f"Shape x0_0dof_pred_logits {x0_0dof_pred_logits.size()[0:-1]}")
            q_0_zero_dof = torch.ones(
                x0_0dof_pred_logits.size(), device=self.device
            ) * (1 / x0_0dof_pred_logits.size(dim=-1))
            # print(f"Shape q_0_zero_dof {q_0_zero_dof}")
            q_0_inf_dof = torch.ones(
                x0_infdof_pred_logits.size(), device=self.device
            ) * (1 / x0_infdof_pred_logits.size(dim=-1))

            # Mathematically D_kl(p,q) when q is the estimated probabilities, in this case, model predicts logits that is converted to probability distribution, q, where all elements sum to 1.
            # One hot encode x matrix to get the target probability distribution
            Dkl_pq_zero_dof = utils.categorical_kl_probs_logits(
                probs=F.one_hot(x_0.x_0_dof.long(), x0_0dof_pred_logits.shape[-1]),
                logits=x0_0dof_pred_logits,
            )
            Dkl_pq_inf_dof = utils.categorical_kl_probs_logits(
                probs=F.one_hot(x_0.x_inf_dof, x0_infdof_pred_logits.shape[-1]),
                logits=x0_infdof_pred_logits,
            ).sum(dim=1)
            # Scatter sum over all variables in the batch, 1 variable for 0-DoF elements, and num_elements variables for inf-DoF.
            Dkl_pq_zero_dof = pyg.utils.scatter(
                Dkl_pq_zero_dof,
                x_t.batch[x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_0_dof),
                reduce="sum",
            )
            Dkl_pq_inf_dof = pyg.utils.scatter(
                Dkl_pq_inf_dof,
                x_t.batch[~x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_inf_dof),
                reduce="sum",
            )
            # Average the scattered sum over all variables
            assert Dkl_pq_zero_dof.shape == t.shape
            Dkl_pq_zero_dof = Dkl_pq_zero_dof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )
            assert Dkl_pq_inf_dof.shape == t.shape
            Dkl_pq_inf_dof = Dkl_pq_inf_dof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )

            # Matematically, D_kl(p,q_0)
            Dkl_pq0_zero_dof = utils.categorical_kl_probs(
                probs1=F.one_hot(x_0.x_0_dof.long(), x0_0dof_pred_logits.shape[-1]),
                probs2=q_0_zero_dof,
            )
            Dkl_pq0_inf_dof = utils.categorical_kl_probs(
                probs1=F.one_hot(x_0.x_inf_dof, x0_infdof_pred_logits.shape[-1]),
                probs2=q_0_inf_dof,
            ).sum(dim=1)
            # Scatter for batch
            Dkl_pq0_zero_dof = pyg.utils.scatter(
                Dkl_pq0_zero_dof,
                x_t.batch[x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_0_dof),
                reduce="sum",
            )
            Dkl_pq0_inf_dof = pyg.utils.scatter(
                Dkl_pq0_inf_dof,
                x_t.batch[~x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_inf_dof),
                reduce="sum",
            )
            assert Dkl_pq0_zero_dof.shape == t.shape
            Dkl_pq0_zero_dof = Dkl_pq0_zero_dof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )
            assert Dkl_pq0_inf_dof.shape == t.shape
            Dkl_pq0_inf_dof = Dkl_pq0_inf_dof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )

            # KL Divergence ratio between KL divergence for predicted probablity q and KL divergence for pure guess q_0.
            # Normalized in a sense to an interval between 0-1 for relevant probabilities of q. >1 if q is worse prediction than q_0.
            # Add small epsilon for numerical stability.
            zero_dof_Dkl_ratio = Dkl_pq_zero_dof / (Dkl_pq0_zero_dof + 1e-6)
            inf_dof_Dkl_ratio = Dkl_pq_inf_dof / (Dkl_pq0_inf_dof + 1e-6)

            # For reference can have what the CE is for pure guess of q, i.e., q_0
            # Mathematically H(p,q) cross-entropy
            pure_guess_nll_0dof = -utils.categorical_log_likelihood_pure_guess(
                x_0.x_0_dof.long(), q_0_zero_dof
            )
            pure_guess_nll_0dof = pyg.utils.scatter(
                pure_guess_nll_0dof,
                x_t.batch[x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_0_dof),
                reduce="sum",
            )
            assert pure_guess_nll_0dof.shape == t.shape
            pure_guess_nll_0dof = pure_guess_nll_0dof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )

            # Mathematically H(p,q) cross-entropy
            pure_guess_nll_infdof = -utils.categorical_log_likelihood_pure_guess(
                x_0.x_inf_dof, q_0_inf_dof
            ).sum(
                dim=1
            )  # sum over the different elements
            pure_guess_nll_infdof = pyg.utils.scatter(
                pure_guess_nll_infdof,
                x_t.batch[~x_t.zero_dof],
                dim=0,
                dim_size=len(x_t.num_inf_dof),
                reduce="sum",
            )
            assert pure_guess_nll_infdof.shape == t.shape
            pure_guess_nll_infdof = pure_guess_nll_infdof / (
                x_t.num_0_dof + x_t.num_inf_dof * self.num_elements
            )

            verbosed_losses = {
                "Dkl_pq_zero_dof": Dkl_pq_zero_dof.mean(),
                "Dkl_pq_inf_dof": Dkl_pq_inf_dof.mean(),
                "zero_dof_Dkl_ratio": zero_dof_Dkl_ratio.mean(),
                "inf_dof_Dkl_ratio": inf_dof_Dkl_ratio.mean(),
                "Dkl_pq0_zero_dof": Dkl_pq0_zero_dof.mean(),
                "Dkl_pq0_inf_dof": Dkl_pq0_inf_dof.mean(),
                "pure_guess_nll_0dof": pure_guess_nll_0dof.mean(),
                "pure_guess_nll_infdof": pure_guess_nll_infdof.mean(),
            }

        else:
            verbosed_losses = None

        # take the mean over the batch
        return nll_0dof.mean(), nll_infdof.mean(), verbosed_losses

    def hybrid_loss(self, x_0_pred_logits, x_0, x_t, t):
        kl_loss = self.vb_loss(x_0_pred_logits, x_0, x_t, t)
        ce_loss = self.ce_loss(x_0_pred_logits, x_0, x_t, t)
        return kl_loss + self.hybrid_lambda * ce_loss

    def no_loss_defined(self):
        raise RuntimeError("A loss function was not defined")

    def get_noisy_data(self, data_0):
        data_t = data_0.clone()
        t = torch.randint(
            1, self.t_max, (data_t.space_group.shape[0],), device=self.device
        )
        t_0_dof = torch.repeat_interleave(t, data_0.num_0_dof)
        t_inf_dof = torch.repeat_interleave(t, data_0.num_inf_dof * self.num_elements)
        data_t.x = None
        data_t.x_inf_dof = self.transition_model_inf.sample_q_t_0(
            data_0.x_inf_dof.flatten(), t_inf_dof
        ).reshape(data_0.x_inf_dof.shape)
        data_t.x_0_dof = self.transition_model_zero.sample_q_t_0(
            data_0.x_0_dof.flatten(), t_0_dof
        ).reshape(data_0.x_0_dof.shape)
        assert data_t.x_0_dof.shape == data_0.x_0_dof.shape
        assert data_t.x_inf_dof.shape == data_0.x_inf_dof.shape
        return data_t, t

    def sample_from_prior(self, num_samples):
        samples = []
        space_groups = self.spg_distr.sample((num_samples,))
        for spg in space_groups:
            spg = int(spg)
            space_group = spg
            num_pos = len(lookup_tables.spg_wyckoff[str(spg)])
            multiplicities = torch.tensor(
                [
                    value
                    for key, value in reversed(
                        lookup_tables.spg_wyckoff_multiplicities[str(spg)].items()
                    )
                ]
            )
            dof = torch.tensor(
                [
                    value
                    for key, value in reversed(
                        lookup_tables.spg_wyckoff_degrees_of_freedom[str(spg)].items()
                    )
                ]
            ).long()
            zero_dof = ~(dof.bool())
            num_0_dof = torch.sum(zero_dof)
            num_inf_dof = torch.sum(~zero_dof)
            x_0_dof = self.transition_model_zero.sample_from_prior(num_0_dof).squeeze(1)
            x_inf_dof = self.transition_model_inf.sample_from_prior(
                num_inf_dof * self.num_elements
            ).reshape(num_inf_dof, self.num_elements)
            x = model_utils.create_x_matrix(x_inf_dof, x_0_dof, zero_dof)
            wyckoff_pos_idx = torch.arange(num_pos).long()
            e_i = torch.arange(num_pos).repeat_interleave(num_pos)
            e_j = torch.arange(num_pos).repeat(num_pos)
            edge_index = torch.stack([e_i, e_j])

            samples.append(
                pyg.data.Data(
                    space_group=torch.tensor(space_group),
                    x_0_dof=x_0_dof,
                    x_inf_dof=x_inf_dof,
                    edge_index=edge_index,
                    zero_dof=zero_dof,
                    wyckoff_pos_idx=wyckoff_pos_idx,
                    num_pos=torch.tensor([num_pos]),
                    num_nodes=torch.tensor([num_pos]),
                    num_0_dof=num_0_dof,
                    num_inf_dof=num_inf_dof,
                    multiplicities=multiplicities,
                    x=x,
                    degrees_of_freedom=dof,
                )
            )
        return pyg.data.batch.Batch.from_data_list(samples).to(self.device)

    @torch.inference_mode()
    def generate(self, num_samples):
        assert not self.model.training, "Model is in training mode"
        samples = []
        print(f"Generating in total {num_samples} materials with D3PM")
        num_samples_list = model_utils.split_number(num_samples, 750)
        for num_samples in num_samples_list:
            print(f"Now generating {num_samples} materials in this batch")
            data_t = self.sample_from_prior(num_samples)
            for t in reversed(range(1, self.t_max)):
                t = t * torch.ones(num_samples, device=self.device).long()
                p_x0_logits_0_dof, p_x0_logits_inf_dof = self.x0_distr(data_t, t)
                x_t_0_dof = self.transition_model_zero.backward_step(
                    data_t.x_0_dof,
                    t.repeat_interleave(data_t.num_0_dof),
                    p_x0_logits_0_dof.softmax(-1),
                )
                x_t_inf_dof = self.transition_model_inf.backward_step(
                    data_t.x_inf_dof,
                    t.repeat_interleave(data_t.num_inf_dof),
                    p_x0_logits_inf_dof.softmax(-1),
                )
                data_t.x_0_dof = x_t_0_dof
                data_t.x_inf_dof = x_t_inf_dof
                data_t.x = model_utils.create_x_matrix(
                    x_t_inf_dof, x_t_0_dof, data_t.zero_dof
                )
            samples.extend(data_t.to_data_list())
        batch = pyg.data.batch.Batch.from_data_list(samples).to(self.device)
        return batch

    def x0_distr(self, x_t, t):
        """
        x_t: data object
        t is a tensor with timesteps (one per material)
        """
        return self.model(x_t, t)

    def forward(self, data_t, t):
        return self.x0_distr(data_t, t)
