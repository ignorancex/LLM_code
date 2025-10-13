import torch
import torch.nn as nn
from ..configs import TrainerConfig, StrainerConfig, SamplingType, LMCConfig
from typing import Type


class MSELoss:
    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, cfg: SamplingType = None
    ):
        if isinstance(cfg, LMCConfig):
            diff = input - target
            imp_loss = torch.abs(diff).mean(-1)
            eps = torch.finfo(torch.float16).eps
            self.correction = 1.0 / torch.clip(imp_loss, min=eps).detach()
            # the stopgradient operator is detach
            # if cfg.alpha != 0:
            #     # Warmup in section 3.3 of LMC paper
            #     r = min((iter / 1000), cfg.alpha)
            # else:
            #     r = cfg.alpha
            r = cfg.alpha  # Trying without warmup for now
            self.correction.pow_(r)
            self.correction.clamp_(
                min=0.2,
                max=self.correction.mean().item() + self.correction.std().item(),
            )
            self.loss_per_pix = (diff**2).mean(-1)
            self.loss_per_pix.mul_(self.correction)
        else:
            dims = tuple(range(2, input.ndim))
            self.loss_per_pix = ((input - target) ** 2).mean(dims)
        return self.loss_per_pix.mean()


class MAELoss:
    def __call__(
        self, input: torch.Tensor, target: torch.Tensor, cfg: SamplingType = None
    ):
        dims = tuple(range(2, input.ndim))
        self.loss_per_pix = torch.abs(input - target).mean(dims)
        return self.loss_per_pix.mean()


# class LatentAlignLoss(nn.Module):
#     def __init__(self):
#         super(LatentAlignLoss, self).__init__()

#     def forward(self, latents, align_feats):
#         # We should normalize in cache?
#         align_feats = align_feats / align_feats.norm(dim=-1, keepdim=True)
#         self.loss = 1 - torch.cosine_similarity(latents, align_feats, dim=-1).mean()

#         return self.loss


# class rate_loss(nn.Module):
#     def __init__(self):
#         super(rate_loss, self).__init__()
#         self.loss = {}

#     def forward(self, input, target, weights, mu=0, sigma=1):
#         sigma_tensor = torch.tensor(sigma, dtype=weights.dtype, device=weights.device)
#         self.loss = (
#             0.5 * torch.log(2 * torch.pi * sigma_tensor**2)
#             + 0.5 * ((weights - mu) / sigma_tensor) ** 2
#         )
#         return self.loss.mean()


LOSSES: dict[str, Type[MSELoss] | Type[MAELoss]] = {
    "mse": MSELoss,
    "l1": MAELoss,
}


# def self_information(weight, prob_model, is_single_model=False, is_val=False, g=None):
#     weight = (
#         (weight + torch.rand(weight.shape, generator=g).to(weight) - 0.5)
#         if not is_val
#         else torch.round(weight)
#     )
#     weight_p = weight + 0.5
#     weight_n = weight - 0.5
#     if not is_single_model:
#         prob = prob_model(weight_p) - prob_model(weight_n)
#     else:
#         prob = prob_model(weight_p.reshape(-1, 1)) - prob_model(weight_n.reshape(-1, 1))
#     total_bits = torch.sum(
#         torch.clamp(-1.0 * torch.log(prob + 1e-10) / torch.log(2.0), 0, 50)
#     )
#     return total_bits, prob


# class entropy_reg(nn.Module):
#     def __init__(self):
#         super(entropy_reg, self).__init__()
#         self.loss = {}

#     def forward(self, latents, prob_models, single_prob_model, lambda_loss):
#         bits = num_elems = 0
#         for group_name in latents:
#             if torch.any(torch.isnan(latents[group_name])):
#                 raise Exception("Weights are NaNs")
#             cur_bits, prob = self_information(
#                 latents[group_name],
#                 prob_models[group_name],
#                 single_prob_model,
#                 is_val=False,
#             )
#             bits += cur_bits
#             num_elems += prob.size(0)
#         self.loss = (
#             bits / num_elems * lambda_loss
#         )  # {'ent_loss': bits/num_elems*lambda_loss}
#         return self.loss, bits.float().item() / 8


class CompositeLoss(nn.Module):
    def __init__(self, cfg: TrainerConfig | StrainerConfig):
        super(CompositeLoss, self).__init__()

        self.loss_names: tuple[str]
        self.loss_names, lambdas = zip(*cfg.losses)
        self.lambdas = torch.tensor(lambdas, device="cuda")
        self.cfg = cfg

        self.losses = [LOSSES[x]() for x in self.loss_names]

    @property
    def loss_per_pix(self) -> torch.Tensor:
        loss_per_pix = self.lambdas[0] * self.losses[0].loss_per_pix
        for lmbda, loss in zip(self.lambdas[1:], self.losses[1:]):
            loss_per_pix += lmbda * loss.loss_per_pix
        return loss_per_pix

    def forward(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        cfg: SamplingType | None = None,
    ):
        model_outputs = outputs
        all_losses = []
        for loss_fn in self.losses:
            cur_loss: torch.Tensor | tuple[torch.Tensor] = loss_fn(
                model_outputs, target, cfg
            )
            all_losses.append(cur_loss)

        losses = torch.stack(all_losses)
        weighted_losses = self.lambdas * losses
        total_loss = weighted_losses.sum()
        return total_loss
