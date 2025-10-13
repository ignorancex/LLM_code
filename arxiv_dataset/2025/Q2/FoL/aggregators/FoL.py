# -*- coding: UTF-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.unsqueeze(1)#.expand(b, 1, n)

    couplings = torch.cat([scores, bins], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class FoL(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(self,
                 num_channels=1536,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )


    def replace_top_values(self, important_matrix):
        num_change = 3
        top_values, top_indices = torch.topk(important_matrix, num_change, dim=1, largest=True, sorted=False)
        sorted_values, sorted_indices = important_matrix.sort(dim=1, descending=True)
        k_value_index = int(0.05 * important_matrix.size(1))
        kth_values = sorted_values[:, k_value_index].unsqueeze(1)
        kth_values_expanded = kth_values.expand(-1, num_change)
        important_matrix.scatter_(1, top_indices, kth_values_expanded)
        return important_matrix

    def forward(self, x, mask, test=False):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t, important_matrix = x[0], x[1], x[2]
        BS, C, H, W = x.shape
        local_f = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = log_optimal_transport(p, 1-mask, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]
        if test:
            weak_supervision_info = None
        else:
            weak_supervision_info = [nn.functional.normalize(x.reshape(BS, -1, 529), p=2, dim=1), p]

        # Calculate Confidence Matrix
        confidence_matrix = torch.mean(p, dim=1)
        important_matrix = self.replace_top_values(important_matrix)
        confidence_matrix = self.replace_top_values(confidence_matrix)

        confidence_matrix = confidence_matrix.softmax(dim=-1)
        important_matrix = important_matrix.softmax(dim=-1)

        loss_kl_1 = nn.KLDivLoss(reduction="none")(confidence_matrix.log(), important_matrix.detach()).sum(-1).mean()
        loss_kl_2 = nn.KLDivLoss(reduction="none")(important_matrix.log(), confidence_matrix.detach()).sum(-1).mean()

        if test:
            mix_matrix = confidence_matrix
        else:
            mix_matrix = confidence_matrix + important_matrix


        num_select = 225
        _, topk_indices = torch.topk(mix_matrix, num_select, dim=-1)
        mask = torch.zeros((BS, H * W), dtype=torch.float32, device=x.device)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view(BS, H, W)

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        
        return nn.functional.normalize(f, p=2, dim=-1), local_f, mask, loss_kl_1, loss_kl_2, mix_matrix, weak_supervision_info
