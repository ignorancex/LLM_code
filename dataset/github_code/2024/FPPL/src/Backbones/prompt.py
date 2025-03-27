import copy
import sys

import torch
import torch.nn as nn


class FedPrompt(nn.Module):
    def __init__(
        self,
        n_task,
        length=5,
        pool_size=4,
        num_layers=5,
        embed_dim=768,
        prompt_init="uniform",
    ):
        super().__init__()

        self.n_task = n_task
        self.length = length
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.prompt_init = prompt_init

        prompt_pool_shape = (
            self.n_task,
            self.num_layers,
            self.pool_size,
            self.length,
            embed_dim,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)

    def forward(self, weights, cur_task=-1, train=False):
        # freeze/control past tasks
        if train:
            prompts = [
                (
                    self.prompt[task]
                    if task == cur_task
                    else self.prompt[task].detach().clone()
                )
                for task in range(self.n_task)
            ]
            p = torch.stack(prompts, dim=0)
        else:
            p = self.prompt

        fused_prompt = torch.einsum("bt,tnpld->bnpld", weights, p)
        fused_prompt = fused_prompt.view(
            fused_prompt.shape[0], fused_prompt.shape[1], -1, fused_prompt.shape[-1]
        )
        return fused_prompt
