"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import numpy as np
import fastmri
from .data import transforms
from torch import autograd

from .unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / (std + 1e-12), mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * (std + 1e-12) + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x



class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
        self.params = [param for param in self.parameters()]

    def forward(
        self,
        kspace_pred: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:

        #kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask)

        return fastmri.complex_abs(fastmri.ifft2c(kspace_pred))

    def estimate_fisher(self, dataset, sample_size, task_net,loss,batch_size=4):
        # Get loglikelihoods from data
        self.F_accum = []
        for v, _ in enumerate(self.params):
            self.F_accum.append(torch.zeros(list(self.params[v].size())).cuda())
        data_loader = dataset
        loglikelihoods = []
        count = 0
        for x, k,m,y,t in data_loader:
            #print(x.size(), y.size())
            #x = x.view(batch_size, -1)
            x = V(x).cuda() 
            k = V(k).cuda() 
            m = V(m).cuda()
            y = V(y).cuda()
            t = V(t).cuda()
            img_shape = x.shape
            if img_shape[1] > 1:
                m = m[0:1,:,:,:]
                k = torch.reshape(k,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))                     
            
            if task_net is None:
                loglikelihood = loss(self(k,k,m.byte()),t)
            else:
                if img_shape[1] > 1:
                    loglikelihood = loss(task_net(torch.reshape(self(k,k,m.byte()),(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)),t)
                else:
                    loglikelihood = loss(task_net(self(k,k,m.byte())),t)
            #loglikelihoods.append(loglikelihood)

            count += 1
            if count >= sample_size :
                break

            #loglikelihood = torch.cat(loglikelihoods).mean(0)
            loglikelihood_grads = autograd.grad(loglikelihood, self.parameters(),retain_graph=False)
            #print("FINISHED GRADING", len(loglikelihood_grads))
            for v in range(len(self.F_accum)):
                #print(len(self.F_accum))
                self.F_accum[v] = self.F_accum[v] + torch.pow(loglikelihood_grads[v], 2)

        for v in range(len(self.F_accum)):
            self.F_accum[v] /= sample_size

        parameter_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        #print("RETURNING", len(parameter_names))

        return {n: g for n, g in zip(parameter_names, self.F_accum)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            #print(dir(fisher[n].data))
            self.register_buffer('{}_estimated_fisher'.format(n), fisher[n])

    def ewc_loss(self, lamda, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(self, '{}_estimated_mean'.format(n))
                fisher = getattr(self, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in Vs.
                mean = V(mean)
                fisher = V(fisher.data)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                V(torch.zeros(1)).cuda() if cuda else
                V(torch.zeros(1))
            )
            
class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))


    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(torch.stack((mask,mask),4), current_kspace - ref_kspace, zero) * self.dc_weight

        model_term = fastmri.fft2c(self.model(fastmri.ifft2c(current_kspace)))

        return current_kspace - soft_dc - model_term
