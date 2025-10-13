# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scrstudio.configs.base_config import InstantiateConfig, PrintableConfig


@dataclass
class BlockListConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BlockList)
    blocks: List[Tuple[InstantiateConfig,int]] = field(default_factory=lambda: [])

class BlockList(nn.Sequential):

    def __init__(self, config: BlockListConfig, **kwargs):
        super().__init__()
        self.config = config
        for block_config, num in self.config.blocks:
            for _ in range(num):
                block=block_config.setup(**kwargs)
                self.append(block)


@dataclass
class ResBlockConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: ResBlock)
    mlp_ratio: Optional[float] = None
    input_name: str = "features"
    output_name: str = "features"

class ResBlock(nn.Module):
    def __init__(self, config: ResBlockConfig,
                    head_channels: int = 768,
                    mlp_ratio: float = 2.0,
                    **kwargs
                 ):
        super().__init__()
        self.config = config
        self.head_channels = head_channels
        self.mlp_ratio = mlp_ratio if config.mlp_ratio is None else config.mlp_ratio
        block_channels = int(self.head_channels * self.mlp_ratio)
        self.c0 = nn.Linear(self.head_channels, self.head_channels)
        self.c1 = nn.Linear(self.head_channels, block_channels)
        self.c2 = nn.Linear(block_channels, self.head_channels)


    def forward(self, data):
        x = data[self.config.input_name]
        res = F.relu(self.c0(x))
        res = F.relu(self.c1(res))
        res = F.relu(self.c2(res))
        x = x + res

        data[self.config.output_name] = x
        return data
    
@dataclass
class InputBlockConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: InputBlock)
    in_channels: Optional[int] = None
    head_channels: Optional[int] = None
    mlp_ratio: Optional[float] = None
    input_name: str = "features"
    output_name: str = "features"

class InputBlock(nn.Module):

    def __init__(self, config: InputBlockConfig,
                    in_channels: int = 512,
                    head_channels: int = 768,
                    mlp_ratio: float = 2.0,
                    **kwargs
                 ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels if config.in_channels is None else config.in_channels
        self.head_channels = head_channels if config.head_channels is None else config.head_channels
        self.mlp_ratio = mlp_ratio if config.mlp_ratio is None else config.mlp_ratio
        block_channels = int(self.head_channels * self.mlp_ratio)
        self.head_skip = nn.Linear(self.in_channels, self.head_channels) if self.in_channels > self.head_channels else None
        self.c0 = nn.Linear(self.in_channels, self.head_channels)
        self.c1 = nn.Linear(self.head_channels, block_channels)
        self.c2 = nn.Linear(block_channels, self.head_channels)

    def forward(self, data):
        x = data[self.config.input_name]
        res = F.relu(self.c0(x))
        res = F.relu(self.c1(res))
        res = F.relu(self.c2(res))

        if self.head_skip is not None:
            ret = self.head_skip(x) + res
        elif self.in_channels < self.head_channels:
            ret = res.clone()
            ret[..., :self.in_channels] += x
        else:
            ret =  res + x


        data[self.config.output_name] = ret

        return data
@dataclass
class PositionEncoderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PositionEncoder)
    input_name: str = "sc0"
    output_name: str = "features"
    num_freqs: int = 10
    max_freq_exp: int = 4
    period: float = 1.0

class PositionEncoder(nn.Module):

    def __init__(self, config: PositionEncoderConfig,
                    **kwargs):
        super().__init__()
        self.config = config
        self.inv_scale = 2 * math.pi / config.period
        

    def forward(self, data):
        sc = data[self.config.input_name] # ..., 3

        sc = sc * self.inv_scale

        freqs = 2 ** torch.linspace(0, self.config.max_freq_exp, self.config.num_freqs, device=sc.device)
        sc = sc.unsqueeze(-1) * freqs # ..., 3, num_freqs
        sc = torch.cat([sc.sin(), sc.cos()], dim=-1) # ..., 3, 2 * num_freqs
        sc = sc.flatten(-2) # ..., 6 * num_freqs
        output = data[self.config.output_name].clone()
        output[..., output.shape[-1]-sc.shape[-1]:] += sc  
        data[self.config.output_name] = output   
        return data

@dataclass
class HomoConfig(PrintableConfig):
    min_scale: float = 0.01
    max_scale: float = 4.0
    
@dataclass
class PositionDecoderConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: PositionDecoder)
    mlp_ratio: Optional[float] = None
    input_name: str = "features"
    output_name: str = "sc"
    center_name: str = "cluster_centers"
    homo: Optional[HomoConfig] = field(default_factory=lambda: HomoConfig())

class PositionDecoder(nn.Module):
    centers: Tensor

    def __init__(self, config: PositionDecoderConfig,
                    head_channels: int = 768,
                    mlp_ratio: float = 2.0,
                    metadata: Optional[Dict] = None,
                    **kwargs
                 ):
        super().__init__()
        self.config = config
        self.head_channels = head_channels
        self.mlp_ratio = mlp_ratio if config.mlp_ratio is None else config.mlp_ratio
        block_channels = int(self.head_channels * self.mlp_ratio)
        self.c0 = nn.Linear(self.head_channels, self.head_channels)
        self.c1 = nn.Linear(self.head_channels, block_channels)
        if config.homo:
            self.c2 = nn.Linear(block_channels, 4)
            self.max_inv_scale=1. / config.homo.max_scale
            self.h_beta=math.log(2) / (1. - self.max_inv_scale)
            self.min_inv_scale=1. / config.homo.min_scale
            
        else:
            self.c2 = nn.Linear(block_channels, 3)
        if metadata is not None:
            centers=metadata[config.center_name]
        else:
            print("No metadata provided, using default centers")
            centers=torch.zeros(1,3)
        self.register_buffer("centers", centers.clone().detach().view(1, centers.shape[0],3))
        self.multi_mean = centers.shape[0] > 1
        if self.multi_mean:
            self.cc = nn.Linear(block_channels, centers.shape[0])

    def forward(self, data):
        feat = data[self.config.input_name]
        feat = F.relu(self.c0(feat))
        feat = F.relu(self.c1(feat))
        if self.multi_mean:
            logits = self.cc(feat) # B x C
            probs = F.softmax(logits, dim=-1).unsqueeze(-1) # B x C x 1
            mean = torch.sum(probs * self.centers, dim=1, keepdim=False)
        else:
            mean=self.centers.squeeze()
        offset = self.c2(feat)
        if self.config.homo:
            h_slice = F.softplus(offset[:, 3].unsqueeze(1), beta=self.h_beta) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            offset = offset[:, :3] / h_slice

        data[self.config.output_name] = offset + mean

        return data

@dataclass
class PositionRefinerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PositionRefiner)
    input_name: str = "features"
    base_name: str = "sc0"
    output_name: str = "sc"
    mlp_ratio: Optional[float] = None
    homo: Optional[HomoConfig] = field(default_factory=lambda: HomoConfig())

class PositionRefiner(nn.Module):

    def __init__(self, config: PositionRefinerConfig,
                    head_channels: int = 768,
                    mlp_ratio: float = 2.0,
                    **kwargs):
        super().__init__()
        self.config = config
        self.head_channels = head_channels
        self.mlp_ratio = mlp_ratio if config.mlp_ratio is None else config.mlp_ratio
        block_channels = int(self.head_channels * self.mlp_ratio)
        self.c0 = nn.Linear(self.head_channels, self.head_channels)
        self.c1 = nn.Linear(self.head_channels, block_channels)
        if config.homo:
            self.c2 = nn.Linear(block_channels, 4)
            self.max_inv_scale=1. / config.homo.max_scale
            self.h_beta=math.log(2) / (1. - self.max_inv_scale)
            self.min_inv_scale=1. / config.homo.min_scale
            
        else:
            self.c2 = nn.Linear(block_channels, 3)

    def forward(self, data):
        feat = data[self.config.input_name]
        base = data[self.config.base_name]

        feat = F.relu(self.c0(feat))
        feat = F.relu(self.c1(feat))

        offset = self.c2(feat)
        if self.config.homo:
            h_slice = F.softplus(offset[:, 3].unsqueeze(1), beta=self.h_beta) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            offset = offset[:, :3] / h_slice

        data[self.config.output_name] = offset + base

        return data
        


