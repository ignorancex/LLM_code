# Copyright 2018 Kornia Team
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
#
# Modified by Xudong Jiang (ETH Zurich)

from dataclasses import dataclass, field
from typing import Type

import torch
from kornia.feature.loftr.backbone.resnet_fpn import ResNetFPN_8_2

from scrstudio.encoders.base_encoder import Encoder, EncoderConfig, PreprocessConfig

urls = {
    "loftr_outdoor": "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt",
    "loftr_indoor_new": "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt",
    "loftr_indoor": "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt",
}

@dataclass
class LoFTREncoderConfig(EncoderConfig):

    _target: Type = field(default_factory=lambda: LoFTREncoder)
    
    model: str = "loftr_outdoor"



class LoFTREncoder(Encoder):
    """
    Loftr encoder, used to extract features from the input images.
    """
    OUTPUT_SUBSAMPLE = 8
    out_channels = 256
    def __init__(self, config: LoFTREncoderConfig, **kwargs):
        super().__init__(config)
        self.preprocess = PreprocessConfig(mean=None, std=None, grayscale=True, use_half=True, size_multiple=8)
        
        backbone = ResNetFPN_8_2({"initial_dim": 128, "block_dims": [128, 196, 256]})
        state_dict = torch.hub.load_state_dict_from_url(urls[config.model], map_location="cpu")['state_dict']
        state_dict = {k.split('backbone.')[1]: v for k, v in state_dict.items() if 'backbone.' in k}
        backbone.load_state_dict(state_dict)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.eval()

    def forward(self, data,det=False):
        x=data['image']
        
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return {
            "features": x3
        }

