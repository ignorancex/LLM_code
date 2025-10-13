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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from kornia.feature.dedode.dedode_models import get_descriptor, get_detector
from kornia.feature.dedode.utils import dedode_denormalize_pixel_coordinates, sample_keypoints

from scrstudio.encoders.base_encoder import Encoder, EncoderConfig, PreprocessConfig


@dataclass
class DedodeEncoderConfig(EncoderConfig):

    _target: Type = field(default_factory=lambda: DedodeEncoder)

    detector:str = "L"
    
    descriptor:str = "B"
    
    k: int = 5000
    
urls = {
    "detector": {
        "L-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
        "Lv2-upright": "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
        "L-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth",
        "L-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth",
    },
    "descriptor": {
        "B-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
        "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
        "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
        "G-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
        "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth",
    },
}


class DedodeEncoder(Encoder):
    """
    Dedode encoder, used to extract features from the input images.
    """
    out_channels = 256 
    sparse = True
    def __init__(self, config: DedodeEncoderConfig, **kwargs):
        super().__init__(config)
        self.preprocess = PreprocessConfig(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225), 
                                            grayscale=False, use_half=True, 
                                            size_multiple=14 if config.descriptor[0] == "G" else 8)

    
        self.OUTPUT_SUBSAMPLE = 14 if config.descriptor[0] == "G" else 8
        self.detector = get_detector(config.detector[0])
        self.descriptor = get_descriptor(config.descriptor[0])

        self.detector.load_state_dict(
            torch.hub.load_state_dict_from_url(urls["detector"][f"{config.detector}-upright"], map_location="cpu")
        )
        self.descriptor.load_state_dict(
            torch.hub.load_state_dict_from_url(urls["descriptor"][f"{config.descriptor}-upright"], map_location="cpu")
        )

        self.k=config.k

        self.eval()


    def keypoint_features(self, data, n=0,generator=None):
        images = data['image']
        self.train(False)


        B, C, H, W = images.shape
        logits = self.detector.forward(images)
        scoremap = logits.reshape(B, H * W).softmax(dim=-1).reshape(B, H, W)
        if 'mask' in data:
            mask = TF.resize(data['mask'], [H, W], interpolation=TF.InterpolationMode.NEAREST).squeeze(1)
            scoremap = scoremap * mask

        keypoints, scores = sample_keypoints(scoremap, num_samples=self.k)

        if n > 0:
            idx = torch.randperm(keypoints.size(1), generator=generator,device=keypoints.device)[:n] if n <= keypoints.size(1) \
                                    else torch.randint(keypoints.size(1), (n,), generator=generator, device=keypoints.device)
            keypoints = keypoints[:, idx]
            scores = scores[:, idx]


           

        descriptions = self.descriptor.forward(images)
        descriptions = F.grid_sample(
            descriptions.float(), keypoints[:, None], mode="bilinear", align_corners=False
        )[:, :, 0].mT


        return {
            "keypoints": dedode_denormalize_pixel_coordinates(keypoints, H, W)[0],
            "keypoint_scores": scores[0],
            "descriptors": descriptions[0]
        }





    def forward(self, data,det=False):
        raise NotImplementedError

