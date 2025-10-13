from dataclasses import dataclass, field
from typing import Type

import torch
from torch import nn

from scrstudio.encoders.base_encoder import Encoder, EncoderConfig


@dataclass
class PCAEncoderConfig(EncoderConfig):

    _target: Type = field(default_factory=lambda: PCAEncoder)

    encoder: EncoderConfig = field(default_factory= EncoderConfig)

    pca_path: str = "pca.pth"




class PCAEncoder(Encoder):

    OUTPUT_SUBSAMPLE = 8
    def __init__(self, config,data_path=None, **kwargs):
        super().__init__(config)
        
        self.encoder = config.encoder.setup(data_path=data_path, **kwargs)
        self.preprocess = self.encoder.preprocess
        state_dict = torch.load(data_path /'proc' / config.pca_path, map_location='cpu',weights_only=True)
        self.out_channels=state_dict['weight'].shape[0]

        if self.encoder.sparse:
            self.linear=nn.Linear(self.encoder.out_channels,self.out_channels,bias='bias' in state_dict.keys())
            state_dict['weight']=state_dict['weight'].view(self.out_channels,self.encoder.out_channels)
            self.linear.load_state_dict(state_dict)
            self.linear.eval()
        else:
            self.conv=nn.Conv2d(self.encoder.out_channels,self.out_channels,1,
                                bias='bias' in state_dict.keys())
            self.conv.load_state_dict(state_dict)
            self.conv.eval()

    def keypoint_features(self, data, n=0,generator=None):
        if self.encoder.sparse:
            ret= self.encoder.keypoint_features(data, n,generator)
            ret['descriptors']=self.linear(ret['descriptors'])
            return ret
        else:
            return super().keypoint_features(data, n,generator)



    def forward(self, data,det=False):

        ret=self.encoder(data)
        ret['features']=self.conv(ret['features'])


        return ret

