import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel


class DeeplabModel(ISModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, is_att=True, is_fbi=True, net_mode='upatt',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

    def backbone_forward(self, image, coord_features=None, points=None):
        raise NotImplementedError

