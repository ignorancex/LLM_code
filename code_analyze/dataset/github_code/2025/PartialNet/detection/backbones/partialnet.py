from mmdet.models.builder import BACKBONES as det_BACKBONES
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parent_parentdir)
from models.partialnet import PartialNet


@det_BACKBONES.register_module()
def PartialNet_s(**kwargs):
    model = PartialNet(
        mlp_ratio=2.0,
        embed_dim=128,
        depths=(1, 2, 13, 2),
        drop_path_rate=0.15,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )
    return model

@det_BACKBONES.register_module()
def PartialNet_m(**kwargs):
    model = PartialNet(
        mlp_ratio=2.0,
        embed_dim=144,
        depths=(3, 4, 18, 3),
        drop_path_rate=0.2,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )
    return model

@det_BACKBONES.register_module()
def PartialNet_l(**kwargs):
    model = PartialNet(
        mlp_ratio=2.0,
        embed_dim=192,
        depths=(3, 4, 18, 3),
        drop_path_rate=0.3,
        act_layer='RELU',
        fork_feat=True,
        **kwargs
        )
    return model
