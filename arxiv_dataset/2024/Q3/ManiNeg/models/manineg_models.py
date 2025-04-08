# allfixed version, no doubledp, no fuse norm
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18
import torch.distributed as dist
from transformers import AutoModel, BertModel

from functools import partial
# from ALBEF.models.vit import VisionTransformer, interpolate_pos_embed
# from ALBEF.models.xbert import BertConfig, BertForMaskedLM
# from HMBM.models.loss_function import NT_Xent

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ManinegIM(nn.Module):

    def __init__(self, drop_attr, vision_pretrained, **kwargs):
        super().__init__()

        self.visual_encoder = VisionEncoder(feature_dim=2048, proj_dim=128, pretrained=vision_pretrained)
        self.attr_encoder = MLP(proj_dim=128, output_dropout=drop_attr)
        self.projector = Projector(input_dim=128, output_dim=128)

    def forward(self, inputs):
        image_embeds1, image_proj_feat1 = self.visual_encoder(inputs['view1'])
        image_embeds2, image_proj_feat2 = self.visual_encoder(inputs['view2'])
        # text_output, text_proj_feat = self.text_encoder(inputs['input_ids'], inputs['attention_mask'])
        attr_embeds, attr_proj_feat = self.attr_encoder(inputs['attr'])
        all_feat = torch.cat([image_proj_feat1, image_proj_feat2, attr_proj_feat], dim=0)
        all_proj = self.projector(all_feat)
        image_proj_feat1, image_proj_feat2, attr_proj_feat = torch.split(all_proj, image_proj_feat1.size()[0])
        image_proj_feat1 = F.normalize(image_proj_feat1, dim=-1)
        image_proj_feat2 = F.normalize(image_proj_feat2, dim=-1)
        attr_proj_feat = F.normalize(attr_proj_feat, dim=-1)

        # keep output parameters the same as ViKL, so the experiment codes do not need to change
        return image_embeds1, image_proj_feat1, image_embeds2, image_proj_feat2, None, None, attr_embeds, attr_proj_feat


class ManinegII(nn.Module):

    def __init__(self, vision_pretrained, **kwargs):
        super().__init__()
        self.visual_encoder = VisionEncoder(feature_dim=2048, proj_dim=128, pretrained=vision_pretrained)
        self.projector = Projector(input_dim=128, output_dim=128)

    def forward(self, inputs):
        image_embeds1, image_proj_feat1 = self.visual_encoder(inputs['view1'])
        image_embeds2, image_proj_feat2 = self.visual_encoder(inputs['view2'])
        # text_output, text_proj_feat = self.text_encoder(inputs['input_ids'], inputs['attention_mask'])
        # attr_embeds, attr_proj_feat = self.attr_encoder(inputs['attr'])
        all_feat = torch.cat([image_proj_feat1, image_proj_feat2], dim=0)
        all_proj = self.projector(all_feat)
        image_proj_feat1, image_proj_feat2 = torch.split(all_proj, image_proj_feat1.size()[0])
        image_proj_feat1 = F.normalize(image_proj_feat1, dim=-1)
        image_proj_feat2 = F.normalize(image_proj_feat2, dim=-1)

        # keep output parameters the same as ViKL, so the experiment codes do not need to change
        return image_embeds1, image_proj_feat1, image_embeds2, image_proj_feat2, None, None, None, None


class Projector(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        # Todo: maybe needs a experiments to evaluate whether BN1d are necessary. Check later.
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, feature):
        return self.projector(feature)


class VisionEncoder(nn.Module):

    def __init__(self, feature_dim=512, proj_dim=128, pretrained=False):  # Todo: add in config
        # memo: we do not add dropout here, the main reason is that vision already has two view
        super(VisionEncoder, self).__init__()
        self.vision_encoder = resnet50(pretrained=pretrained)
        self.vision_encoder.fc = nn.Identity()
        self.vision_aligner = nn.Linear(feature_dim, out_features=proj_dim, bias=False)

    def forward(self, x):
        feature = self.vision_encoder(x)
        output = self.vision_aligner(feature)
        return feature, output


class MLP(nn.Module):
    """A multi-layer perceptron module.
    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.
    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    """

    def __init__(self, proj_dim=128, output_dropout=0.0) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(35, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.aligner = nn.Sequential(
            nn.Linear(512, proj_dim, bias=False),
            nn.Dropout(p=output_dropout) if output_dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.aligner(feat)

        return feat, out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import os

    os.environ['http_proxy'] = "http://172.17.146.34:8891"
    os.environ['https_proxy'] = "http://172.17.146.34:8891"
    os.environ['no_proxy'] = "localhost,127.0.0.1,172.17.146.34"

    from dataset import create_dataset, generate_attr_unique, create_loader

    a = ManinegII(t_backbone='bert-base-chinese', drop_text=0.5, drop_attr=0.5, vision_pretrained=True)
    config = {'batch_size': 48, 'image_res': 256, 'crop_min_scale': 0.5, "binary_label": True,
              "pre_downsample": True, 'modal': 'd', 't_backbone': 'bert-base-chinese', 'attr_noise': None,
              'partial_data': None}
    datasets = create_dataset('hmbm', config)
    train, val, test = datasets[0], datasets[1], datasets[2]

    train_loader, val_loader, test_loader = create_loader(
        [train, val, test],
        samplers=[None, None, None],
        batch_size=[config['batch_size'], config['batch_size'], config['batch_size']],
        num_workers=[config['batch_size'] // 4, config['batch_size'] // 4, config['batch_size'] // 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
        drop_last=[False, False, False],
        pin_memory=False
    )
    count = 0
    for i in train_loader:
        (image_embeds1, image_proj_feat1, image_embeds2, image_proj_feat2, text_output, text_proj_feat, attr_embeds,
         attr_proj_feat) = a(i)

        print(image_embeds1.size())
        print(attr_embeds)

        count += 1
        if count == 3:
            break
