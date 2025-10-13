#!/usr/bin/env python
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from models.unet_3d import mink_unet as model3D
from models.resnet_2d import ResNet as ResNet_small
from models.resnet_2d import BasicBlock as BasicBlock_small
import MinkowskiEngine as ME
import os
import requests
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def get_coords_map(x, y):
    assert (
        x.coordinate_manager == y.coordinate_manager
    ), "X and Y are using different CoordinateManagers. Y must be derived from X through strided conv/pool/etc."
    return x.coordinate_manager.stride_map(x.coordinate_map_key, y.coordinate_map_key)


def adapt_weights(architecture):
    if architecture == "imagenet" or architecture == "imagenet_no" or architecture is None:
        return

    weights_url = {
        "moco_v2": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
        "moco_v1": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
        "swav": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
        "deepcluster_v2": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar",
        "dino": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    }

    if not os.path.exists(f"weights/{architecture}.pt"):
        r = requests.get(weights_url[architecture], allow_redirects=True)
        os.makedirs("weights", exist_ok=True)
        with open(f"weights/{architecture}.pt", 'wb') as f:
            f.write(r.content)

    weights = torch.load(f"weights/{architecture}.pt")

    if architecture == "obow":
        return weights["network"]

    if architecture == "pixpro":
        weights = {
            k.replace("module.encoder.", ""): v
            for k, v in weights["model"].items()
            if k.startswith("module.encoder.")
        }
        return weights

    if architecture in ("moco_v1", "moco_v2", "moco_coco"):
        weights = {
            k.replace("module.encoder_q.", ""): v
            for k, v in weights["state_dict"].items()
            if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc")
        }
        return weights

    if architecture in ("swav", "deepcluster_v2"):
        weights = {
            k.replace("module.", ""): v
            for k, v in weights.items()
            if k.startswith("module.") and not k.startswith("module.pro")
        }
        return weights

    if architecture == "dino":
        return weights


class Cam_3D(nn.Module):

    def __init__(self, cfg=None):
        super(Cam_3D, self).__init__()
        
        # 3D net initialization
        net3d = model3D(in_channels=cfg.in_channel, out_channels=cfg.classes, D=3, arch=cfg.arch_3d)
        net3d.weight_initialization()

        # cam encoder
        self.layer0_3d = nn.Sequential(net3d.conv0p1s1, net3d.bn0, net3d.relu)
        self.layer1_3d = nn.Sequential(net3d.conv1p1s2, net3d.bn1, net3d.relu, net3d.block1)
        self.layer2_3d = nn.Sequential(net3d.conv2p2s2, net3d.bn2, net3d.relu, net3d.block2)
        self.layer3_3d = nn.Sequential(net3d.conv3p4s2, net3d.bn3, net3d.relu, net3d.block3)
        self.layer4_3d = nn.Sequential(net3d.conv4p8s2, net3d.bn4, net3d.relu, net3d.block4)

        # cam decoder
        self.layer5_3d = nn.Sequential(net3d.convtr4p16s2, net3d.bntr4, net3d.relu)
        self.layer6_3d = nn.Sequential(net3d.block5, net3d.convtr5p8s2, net3d.bntr5, net3d.relu)
        self.layer7_3d = nn.Sequential(net3d.block6, net3d.convtr6p4s2, net3d.bntr6, net3d.relu)
        self.layer8_3d = nn.Sequential(net3d.block7, net3d.convtr7p2s2, net3d.bntr7, net3d.relu)
        self.layer9_3d = net3d.block8
        self.cls_3d = net3d.final
        self.out_planes = net3d.PLANES[-1]

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                # print(m)
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sparse_3d):
        # 3D feature extract
        out_p1 = self.layer0_3d(sparse_3d)
        out_b1p2 = self.layer1_3d(out_p1)
        out_b2p4 = self.layer2_3d(out_b1p2)
        out_b3p8 = self.layer3_3d(out_b2p4)
        out_b4p16 = self.layer4_3d(out_b3p8)  # corresponding to FPN p5

        out = self.layer5_3d(out_b4p16)
        out = self.layer6_3d(ME.cat(out, out_b3p8))
        out = self.layer7_3d(ME.cat(out, out_b2p4))
        out = self.layer8_3d(ME.cat(out, out_b1p2))
        out = self.layer9_3d(ME.cat(out, out_p1))

        feat_3d = out
        pred_3d = self.cls_3d(out)
        return [pred_3d, feat_3d]


class Cam_2D(nn.Module):

    def __init__(self, cfg=None):
        super(Cam_2D, self).__init__()

        self.layers_2d = cfg.layers_2d
        self.image_weights = cfg.get("image_weights", "imagenet")

        if cfg.layers_2d == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            dict_url = model_urls["resnet50"]
            net2d = ResNet(block=block, layers=layers, replace_stride_with_dilation=[False, True, True])
            if self.image_weights == "imagenet":
                net2d.load_state_dict(model_zoo.load_url(dict_url))
                print("load image net weight")
            else:
                weights = adapt_weights(architecture=self.image_weights)
                if weights is not None:
                    load_state_info = net2d.load_state_dict(weights, strict=False)
                    print(f"Missing keys of 2D backbone: {load_state_info[0]}")
            
        elif cfg.layers_2d == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
            dict_url = model_urls["resnet101"]
            net2d = ResNet(block=block, layers=layers, replace_stride_with_dilation=[False, True, True])
            net2d.load_state_dict(model_zoo.load_url(dict_url))

        elif cfg.layers_2d == 34:
            block = BasicBlock_small
            layers = [3, 4, 6, 3]
            dict_url = model_urls["resnet34"]
            net2d = ResNet_small(block=block, layers=layers, replace_stride_with_dilation=[False, True, True])
            if self.image_weights == "imagenet":
                net2d.load_state_dict(model_zoo.load_url(dict_url))
                print("load image net weight")

        elif cfg.layers_2d == 18:
            block = BasicBlock_small
            layers = [2, 2, 2, 2]
            dict_url = model_urls["resnet18"]
            net2d = ResNet_small(block=block, layers=layers, replace_stride_with_dilation=[False, True, True])
            if self.image_weights == "imagenet":
                net2d.load_state_dict(model_zoo.load_url(dict_url))
                print("load image net weight")

        self.layer0_2d = nn.Sequential(net2d.conv1, net2d.bn1, net2d.maxpool)
        self.layer1_2d = net2d.layer1
        self.layer2_2d = net2d.layer2
        self.layer3_2d = net2d.layer3
        self.layer4_2d = net2d.layer4
        self.layer4_2d[-1].relu = nn.Identity()

        out_channels = [64,128,256,512]
        if cfg.layers_2d == 50 or cfg.layers_2d == 101:
            out_channels = [256,512,1024,2048]
        self.out_channels = out_channels[-1]
        self.cls_2d = nn.Conv2d(out_channels[-1], cfg.classes, 1, bias=False)
        self.out_planes = out_channels[-1]

    def forward(self, images):
        
        # 2D feature extract
        x_size = images.size()
        h, w = x_size[2], x_size[3]
    
        data_2d = images.permute(0, 4, 1, 2, 3).contiguous()  # VBCHW
        data_2d = data_2d.view(x_size[0] * x_size[4], x_size[1], x_size[2], x_size[3])
        
        if self.layers_2d == 38:
            d = self.net2d.forward_as_dict(data_2d)
            x2 = d['conv3']
            x3 = d['conv4']
            x4 = d['conv5']
            x5 = d['conv6']
            x5 = F.relu(x5)
            # print(x2.shape,x3.shape,x4.shape,x5.shape)
        else:
            x = self.layer0_2d(data_2d)  # 1/4
            x2 = self.layer1_2d(x)  # 1/4
            x3 = self.layer2_2d(x2)  # 1/8
            x4 = self.layer3_2d(x3)  # 1/8
            x5 = self.layer4_2d(x4)  # 1/8

        cam_2d = self.cls_2d(x5)
        pred_2d = F.interpolate(cam_2d, size=(h, w), mode='bilinear', align_corners=True)
        feat_2d = x5

        return [pred_2d, feat_2d]


class WSegPC(nn.Module):

    def __init__(self, cfg=None):
        super(WSegPC, self).__init__()

        self.net_2d = Cam_2D(cfg)
        self.net_3d = Cam_3D(cfg)

        self.proj_3d = nn.Linear(self.net_3d.out_planes, cfg.proj_3d_channel)
        self.proj_2d = nn.Linear(self.net_2d.out_planes, cfg.proj_2d_channel)

    def forward(self, sparse_3d, images, links=None):
        pred_2d, feat_2d = self.net_2d(images)
        pred_3d, feat_3d = self.net_3d(sparse_3d)
        if links is not None:
            pairing_points, pairing_pixels = get_pairing_points_pixels(links, feat_3d, feat_2d, pred_2d)
            
            feat_proj_3d = [self.proj_3d(fea) for fea in pairing_points]
            feat_detach_3d = [fea.detach() for fea in pairing_points]

            feat_proj_3d = [F.normalize(fea, p=2, dim=1) for fea in feat_proj_3d]
            feat_detach_3d = [F.normalize(fea, p=2, dim=1) for fea in feat_detach_3d]

            feat_proj_2d = [self.proj_2d(fea) for fea in pairing_pixels]
            feat_detach_2d = [fea.detach() for fea in pairing_pixels]

            feat_proj_2d = [F.normalize(fea, p=2, dim=1) for fea in feat_proj_2d]
            feat_detach_2d = [F.normalize(fea, p=2, dim=1) for fea in feat_detach_2d]

            return [pred_3d, feat_proj_3d, feat_detach_3d], [pred_2d, feat_proj_2d, feat_detach_2d]
        else:
            return [pred_3d], [pred_2d]


def get_pairing_points_pixels(links, feat_3d, feat_2d, pred_2d):
    _,_,h,w = pred_2d.shape
    V_B, C, H, W = feat_2d.shape
    V_NUM = links.shape[-1]
    feat_2d = feat_2d.view(V_B//V_NUM, V_NUM, C, H, W)
    feat_3d = feat_3d.F

    _, _, v = links.shape
    pairing_points = []
    pairing_pixels = []
    for b_i in range(V_B//V_NUM):
        pairing_point = []
        pairing_pixel = []
        for v_i in range(v):
            links_c = links[:,:,v_i]
            links_c_mask = (links_c[:,0]==b_i) * (links_c[:,-1]==1)
            
            p_point = torch.nonzero(links_c_mask, as_tuple=True)
            p_point = p_point[0]
            if len(p_point) > 64:
                
                idx = np.random.choice(len(p_point), min(len(p_point), 512), replace=False)
                p_point = p_point[idx]

                feat_3d_c = feat_3d[p_point]
                pairing_point.append(feat_3d_c)

                feat_2d_c = feat_2d[b_i, v_i].unsqueeze(0)
                p_pixel = links_c[:,1:3][p_point]
                p_pixel_x = 2*(p_pixel[:,1]/w)-1
                p_pixel_y = 2*(p_pixel[:,0]/h)-1
                p_pixel_grid = torch.stack([p_pixel_x, p_pixel_y], dim=-1)
                p_pixel_grid = p_pixel_grid.unsqueeze(0).unsqueeze(1)
                feat_2d_c = torch.nn.functional.grid_sample(feat_2d_c, p_pixel_grid, mode='bilinear', align_corners=True)
                feat_2d_c = feat_2d_c.squeeze().permute(1, 0)
                pairing_pixel.append(feat_2d_c)

        if len(pairing_point) > 0 and len(pairing_pixel) > 0:
            pairing_points.append(torch.cat(pairing_point).contiguous())
            pairing_pixels.append(torch.cat(pairing_pixel).contiguous())

    return pairing_points, pairing_pixels