# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional

from .resnet import ResNet
import pdb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BatchDrop(nn.Module):
    def __init__(self, drop_prob, drop_thr):
        super(BatchDrop, self).__init__()
        self.drop_prob = drop_prob
        self.drop_thr = drop_thr
    
    def forward(self, x):
        if self.training:
            # Generate self-attention map

            attenion = torch.mean(x, dim=1, keepdim=True)
            pdb.set_trace()

            # Generate importance map
            # importance_map = nn.Sigmoid(attenion)

            # Generate drop mask
            max_val = torch.max(attenion, [1,2,3])
            thr_val = max_val * self.drop_thr
            p = attenion
            p = torch.where(p >= thr_val, torch.full_like(p, 0), p)
            drop_mask = torch.where(p < thr_val, torch.full_like(p, 1), p)
            pdb.set_trace()

            # Random selection
            random_tensor = torch.rand([])
            add_tensor = torch.fill(random_tensor.size(), self.drop_prob)
            random_tensor = torch.add(random_tensor, add_tensor)
            binary_tensor = int(random_tensor)
            selected_map = (1. - binary_tensor) * importance_map + binary_tensor * drop_mask


            # Spatialwise multiplication to input feature map
            x = x * selected_map



        return x

class ADL(nn.Module):
    def __init__(self, drop_rate, drop_thr):
        super(ADL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr
        self.attention = None
        self.drop_mask = None
        self.attention2 = None

    def extra_repr(self):
        return 'drop_rate={}, drop_thr={}'.format(
            self.drop_rate, self.drop_thr
        )

    def forward(self, x, out):
        if self.training:
            b = x.size(0)

            # Generate self-attention map
            attention = torch.mean(x, dim=1, keepdim=True)
            self.attention = attention



            # Generate importance map
            importance_map = out

            # Generate drop mask
            max_val, _ = torch.max(attention.view(b, -1), dim=1, keepdim=True)
            thr_val = max_val * self.drop_thr
            thr_val = thr_val.view(b, 1, 1, 1).expand_as(attention)
            drop_mask = (attention < thr_val).float()
            self.drop_mask = drop_mask
            output = x.mul(drop_mask)
            # Random selection
            random_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
            binary_tensor = random_tensor.floor()
            selected_map = (1. - binary_tensor) * importance_map + binary_tensor * output

            # Spatial multiplication to input feature map


        else:
            selected_map = x
        return selected_map

    def get_maps(self):
        return self.attention, self.drop_mask

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x

class BatchCropElements(nn.Module):
    def __init__(self, prob):
        super(BatchCropElements, self).__init__()
        self.prob = prob
    def forward(self, x):
        if self.training:
            mask = x.new_zeros(x.size())
            h, w = x.size()[-2:]
            for i in range(h):
                for j in range(w):
                    if random.random() > self.prob:
                        mask[:, :, i, j] = 1
            x = x * mask
        return x


class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))

        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.in_planes, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [
                {'params': base_param_group},
                {'params': add_param_group}
            ]
        else:
            return [
                {'params': base_param_group}
            ]

class BFE(nn.Module):
    def __init__(self, num_classes, drop_prob=0.25, drop_thr=0.80):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
         # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_bn = nn.BatchNorm1d(512)
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        '''
        self.res_part2 = Bottleneck(2048, 512)
     
        self.part_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.adl = ADL(drop_prob, drop_thr)
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(1024)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)

        # attention
        '''
        #self.chanel_in = 2048
        #self.query_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // 8, kernel_size=1)
        #self.key_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in // 8, kernel_size=1)
        #self.value_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.chanel_in, kernel_size=1)
        #self.gamma = nn.Parameter(torch.zeros(1))

        #self.softmax = nn.Softmax(dim=-1)  #
        '''
        channels = 2048
        reduction = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid_spatial = nn.Sigmoid()
       '''
    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)

        predict = []
        triplet_features = []
        softmax_features = []

        #global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        global_bn = self.global_bn(global_triplet_feature)
        global_softmax_class = self.global_softmax(global_bn)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)
       
        #part branch
        #x = self.res_part2(x)


        # attention
        '''
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        '''
        '''
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x_out = avg + mx 
        x_out = self.sigmoid_channel(x_out)
        # Spatial attention module
        x_out = module_input * x_out
        module_input = x_out
        avg = torch.mean(x_out, 1, True)
        mx, _ = torch.max(x_out, 1, True)
        x_out = torch.cat((avg, mx), 1)
        x_out = self.conv_after_concat(x_out)
        x_out = self.sigmoid_spatial(x_out)
        x_out = module_input * x_out


        x = self.adl(x, x_out)
        triplet_feature = self.part_maxpool(x).squeeze()
        feature = self.reduction(triplet_feature)
        part_bn = self.part_bn(feature)
        softmax_feature = self.softmax(part_bn)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)
        '''
        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            #{'params': self.res_part2.parameters()},
            #{'params': self.reduction.parameters()},
            #{'params': self.softmax.parameters()},

            #{'params': self.query_conv.parameters()},
            #{'params': self.key_conv.parameters()},
            #{'params': self.value_conv.parameters()},

            #{'params': self.avg_pool.parameters()},
           # {'params': self.max_pool.parameters()},
            #{'params': self.fc1.parameters()},
            #{'params': self.relu.parameters()},
            #{'params': self.fc2.parameters()},
            #{'params': self.conv_after_concat.parameters()},
        ]
        return params

class Resnet(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return [], [feature]
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()

class IDE(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()