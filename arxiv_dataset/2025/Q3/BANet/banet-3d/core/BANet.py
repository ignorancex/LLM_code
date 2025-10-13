import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
from .fnet import FeatureNet, DeconvLayer
import math
import gc
import time


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv0 = BasicConv(96, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = BasicConv(192, 64, kernel_size=3, stride=1, padding=1)
        
        self.lastconv = nn.Sequential(
            BasicConv(64*3, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 1))

        self.attn = nn.Conv2d(64, 1, 5, padding=2, bias=False)

    def forward(self, features_left):
        x4 = self.conv0(features_left[0])
        x8 = self.conv1(features_left[1])
        x16 = self.conv2(features_left[2])
        x8_4 = F.interpolate(x8, x4.shape[2:], mode='bilinear', align_corners=False)
        x16_4 = F.interpolate(x16, x4.shape[2:], mode='bilinear', align_corners=False)
        x = self.lastconv(torch.cat([x4, x8_4, x16_4], dim=1))
        att = self.attn(x)
        att = torch.sigmoid(att)
        return att.unsqueeze(2)

class Guided_Cost_Volume_Excitation(nn.Module):
    def __init__(self, cv_chan, im_chan):
        super(Guided_Cost_Volume_Excitation, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan, cv_chan, 1))

    def forward(self, cv, im):
        im_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(im_att) * cv
        return cv
    

class Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Aggregation, self).__init__()
        
        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, padding=1, stride=1)
        self.downconv1 = BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1)
        
        self.conv1 = BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1)

        self.downconv2 = BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1)
        
        self.conv2 = BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1)

        self.downconv3 = BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1)
        
        self.conv3 = BasicConv(in_channels*6, in_channels*6, is_3d=True, kernel_size=3, padding=1, stride=1)



        self.upconv1 = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.concat1 = BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1)

        self.agg1 = nn.Sequential(BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.upconv2 = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.concat2 = BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1)

        self.agg2 = nn.Sequential(BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.upconv3 = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.gce_4 = Guided_Cost_Volume_Excitation(in_channels, 96)
        self.gce_8 = Guided_Cost_Volume_Excitation(in_channels*2, 64)
        self.gce_16 = Guided_Cost_Volume_Excitation(in_channels*4, 192)
        self.gce_32 = Guided_Cost_Volume_Excitation(in_channels*6, 160)
        self.gce_16_up = Guided_Cost_Volume_Excitation(in_channels*4, 192)
        self.gce_8_up = Guided_Cost_Volume_Excitation(in_channels*2, 64)

    def forward(self, x, imgs):
        cv_4 = self.gce_4(x, imgs[0])
        cv_4 = self.conv0(cv_4)

        cv_8 = self.downconv1(cv_4)
        cv_8 = self.gce_8(cv_8, imgs[1])
        cv_8 = self.conv1(cv_8)

        cv_16 = self.downconv2(cv_8)
        cv_16 = self.gce_16(cv_16, imgs[2])
        cv_16 = self.conv2(cv_16)

        cv_32 = self.downconv3(cv_16)
        cv_32 = self.gce_32(cv_32, imgs[3])
        cv_32 = self.conv3(cv_32)

        cv_16_up = self.upconv1(cv_32)
        cv_16 = self.concat1(torch.cat((cv_16_up, cv_16), dim=1))
        cv_16 = self.gce_16_up(cv_16, imgs[2])
        cv_16 = self.agg1(cv_16)

        cv_8_up = self.upconv2(cv_16)
        cv_8 = self.concat2(torch.cat((cv_8_up, cv_8), dim=1))
        cv_8 = self.gce_8_up(cv_8, imgs[1])
        cv_8 = self.agg2(cv_8)

        cv_4 = self.upconv3(cv_8)
        return cv_4
    

class BANet(nn.Module):
    def __init__(self, args):
        super(BANet, self).__init__()
        self.fnet = FeatureNet()
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.cost_agg0 = Aggregation(8)
        self.cost_agg1 = Aggregation(8)
        self.spa_att = SpatialAttention()
        
        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 48, kernel_size=3, stride=1, padding=1)
            )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = DeconvLayer(32, 32)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=1, padding=1)
            )
        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        
    def upsample_disp(self, disp, mask, scale=4):
        """ Upsample disp field [H//4, W//4] -> [H, W] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(scale * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, scale*H, scale*W)

    def forward(self, left, right, max_disp=192):
        features_left = self.fnet(left)
        features_right = self.fnet(right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, max_disp//4)
        spa_att = self.spa_att(features_left)
        volume = self.corr_stem(corr_volume)
        cv_0 = spa_att * volume
        cv_1 = (1. - spa_att) * volume
        cv_0 = self.cost_agg0(cv_0, features_left)
        cv_1 = self.cost_agg1(cv_1, features_left)
        cv = spa_att * cv_0 + (1. - spa_att) * cv_1
        prob = F.softmax(cv.squeeze(1), dim=1)
        disp = disparity_regression(prob, max_disp // 4)  # [B, 1, H//4, W//4]

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        disp_up = context_upsample(disp, spx_pred) # [B, 1, H, W]
        
        if self.training:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up*4., disp_linear*4.]
        else:
            return disp_up*4.