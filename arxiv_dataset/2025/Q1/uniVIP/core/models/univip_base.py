import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn

from ..utils import correlation
from ..models.softsplat import softsplat


#**************************************************************************************************#
# => Feature Pyramid
#**************************************************************************************************#
class FeatPyramid(nn.Module):
    """A 3-level feature pyramid, which by default is shared by the motion
    estimator and synthesis network.
    """
    def __init__(self):
        super(FeatPyramid, self).__init__()
        self.conv_stage0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))

    def forward(self, img):
        C0 = self.conv_stage0(img)
        C1 = self.conv_stage1(C0)
        C2 = self.conv_stage2(C1)
        return [C0, C1, C2]




#**************************************************************************************************#
# => Motion Estimation
#**************************************************************************************************#
class MotionEstimator(nn.Module):
    """Bi-directional optical flow estimator
    1) construct partial cost volume with the CNN features from the stage 2 of the feature pyramid;
    2) estimate bi-directional flows, from cost volume, CNN features of warped images (towards hidden middle frame), CNN
    feature for prediction in previous iteration, and bi-directional flow upsampled from previous iteration.
    """
    def __init__(self):
        super(MotionEstimator, self).__init__()
        # (4*2 + 1) ** 2 + 64 * 2 + 64 + 4 = 277
        self.conv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=277, out_channels=160,
                    kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=128,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=112,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels=112, out_channels=96,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=64,
                    kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_layer6 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=4,
                    kernel_size=3, stride=1, padding=1))


    def forward(self, feat0, feat1, last_feat, last_flow):
        corr_fn=correlation.FunctionCorrelation
        feat0 = softsplat.FunctionSoftsplat(
                tenInput=feat0, tenFlow=last_flow[:, :2]*0.25*0.5,
                tenMetric=None, strType='average')
        feat1 = softsplat.FunctionSoftsplat(
                tenInput=feat1, tenFlow=last_flow[:, 2:]*0.25*0.5,
                tenMetric=None, strType='average')

        volume = F.leaky_relu(
                input=corr_fn(tenFirst=feat0, tenSecond=feat1),
                negative_slope=0.1, inplace=False)
        input_feat = torch.cat([volume, feat0, feat1, last_feat, last_flow], 1)
        feat = self.conv_layer1(input_feat)
        feat = self.conv_layer2(feat)
        feat = self.conv_layer3(feat)
        feat = self.conv_layer4(feat)
        feat = self.conv_layer5(feat)
        flow = self.conv_layer6(feat)

        return flow, feat




#**************************************************************************************************#
# => Frame Synthesis
#**************************************************************************************************#
class SynthesisNetwork(nn.Module):
    """ Synthesis network that performs task-specific frame synthesis.
    That is, the synthesis network has two modes: 1) interpolation; 2) prediction.

    The synthesis mode is determined by the target time step.
    If the time step t falls between input frames (i.e., t in [0, 1]), it is frame interpolation.
    Otherwise, if t < 0 or t > 1, the target frame will be obtained by frame prediction.

    To make the synthesis network aware of current mode, we augment the input with a special task channel.
    """
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        # basic input: upsampled frame, original input frames, warped frames
        input_channels = 3 + 6 + 6
        # add the task-specific channel
        input_channels += 1
        self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32,
                    kernel_size=3, stride=1, padding=1),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.encoder_down1 = nn.Sequential(
                nn.Conv2d(in_channels=32 + 16 + 16, out_channels=64,
                    kernel_size=3, stride=2, padding=1),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64))
        self.encoder_down2 = nn.Sequential(
                nn.Conv2d(in_channels=64 + 32 + 32, out_channels=128,
                    kernel_size=3, stride=2, padding=1),
                nn.PReLU(num_parameters=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=128))
        self.decoder_up1 = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=128 + 64 + 64,
                    out_channels=64, kernel_size=4, stride=2,
                    padding=1, bias=True),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64))
        self.decoder_up2 = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=64 + 64,
                    out_channels=32, kernel_size=4, stride=2,
                    padding=1, bias=True),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.decoder_conv = nn.Sequential(
                nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.pred = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3,
                stride=1, padding=1)


    def get_warped_representations(self, bi_flow, c0, c1,
            i0=None, i1=None, time_period=0.5):
        # It is worth noting that interpolation and prediction mode share the same flow scaling formula
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)
        warped_c0 = softsplat.FunctionSoftsplat(
                tenInput=c0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_c1 = softsplat.FunctionSoftsplat(
                tenInput=c1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        if (i0 is None) and (i1 is None):
            return warped_c0, warped_c1
        else:
            warped_img0 = softsplat.FunctionSoftsplat(
                    tenInput=i0, tenFlow=flow_0t,
                    tenMetric=None, strType='average')
            warped_img1 = softsplat.FunctionSoftsplat(
                    tenInput=i1, tenFlow=flow_1t,
                    tenMetric=None, strType='average')
            flow_0_mid = bi_flow[:, :2] * 0.5
            flow_1_mid = bi_flow[:, 2:4] * 0.5
            flow_to_mid = torch.cat((flow_0_mid, flow_1_mid), 1)
            return warped_img0, warped_img1, warped_c0, warped_c1, flow_to_mid


    def forward(self, last_i, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr,
            time_period=0.5):
        warped_img0, warped_img1, warped_c0, warped_c1, flow_to_mid = \
                self.get_warped_representations(
                        bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1,
                        time_period=time_period)
        input_feat = torch.cat(
                (last_i, warped_img0, warped_img1, i0, i1), 1)

        # set task specific channel
        n, c, h, w = last_i.shape
        if (time_period > 0) and (time_period < 1): # interpolation
            flag_channel = torch.zeros((n, 1, h, w)).to(last_i.device)
        elif time_period >= 1: # predict next
            flag_channel = torch.ones((n, 1, h, w)).to(last_i.device)
        else: # time_period < 0, predict previous
            flag_channel = -torch.ones((n, 1, h, w)).to(last_i.device)
        input_feat = torch.cat((input_feat, flag_channel), 1)

        s0 = self.encoder_conv(input_feat)
        s1 = self.encoder_down1(torch.cat((s0, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
                        bi_flow_pyr[1], c0_pyr[1], c1_pyr[1],
                        time_period=time_period)
        s2 = self.encoder_down2(torch.cat((s1, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
                        bi_flow_pyr[2], c0_pyr[2], c1_pyr[2],
                        time_period=time_period)

        x = self.decoder_up1(torch.cat((s2, warped_c0, warped_c1), 1))
        x = self.decoder_up2(torch.cat((x, s1), 1))
        x = self.decoder_conv(torch.cat((x, s0), 1))

        # prediction
        refine = self.pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask0 = torch.sigmoid(refine[:, 3:4])
        refine_mask1 = torch.sigmoid(refine[:, 4:5])

        # set temporal weights for fusion
        if (time_period > 0) and (time_period < 1):
            weight0 = 1 - time_period
            weight1 = time_period
        else:
            weight0 = (1 - time_period) / (1 - 2 * time_period)
            weight1 = - time_period / (1 - 2 * time_period)

        merged_img = (warped_img0 * refine_mask0 * weight0 + \
                warped_img1 * refine_mask1 * weight1)
        merged_img = merged_img / (refine_mask0 * weight0 + \
                refine_mask1 * weight1)
        pred_img = merged_img + refine_res
        pred_img = torch.clamp(pred_img, 0, 1)

        extra_dict = {}
        extra_dict["refine_res"] = refine_res
        extra_dict["warped_img0"] = warped_img0
        extra_dict["warped_img1"] = warped_img1
        extra_dict["merged_img"] = merged_img

        return pred_img, extra_dict



#**************************************************************************************************#
# => Unified model
#**************************************************************************************************#
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feat_pyramid = FeatPyramid()
        self.motion_estimator = MotionEstimator()
        self.synthesis_network = SynthesisNetwork()

    def forward_one_lvl(self,
            img0, img1, last_feat, last_flow, last_interp=None,
            time_period=0.5, skip_me=False):
        """Forward process at each pyramid level
        """

        # context feature extraction
        feat0_pyr = self.feat_pyramid(img0)
        feat1_pyr = self.feat_pyramid(img1)

        # bi-directional flow estimation
        if not skip_me:
            flow, feat = self.motion_estimator(
                    feat0_pyr[-1], feat1_pyr[-1],
                    last_feat, last_flow)
        else:
            flow = last_flow
            feat = last_feat

        # frame synthesis
        ## optical flow is estimated at 1/4 resolution
        ori_resolution_flow = F.interpolate(
                input=flow, scale_factor=4.0,
                mode="bilinear", align_corners=False)

        ## construct 3-level flow pyramid for synthesis network
        bi_flow_pyr = []
        tmp_flow = ori_resolution_flow
        bi_flow_pyr.append(tmp_flow)
        for i in range(2):
            tmp_flow = F.interpolate(
                    input=tmp_flow, scale_factor=0.5,
                    mode="bilinear", align_corners=False) * 0.5
            bi_flow_pyr.append(tmp_flow)

        ## merge warped frames as initial interpolation for frame synthesis
        if last_interp is None:
            flow_0t = ori_resolution_flow[:, :2] * time_period
            flow_1t = ori_resolution_flow[:, 2:4] * (1 - time_period)
            warped_img0 = softsplat.FunctionSoftsplat(
                    tenInput=img0, tenFlow=flow_0t,
                    tenMetric=None, strType='average')
            warped_img1 = softsplat.FunctionSoftsplat(
                    tenInput=img1, tenFlow=flow_1t,
                    tenMetric=None, strType='average')

            if (time_period > 0) and (time_period < 1): # interpolation
                weight0 = 1 - time_period
                weight1 = time_period
            else: # prediction
                weight0 = (1 - time_period) / (1 - 2 * time_period)
                weight1 = - time_period / (1 - 2 * time_period)
            last_interp = warped_img0 * weight0 + warped_img1 * weight1

        ## do synthesis
        pred_img, extra_dict = self.synthesis_network(
                last_interp, img0, img1, feat0_pyr, feat1_pyr, bi_flow_pyr,
                time_period=time_period)
        return flow, feat, pred_img, extra_dict

    def forward(self, img0, img1, time_period, skip_pad=False, fixed_pyr_level=False):
        # pre-process input images and calculate image pyramidal levels based on resolution
        N, _, H, W = img0.shape
        pyr_level = fixed_pyr_level if fixed_pyr_level else math.ceil(math.log2(W/448) + 3)
        divisor = 2 ** (pyr_level-1+2) # based on image pyramid levels and the downsampling layers in feature extractor
        if (not skip_pad) and ((H % divisor != 0) or (W % divisor != 0)):
            ph = ((H - 1) // divisor + 1) * divisor
            pw = ((W - 1) // divisor + 1) * divisor
            padding = (0, pw - W, 0, ph - H)
            img0 = F.pad(img0, padding, "constant", 0.5)
            img1 = F.pad(img1, padding, "constant", 0.5)

        # When nr_lvl_skipped > 0, it means skipping corresponding (high-resolution) levels for motion estimation, and
        # skipping (nr_lvl_skipped-1) levels for frame synthesis. For the last pyramid level of original input
        # resolution, the frame synthesis step cannot be skipped.
        nr_lvl_skipped = 0 if pyr_level<= 5 else pyr_level - 5
        N, _, H, W = img0.shape # update the shape

        bi_flow_pyr = []
        pred_img_pyr = []
        skipped_levels = [] if nr_lvl_skipped == 0 else list(range(pyr_level))[::-1][-nr_lvl_skipped:]

        # The original input resolution corresponds to level 0.
        for level in list(range(pyr_level))[::-1]:
            if level != 0:
                scale_factor = 1 / 2 ** level
                img0_this_lvl = F.interpolate(
                        input=img0, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False)
                img1_this_lvl = F.interpolate(
                        input=img1, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False)
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1

            # flag to skip motion estimation, directly using up-sampled optical flow
            skip_me = False

            # the lowest-resolution pyramid level
            if level == pyr_level - 1:
                last_flow = torch.zeros(
                        (N, 4, H // (2 ** (level+2)), W //(2 ** (level+2)))
                        ).to(img0.device)
                last_feat = torch.zeros(
                        (N, 64, H // (2 ** (level+2)), W // (2 ** (level+2)))
                        ).to(img0.device)
                last_interp = None
            # skip some levels for both motion estimation and frame synthesis
            elif level in skipped_levels[:-1]:
                    continue
            # last level (original input resolution), only skip motion estimation
            elif (level == 0) and len(skipped_levels) > 0:
                if len(skipped_levels) == pyr_level:
                    last_flow = torch.zeros(
                            (N, 4, H // 4, W // 4)).to(img0.device)
                    last_interp = None
                else:
                    resize_factor = 2 ** len(skipped_levels)
                    last_flow = F.interpolate(
                            input=flow, scale_factor=resize_factor,
                            mode="bilinear", align_corners=False) * resize_factor
                    last_interp = F.interpolate(
                            input=pred_img, scale_factor=resize_factor,
                            mode="bilinear", align_corners=False)
                skip_me = True
            # last level (original input resolution), motion estimation + frame synthesis
            else:
                last_flow = F.interpolate(input=flow, scale_factor=2.0,
                        mode="bilinear", align_corners=False) * 2
                last_feat = F.interpolate(input=feat, scale_factor=2.0,
                        mode="bilinear", align_corners=False) * 2
                last_interp = F.interpolate(
                        input=pred_img, scale_factor=2.0,
                        mode="bilinear", align_corners=False)

            flow, feat, pred_img, _ = self.forward_one_lvl(
                    img0_this_lvl, img1_this_lvl,
                    last_feat, last_flow, last_interp,
                    time_period, skip_me=skip_me)
            bi_flow_pyr.append(
                    F.interpolate(input=flow, scale_factor=4.0,
                        mode="bilinear", align_corners=False))
            pred_img_pyr.append(pred_img)

        # directly up-sample estimated flow to full resolution with bi-linear interpolation
        bi_flow = F.interpolate(
                input=flow, scale_factor=4.0,
                mode="bilinear", align_corners=False)

        return pred_img, {"bi_flow": bi_flow, "pred_img_pyr": pred_img_pyr, "bi_flow_pyr": bi_flow_pyr}


if __name__ == "__main__":
    pass
