
from collections import OrderedDict
import numpy as np
import math
import cv2

from typing import List
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net_utils import DenseBlock, UpSample2x
from backbones import create_model_unet
from .encoder_model import ResNetExt, SeResNextExt
from utils.util_funcs import get_bounding_box


####
class HoVerNetExt(nn.Module):
    """Initialise HoVer-Net."""

    def __init__(
            self,
            num_types=None,
            freeze=False,
            freeze_tp=False,
            freeze_np=False,
            freeze_hv=False,
            pretrained_backbone=None,
            encoder_name="resnet50",
            **kwargs,
            ):
        super().__init__()
        self.freeze = freeze
        self.freeze_tp = freeze_tp
        self.freeze_np = freeze_np
        self.freeze_hv = freeze_hv
        self.num_types = num_types
        self.output_ch = 3 if num_types is None else 4

        if encoder_name.lower() == "resnet50":
            self.backbone = ResNetExt.resnet50(
                3, pretrained=pretrained_backbone)
        elif encoder_name.lower() == "resnext50":
            self.backbone = ResNetExt.resnext50(
                3, pretrained=pretrained_backbone)
        elif encoder_name.lower() == "resnext101":
            self.backbone = ResNetExt.resnext101(
                3, pretrained=pretrained_backbone)
        elif encoder_name.lower() == "seresnet50":
            self.backbone = SeResNextExt.seresnet50(
                3, pretrained=pretrained_backbone)
        elif encoder_name.lower() == "seresnext50_32x4d":
            self.backbone = SeResNextExt.seresnext50_32x4d(
                3, pretrained=pretrained_backbone)

        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        d0, d1, d2, d3 = self.backbone(imgs, self.freeze)
        d3 = self.conv_bot(d3)

        d = [d0, d1, d2, d3]

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            if (self.freeze_tp and branch_name == "tp") or (self.freeze_np and branch_name == "np") \
                    or (self.freeze_hv and branch_name == "hv"):
                branch_desc[0].eval()
                branch_desc[1].eval()
                branch_desc[2].eval()
                branch_desc[3].eval()
                
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        # np_predicted, hv_predicted, tp_predicted
        return out_dict["np"], out_dict["hv"], out_dict["tp"]

    @staticmethod
    def _proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray, fx: float = 1):
        blb_raw = np_map[..., 0]
        h_dir_raw = hv_map[..., 0]
        v_dir_raw = hv_map[..., 1]

        # processing
        blb = np.array(blb_raw >= 0.4, dtype=np.int32)

        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * fx) + 1)
        obj_size = math.ceil(10 * (fx ** 2))
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=obj_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred

    @staticmethod
    def _get_instance_info(pred_inst, pred_type=None):
        inst_id_list = np.unique(pred_inst)[1:]  # exclude background

        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id

            [rmin, rmax, cmin, cmax] = get_bounding_box(inst_map)
            inst_box = np.array([cmin, rmin, cmax, rmax])
            inst_box_tl = inst_box[:2]
            inst_map = inst_map[inst_box[1] : inst_box[3], inst_box[0] : inst_box[2]]
            inst_map = inst_map.astype(np.uint8)
            
            # cv2.imwrite("visualize.png", inst_map)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break

            inst_contour = inst_contour[0][0].astype(np.int32)
            inst_contour = np.squeeze(inst_contour)

            # < 3 points does not make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small
            if inst_contour.shape[0] < 3:  # pragma: no cover
                continue

            # ! check for trickery shape
            if len(inst_contour.shape) != 2:  # pragma: no cover
                continue

            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour += inst_box_tl[None]
            inst_centroid += inst_box_tl  # X
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "box": inst_box,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "prob": None,
                "type": None,
            }

        if pred_type is not None:
            # * Get class of each instance id, stored at index id-1
            for inst_id in list(inst_info_dict.keys()):
                cmin, rmin, cmax, rmax = inst_info_dict[inst_id]["box"]
                inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]

                (type_list, type_pixels) = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)

                inst_type = type_list[0][0]

                # ! pick the 2nd most dominant if it exists
                if inst_type == 0 and len(type_list) > 1:  # pragma: no cover
                    inst_type = type_list[1][0]

                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["prob"] = float(type_prob)

        return inst_info_dict

    @staticmethod
    # skipcq: PYL-W0221
    def postproc(raw_maps: List[np.ndarray]):
        if len(raw_maps) == 3:
            np_map, hv_map, tp_map = raw_maps
        else:
            tp_map = None
            np_map, hv_map = raw_maps

        pred_type = tp_map
        pred_inst = HoVerNetExt._proc_np_hv(np_map, hv_map)
        nuc_inst_info_dict = HoVerNetExt._get_instance_info(pred_inst, pred_type)

        return pred_inst, nuc_inst_info_dict

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        patch_imgs = batch_data

        device = "cuda" if on_gpu else "cpu"
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.inference_mode():
            pred_dict = model(patch_imgs_gpu)
            pred_dict = OrderedDict(
                [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.float32)
                pred_dict["tp"] = type_map
            pred_dict = {k: v.cpu().numpy() for k, v in pred_dict.items()}

        if "tp" in pred_dict:
            return pred_dict["np"], pred_dict["hv"], pred_dict["tp"]
        return pred_dict["np"], pred_dict["hv"]

    @staticmethod
    def infer_batch_second_stage(model, batch_data, on_gpu):
        patch_imgs = batch_data

        device = "cuda" if on_gpu else "cpu"
        patch_imgs_gpu = patch_imgs.to(device).type(torch.float32)  # to NCHW
        patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

        model.eval()  # infer mode

        # --------------------------------------------------------------

        with torch.inference_mode():
            np_map, hv_map, tp_raw = model(patch_imgs_gpu)
            np_map = np_map.permute(0, 2, 3, 1).contiguous()
            np_map = F.softmax(np_map, dim=-1)[..., 1:]
            np_map = np_map.cpu().numpy()

            hv_map = hv_map.permute(0, 2, 3, 1).contiguous()
            # hv_map = torch.tanh(hv_map)
            hv_map = hv_map.cpu().numpy()

            tp_raw = F.softmax(tp_raw.permute(0, 2, 3, 1).contiguous(), dim=-1)
            type_map = torch.argmax(tp_raw, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            type_map = type_map.cpu().numpy()
            tp_raw = tp_raw.cpu().numpy()

        return np_map, hv_map, type_map, tp_raw


####
def create_model(mode=None, **kwargs):
    if "hovernet" in mode:
        return HoVerNetExt(**kwargs)
    elif mode == "unet":
        return create_model_unet(**kwargs)

