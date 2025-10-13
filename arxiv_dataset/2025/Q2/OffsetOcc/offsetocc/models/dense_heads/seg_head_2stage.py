from typing import List, Optional, Tuple

import torch
from mmcv.cnn import Linear
from mmengine.model import BaseModule
from torch import nn
from torch.nn import functional as F

from offsetocc.registry import MODELS
from offsetocc.structures import OccDataSample, OccupancyData
from offsetocc.utils import ConfigDict


@MODELS.register_module()
class SegmentationHead2Stage(BaseModule):

    """
    Segmentation head for occupancy grid prediction.

    Args:
        occ_l (int): The length of the occupancy grid.
        occ_w (int): The width of the occupancy grid.
        occ_h (int): The height of the occupancy grid.
        embed_dims (int): The embedding dimension.
        num_classes (int): The number of classes.
        scale_factors (tuple[float]): The scale factors for upsampling.
        loss_cls (dict): Config of the loss for segmentation head.
    """

    def __init__(self,
                 occ_l: int,
                 occ_w: int,
                 occ_h: int,
                 embed_dims: int,
                 num_classes: int,
                 scale_factors: Tuple[float, float, float],
                 mask_camera: bool) -> None:
                 # loss_cls: ConfigDict

        super().__init__()

        self.occ_l = int(occ_l * 1/scale_factors[0])
        self.occ_w = int(occ_w * 1/scale_factors[1])
        self.occ_h = int(occ_h * 1/scale_factors[2])
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.upscale = nn.Upsample(scale_factor=scale_factors, mode='trilinear', align_corners=True)
        self.mask_camera = mask_camera
        # self.loss_cls = MODELS.build(loss_cls)
        self._init_layers()
        # TODO: check what align_corners is

    def _init_layers(self) -> None:
        self.fc_cls = Linear(self.embed_dims, self.num_classes + 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # (bs, num_voxels c) -> (bs, c, l, w, h)
        feats = feats.permute(0, 2, 1)
        feats = feats.reshape(feats.shape[0], feats.shape[1], self.occ_l, self.occ_w, self.occ_h)
        feats = self.upscale(feats)
        feats = feats.transpose(4, 1)
        pred_logits = self.fc_cls(feats).transpose(1, 4)
        return pred_logits

    # def loss(self, feats: torch.Tensor, data_samples: List[OccDataSample],
    #          **kwargs) -> dict[str, torch.Tensor]:
    #     pred_logits = self(feats)
    #     losses = self._get_loss(pred_logits, data_samples)
    #     return losses

    # def _get_loss(self, cls_score: torch.Tensor, data_samples: List[OccDataSample]) -> dict[str, torch.Tensor]:
    #     """Unpack data samples and compute loss."""
    #
    #     target = torch.stack([i.gt_occ_map.occ_map for i in data_samples])
    #     mask_camera = torch.stack([i.gt_occ_map.occ_map_mask_camera for i in data_samples])
    #
    #     # set ignore value in the target based on the visibility mask
    #     if self.mask_camera:
    #         target[~mask_camera] = 255
    #
    #     # compute loss
    #     losses = dict()
    #     loss = self.loss_cls(cls_score, target)
    #     losses['loss'] = loss
    #
    #     return losses

    def predict(self, feats: Tuple[torch.Tensor],
                data_samples: Optional[List[Optional[OccDataSample]]] = None) -> List[OccDataSample]:
        pred_logits = self(feats)
        predictions = self._get_predictions(pred_logits, data_samples)
        return predictions

    def _get_predictions(self, pred_logits: torch.Tensor, data_samples: list[OccDataSample]) -> List[OccDataSample]:
        pred_scores = F.softmax(pred_logits, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
        out_data_samples = []

        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, logits, labels in zip(data_samples, pred_logits, pred_labels):

            if data_sample is None:
                data_sample = OccDataSample()

            occ_data_pred_logits = OccupancyData()
            occ_data_pred_occ_map = OccupancyData()

            occ_data_pred_logits['logits'] = logits
            occ_data_pred_occ_map['occ_map'] = labels

            data_sample.pred_occ_map_logits = occ_data_pred_logits
            data_sample.pred_occ_map = occ_data_pred_occ_map
            out_data_samples.append(data_sample)

        return out_data_samples
