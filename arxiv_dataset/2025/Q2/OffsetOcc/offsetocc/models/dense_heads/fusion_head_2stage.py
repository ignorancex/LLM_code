from typing import Dict
from typing import List

import torch
from mmdet3d.structures import PointData
from mmengine.model import BaseModule
from torch.nn import functional as F

from offsetocc.registry import MODELS
from offsetocc.structures import OccDataSample, OccupancyData


@MODELS.register_module()
class FusionHead2Stage(BaseModule):

    """
    Segmentation head for occupancy grid prediction.

    Args:
        occ_l (int): The length of the occupancy grid.
        occ_w (int): The width of the occupancy grid.
        occ_h (int): The height of the occupancy grid.
        embed_dims (int): The embedding dimension.
        num_classes (int): The number of classes.
        scale_factors (tuple[float]): The scale factors for upsampling.
        loss_cls_cfg (dict): Config of the loss for segmentation head.
    """

    def __init__(self,
                 embed_dims: int,
                 num_classes: int,
                 lambda_merge: float,
                 no_object_logit_value: float,
                 obj_classes_indices: List[int],
                 occ_l: int,
                 occ_w: int,
                 occ_h: int,
                 pc_range: List[float],
                 max_voxel_distance: int,
                 majority_voting: bool,
                 pred_point_labels: bool) -> None:

        super().__init__()

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.lambda_merge = lambda_merge
        self.no_object_logit_value = no_object_logit_value
        self.obj_classes_indices = obj_classes_indices
        self.pc_range = pc_range
        self.majority_voting = majority_voting
        self.max_voxel_distance = max_voxel_distance
        self.occ_l, self.occ_w, self.occ_h = occ_l, occ_w, occ_h
        if majority_voting:
            self.neighbor_indices: torch.Tensor
            self.valid_mask: torch.Tensor
            self.init_neighbor_map()
        self.pred_point_labels = pred_point_labels
        # self._init_layers()

    # def _init_layers(self) -> None:
    #     self.fc_cls = Linear(self.embed_dims, self.num_classes + 1)

    def _apply(self, fn, **kwargs) -> 'FusionHead2Stage':
        super()._apply(fn, **kwargs)
        if self.majority_voting:
            self.neighbor_indices = fn(self.neighbor_indices)
            self.valid_mask = fn(self.valid_mask)
        return self

    def init_neighbor_map(self) -> None:

        D = self.max_voxel_distance

        offsets = [(dx, dy, dz) for dx in range(-D, D + 1)
                   for dy in range(-D, D + 1)
                   for dz in range(-D, D + 1)
                   if abs(dx) + abs(dy) + abs(dz) <= D]

        offsets = torch.tensor(offsets, dtype=torch.int)

        # Create a 3D grid of voxel coordinates
        i_coords, j_coords, k_coords = torch.meshgrid(
            torch.arange(self.occ_l), torch.arange(self.occ_w), torch.arange(self.occ_h), indexing="ij"
        )

        voxel_coords = torch.stack([i_coords, j_coords, k_coords], dim=-1).reshape(-1, 3)
        voxel_coords = voxel_coords.int()

        # Compute absolute neighbor positions for each voxel
        neighbor_indices = voxel_coords[:, None, :] + offsets[None, :, :]

        # mask invalid neighbors (outside the grid)

        valid_mask = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 0] <= self.occ_l) & \
                     (neighbor_indices[..., 1] >= 0) & (neighbor_indices[..., 1] <= self.occ_w) & \
                     (neighbor_indices[..., 2] >= 0) & (neighbor_indices[..., 2] <= self.occ_h)


        # valid_mask = (neighbor_indices >= 0) & (neighbor_indices < torch.tensor([self.occ_l, self.occ_w, self.occ_h])[None, None, None, :])

        # Clamp to ensure we remain inside valid tensor bounds
        neighbor_indices[..., 0] = neighbor_indices[..., 0].clamp(0, self.occ_l - 1)
        neighbor_indices[..., 1] = neighbor_indices[..., 1].clamp(0, self.occ_w - 1)
        neighbor_indices[..., 2] = neighbor_indices[..., 2].clamp(0, self.occ_h - 1)

        self.neighbor_indices = neighbor_indices
        self.valid_mask = valid_mask


    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, occupancy_logits: torch.Tensor, pred_obj_occ_logits: List[torch.Tensor]) -> [torch.Tensor, torch.Tensor]:

        """
        Args:
            occupancy_logits (torch.Tensor): The scores from the main head. Size: (bs, l, w, h, self.num_classes)
            pred_obj_occ_logits (List[torch.Tensor]): The predicted object occupancy logits. Size:
            bs * (num_objs, l, w, h)
        """

        bs, num_clas_plus1, _, _, _ = occupancy_logits.shape
        l, w, h = occupancy_logits[0].shape[-3:]

        assert num_clas_plus1 == self.num_classes + 1, f'Number of classes in occ logits ({num_clas_plus1}) ' \
                                               f'does not match the number of classes in the head ({self.num_classes})'

        occ_labels_list = []
        panoptic_map_list = []

        for b in range(bs):

            occ_labels = occupancy_logits[b].argmax(dim=0)
            pred_obj_occ_score = F.sigmoid(pred_obj_occ_logits[b])

            # panoptic map is 0 if the voxel contains no object, otherwise it contains the object ID
            if len(pred_obj_occ_score) > 0:
                panoptic_map_from_offsets_max, panoptic_map_from_offsets_argmax = pred_obj_occ_score.max(dim=0)
                panoptic_map_from_offsets = panoptic_map_from_offsets_argmax + 1
                panoptic_map_from_offsets = panoptic_map_from_offsets * (panoptic_map_from_offsets_max > 0.5)

            else:
                panoptic_map_from_offsets = torch.zeros(self.occ_l, self.occ_w, self.occ_h, device=pred_obj_occ_score.device)

            panoptic_map = torch.zeros_like(panoptic_map_from_offsets)

            # find indices that contain objects from the main occupancy map
            mask = torch.zeros_like(occ_labels).bool()
            for j in self.obj_classes_indices:
                mask = torch.logical_or(mask, occ_labels == j)

            if panoptic_map_from_offsets.nonzero().sum() > 0 and mask.sum() > 0:

                if self.majority_voting:

                    mask_flatten = mask.flatten()

                    neighbor_indices = self.neighbor_indices[mask_flatten]
                    valid_mask = self.valid_mask[mask_flatten]

                    neighbors_labels = panoptic_map_from_offsets[neighbor_indices[..., 0], neighbor_indices[..., 1], neighbor_indices[..., 2]]

                    neighbors_labels = neighbors_labels * valid_mask
                    neighbors_labels_one_hot = torch.nn.functional.one_hot(neighbors_labels)
                    neighbors_labels_votes = neighbors_labels_one_hot.sum(dim=1)
                    neighbors_labels_votes[..., 0] = 0
                    ids = torch.argmax(neighbors_labels_votes, dim=-1)

                    panoptic_map[mask] = ids

                else:

                    indices_to_replace = torch.nonzero(mask)

                    # Get all coordinates and nonzero values for tensor B
                    coords_panoptic_map_from_offsets = torch.stack(torch.meshgrid(
                        torch.arange(l), torch.arange(w), torch.arange(h), indexing='ij'
                    ), dim=-1).reshape(-1, 3).to(occupancy_logits.device)  # Shape (N, 3)
                    values_panoptic_map_from_offsets = panoptic_map_from_offsets.reshape(-1)
                    nonzero_mask = values_panoptic_map_from_offsets != 0
                    nonzero_coords_panoptic_map_from_offsets = coords_panoptic_map_from_offsets[nonzero_mask]
                    nonzero_panoptic_map_from_offsets = values_panoptic_map_from_offsets[nonzero_mask]

                    # Iterate over all elements to replace
                    for idx in indices_to_replace:
                        # Compute Manhattan distances to nonzero elements in B
                        distances = torch.sum(torch.abs(nonzero_coords_panoptic_map_from_offsets - idx), dim=1)

                        # Find the index of the minimum distance
                        # min_idx = torch.argmin(distances)
                        min_dis, min_idx = torch.min(distances, dim=0)

                        if min_dis < self.max_voxel_distance:
                            # Replace element in A with the corresponding nonzero value from B
                            panoptic_map[tuple(idx)] = nonzero_panoptic_map_from_offsets[min_idx]

            occ_labels_list.append(occ_labels)
            panoptic_map_list.append(panoptic_map)

        pred_labels = torch.stack(occ_labels_list)
        panoptic_map = torch.stack(panoptic_map_list)

        return pred_labels, panoptic_map

    def loss(self, data_samples: List[OccDataSample], **kwargs) \
            -> dict[str, torch.Tensor]:

        return dict()

    def predict(self, batch_inputs: Dict, data_samples: list[OccDataSample]) -> List[OccDataSample]:

        bs = len(data_samples)

        if data_samples is None:
            raise ValueError('Data samples are None')

        pred_logits = []
        obj_pred_cls_logits_valid = []
        obj_pred_occ_logits_valid = []
        for data_sample in data_samples:
            if data_sample is None:
                raise ValueError('Data sample is None')

            pred_logits.append(data_sample.pred_occ_map_logits.logits)
            obj_pred_cls_logits_valid.append(data_sample.pred_instance_cls_logits_valid.scores)
            obj_pred_occ_logits_valid.append(data_sample.pred_instance_occ_logits_valid.masks)

        pred_logits = torch.stack(pred_logits)

        pred_labels, pred_panoptic_map = self(pred_logits, obj_pred_occ_logits_valid)

        out_data_samples = []

        for data_sample, logits, labels, panoptic_map in zip(data_samples, pred_logits, pred_labels, pred_panoptic_map):

            occ_data_pred_logits = OccupancyData()
            occ_data_pred_occ_map = OccupancyData()
            occ_data_pred_panoptic_map  = OccupancyData()

            occ_data_pred_logits['logits'] = logits
            occ_data_pred_occ_map['occ_map'] = labels
            occ_data_pred_panoptic_map['panoptic_map'] = panoptic_map

            data_sample.pred_occ_map_logits = occ_data_pred_logits
            data_sample.pred_occ_map = occ_data_pred_occ_map
            data_sample.pred_panoptic_map = occ_data_pred_panoptic_map
            out_data_samples.append(data_sample)

        if self.pred_point_labels:

            device = batch_inputs['imgs'].device

            pred_points_labels_list = []
            pred_points_ids_list = []
            # predict LiDAR semantic segmentation
            pred_grid_size = torch.tensor([[[[self.occ_l, self.occ_w, self.occ_h]]]], device=device)
            for b in range(bs):
                points = batch_inputs['points'][b]

                # normalize given pc_range
                points[..., 0] = (points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                points[..., 1] = (points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                points[..., 2] = (points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

                pre_points_indices = points * pred_grid_size
                pre_points_indices = torch.floor(pre_points_indices).long()

                # clamp indices
                pre_points_indices[..., 0] = torch.clamp(pre_points_indices[..., 0], 0, self.occ_l - 1)
                pre_points_indices[..., 1] = torch.clamp(pre_points_indices[..., 1], 0, self.occ_w - 1)
                pre_points_indices[..., 2] = torch.clamp(pre_points_indices[..., 2], 0, self.occ_h - 1)

                pre_points_indices = pre_points_indices.squeeze()

                # use indices to get the predicted labels
                _pred_labels = pred_labels[b].clone().squeeze()
                pred_points_labels = _pred_labels[pre_points_indices[..., 0], pre_points_indices[..., 1], pre_points_indices[..., 2]]

                # all points classified as free space set to ignore index 0
                pred_points_labels[pred_points_labels == self.num_classes] = 0

                # panoptic map
                pred_points_ids = pred_panoptic_map[b, pre_points_indices[..., 0], pre_points_indices[..., 1], pre_points_indices[..., 2]]

                pred_points_labels_list.append(pred_points_labels)
                pred_points_ids_list.append(pred_points_ids)

            for i in range(len(out_data_samples)):

                point_data_pred_labels = PointData()

                point_data_pred_labels['pts_semantic_mask'] = pred_points_labels_list[i]
                point_data_pred_labels['pts_instance_mask'] = pred_points_ids_list[i]

                out_data_samples[i].pred_pts_seg = point_data_pred_labels

        return out_data_samples
