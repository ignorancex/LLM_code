# code inspired by https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/detr_head.py
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.task_modules.assigners import HungarianAssigner
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import BaseModule
from mmengine.model import constant_init
from mmengine.structures import InstanceData

from offsetocc.registry import MODELS
from offsetocc.structures import OccDataSample
from offsetocc.utils import ConfigDict


@MODELS.register_module()
class PanopticHead(BaseModule):

    """
    Panoptic occupancy head prediction.

    This implementation supports object occupancy prediction from one decoder layer only.

    Args:
        occ_l (int): The length of the occupancy grid.
        occ_w (int): The width of the occupancy grid.
        occ_h (int): The height of the occupancy grid.
        embed_dims (int): The embedding dimension.
        num_classes (int): The number of classes.
        loss_cls_cfg (dict): Config of the loss for segmentation head.
    """

    def __init__(self,
                 occ_l: int,
                 occ_w: int,
                 occ_h: int,
                 embed_dims: int,
                 num_classes: int,
                 pc_range: List[float],
                 mask_camera: bool,
                 obj_classes_indices: List[int],
                 decode_ffn_cfg: ConfigDict,
                 scene_assigner_cfg: ConfigDict,
                 obj_assigner_cfg: ConfigDict,
                 num_offsets: int,
                 loss_scene_cls_cfg: ConfigDict,
                 loss_scene_reg_cfg: ConfigDict,
                 loss_scene_lwh_cfg: ConfigDict,
                 loss_obj_cls_cfg: ConfigDict,
                 loss_obj_occ_cfg: ConfigDict,
                 bg_cls_weight: float,
                 bg_occ_weight: float,
                 obj_occ_label_smoothing: float = 0.0,
                 obj_occ_voxel_center_noise: bool = False,
                 sync_avg_factor: bool = True,
                 center_ablation: bool = False) -> None:

        super().__init__()

        self.center_ablation = center_ablation

        self.occ_l = occ_l
        self.occ_w = occ_w
        self.occ_h = occ_h

        self.embed_dims = embed_dims

        self.num_classes = num_classes
        self.pc_range = pc_range
        self.mask_camera = mask_camera
        self.obj_classes_indices = obj_classes_indices
        self.cls_out_channels = num_classes + 1
        self.bg_cls_weight = bg_cls_weight
        self.bg_occ_weight = bg_occ_weight

        self.obj_occ_label_smoothing = obj_occ_label_smoothing
        self.obj_occ_voxel_center_noise = obj_occ_voxel_center_noise

        self.voxel_dim_x = (self.pc_range[3] - self.pc_range[0]) / self.occ_l
        self.voxel_dim_y = (self.pc_range[4] - self.pc_range[1]) / self.occ_w
        self.voxel_dim_z = (self.pc_range[5] - self.pc_range[2]) / self.occ_h

        self.sphere_init_radius = 4.5   # 4.5 meters

        self.num_offsets = num_offsets

        self.decode_ffn_cfg = decode_ffn_cfg

        self.sync_avg_factor = sync_avg_factor

        self.scene_assigner = HungarianAssigner(**scene_assigner_cfg)

        self.obj_assigner = HungarianAssigner(**obj_assigner_cfg)

        self.loss_scene_cls = MODELS.build(loss_scene_cls_cfg)
        self.loss_scene_reg = MODELS.build(loss_scene_reg_cfg)
        self.loss_scene_lwh = MODELS.build(loss_scene_lwh_cfg)
        self.loss_obj_cls = MODELS.build(loss_obj_cls_cfg)
        self.loss_obj_occ = MODELS.build(loss_obj_occ_cfg)

        self.decode_ffn = FFN(**self.decode_ffn_cfg)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.center_3d_pred = Linear(self.embed_dims, 3)
        self.lwh_pred = Linear(self.embed_dims, 3)
        self.occ_off_reg = Linear(self.embed_dims, self.num_offsets * 3)
        self.occ_off_cls = Linear(self.embed_dims, self.num_offsets)
        self._init_weights()

    def _init_weights(self) -> None:
        # constant_init(self.occ_off_reg, 0.)

        # def distribute_points_on_sphere(N, dtype):
        #     goldenRatio = (1 + 5 ** 0.5) / 2
        #     i = torch.arange(0, N, dtype=dtype)
        #     theta = 2 * torch.pi * i / goldenRatio
        #     phi = torch.acos(1 - 2 * (i + 0.5) / N)
        #     return theta, phi

        # thetas, phis = distribute_points_on_sphere(self.num_offsets, dtype=torch.float32)
        # sphere_init = torch.stack([thetas.cos() * phis.sin(), thetas.sin() * phis.sin(), phis.cos()], -1)
        # sphere_init *= self.sphere_init_radius

        # self.occ_off_reg.bias.data = sphere_init.view(-1)

        # constant_init(self.occ_off_cls, 0., bias=0.)

        constant_init(self.occ_off_reg, 0.)

        def distribute_points_on_grid(N) -> torch.Tensor:
            # take cubic root of N
            n = round(N ** (1 / 3))
            reso_x = (self.pc_range[3] - self.pc_range[0]) / self.occ_l
            reso_y = (self.pc_range[4] - self.pc_range[1]) / self.occ_w
            reso_z = (self.pc_range[5] - self.pc_range[2]) / self.occ_h
            # create a grid of n x n x n
            xs = torch.linspace(-(n // 2), n // 2, n, dtype=torch.float32).view(n, 1, 1).expand(n, n, n) * reso_x
            ys = torch.linspace(-(n // 2), n // 2, n, dtype=torch.float32).view(1, n, 1).expand(n, n, n) * reso_y
            zs = torch.linspace(0, n-1, n, dtype=torch.float32).view(1, 1, n).expand(n, n, n) * reso_z
            grid = torch.stack((xs, ys, zs), 3)
            #
            return grid

        self.occ_off_reg.bias.data = distribute_points_on_grid(self.num_offsets).view(-1)

        constant_init(self.occ_off_cls, 0., bias=0.)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, num_queries, c = queries.shape
        # (bs, num_queries, c)
        queries = self.decode_ffn(queries)

        # predict object label (bs, num_queries, num_classes + 1)
        obj_label_logits = self.fc_cls(queries)

        # predict the object centers in 3D
        # (bs, num_queries, 3)
        center_3d = self.center_3d_pred(queries)

        # predict the object dimensions in 3D
        # (bs, num_queries, 3)
        lwh = self.lwh_pred(queries)

        # predict the occupancy/point cloud of the object - occupancy
        # (bs, num_queries, num_offsets)
        obj_occ_logits = self.occ_off_cls(queries)

        # predict the occupancy/point cloud of the object - structure
        # (bs, num_queries, num_offsets, 3)
        obj_occ_offsets = self.occ_off_reg(queries).view(bs, num_queries, self.num_offsets, 3)

        return obj_label_logits, center_3d, lwh, obj_occ_logits, obj_occ_offsets

    def loss(self, queries: torch.Tensor, data_samples: List[OccDataSample], **kwargs) \
            -> Tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:

        bs = len(queries)

        # Forward pass
        (
            pred_obj_cls_logits_batch,
            pred_obj_center3d_batch,
            pred_obj_lwh_batch,
            pred_offsets_occ_logits_batch,
            pred_offsets_center_batch
        ) = self(queries)

        # Extract ground truth
        gt_obj_labels, gt_obj_centers3d, gt_obj_lwh, gt_obj_occ = self._get_gt(
            data_samples, bs)

        # # discretize predictions to pass to fusion head
        # pred_obj_cls_logits_valid, pred_obj_occ_logits_valid = self.discretize_for_fusion(
        #     pred_obj_cls_logits_batch, pred_obj_center3d_batch,
        #     pred_offsets_occ_logits_batch, pred_offsets_center_canonical_batch
        # )

        # Compute loss
        # TODO: classification loss to focal loss (both object and offsets level)
        #  - requires sigmoid activation???
        # TODO: stop loss when offsets is already inside the correct voxel? - test

        losses = dict()

        # compute matching at object level (object labels and object centers3d)
        #   matched labels (padded with background label)
        #   matched regression targets (padded with zeros)
        #   positive object mask (1 indicates a gt match)
        # TODO: compute mask of correctly classified positive objects - test (1)
        (
            matched_obj_labels,
            matched_obj_centers3d,
            pos_obj_mask,
            pos_assigned_gt_inds,
            pos_inds
        ) = self._match_gt(
            pred_obj_cls_logits_batch, pred_obj_center3d_batch,
            gt_obj_labels, gt_obj_centers3d,
            self.scene_assigner,
            self.num_classes
        )

        matched_obj_lwh = [torch.zeros_like(pred_obj_lwh_batch[i]) for i in range(bs)]
        for i in range(bs):
            matched_obj_lwh[i][pos_inds[i]] = gt_obj_lwh[i][pos_assigned_gt_inds[i]]

        matched_obj_lwh = torch.stack(matched_obj_lwh, dim=0)

        loss_obj_cls, loss_obj_center3d, loss_obj_lwh, avg_num_pos_obj = self._loss_obj_level(
            pred_obj_cls_logits_batch,
            pred_obj_center3d_batch,
            pred_obj_lwh_batch,
            matched_obj_labels,
            matched_obj_centers3d,
            matched_obj_lwh,
            pos_obj_mask
        )
        losses['loss_obj_cls'] = loss_obj_cls
        losses['loss_obj_center3d'] = loss_obj_center3d
        losses['loss_obj_lwh'] = loss_obj_lwh

        # Index positive objects offsets predictions
        # relevant_obj = pos_obj_mask
        # relevant_obj = pos_obj_mask * (pred_obj_cls_logits_batch.argmax(dim=-1) == matched_obj_labels) # TODO: (1)

        if pos_obj_mask.sum() == 0:
            losses['loss_offsets_occ'] = pred_offsets_occ_logits_batch.sum() * 0
            losses['loss_offsets_center3d'] = pred_offsets_center_batch.sum() * 0
            return losses #, pred_obj_cls_logits_valid, pred_obj_occ_logits_valid

        # pred_offsets_occ_logits = pred_offsets_occ_logits_batch[relevant_obj]
        # pred_offsets_center = pred_offsets_center_batch[relevant_obj]

        pred_offsets_occ_logits = torch.cat([pred_offsets_occ_logits_batch[i][pos_inds[i]] for i in range(bs)])
        pred_offsets_center = torch.cat([pred_offsets_center_batch[i][pos_inds[i]] for i in range(bs)])
        # assert torch.equal(pred_offsets_occ_logits, _pred_offsets_occ_logits)
        # assert torch.equal(pred_offsets_center, _pred_offsets_center)

        # obj_center3d = pred_obj_center3d_batch[relevant_obj]
        # obj_center3d = matched_obj_centers3d[relevant_obj]    # TODO: (2)
        # gt_obj_occ_matched = [
        #     gt_obj_occ[i][j]
        #     for i, pos_gt_inds_in_frame in enumerate(pos_assigned_gt_inds)
        #     for j in pos_gt_inds_in_frame
        # ]

        gt_obj_occ_matched = [gt_obj_occ[i][j] for i in range(bs) for j in pos_assigned_gt_inds[i]]
        obj_center3d = torch.cat([gt_obj_centers3d[i][pos_assigned_gt_inds[i]] for i in range(bs)])
        # assert all([torch.equal(gt_obj_occ_matched[i], _gt_obj_occ_matched[i]) for i in range(len(gt_obj_occ_matched))])
        # assert torch.equal(obj_center3d, _obj_center3d)

        # compute absolute offsets positions by summing with object center3d
        #  (TODO: predicted or gt center3d - test) (2)
        if self.center_ablation:
            _obj_center3d = torch.cat([pred_obj_center3d_batch[i][pos_inds[i]] for i in range(bs)])
            pred_abs_center = _obj_center3d.unsqueeze(1) + pred_offsets_center
        else:
            pred_abs_center = obj_center3d.unsqueeze(1) + pred_offsets_center

        # _pred_abs_center = _obj_center3d.unsqueeze(1) + _pred_offsets_center
        # assert torch.equal(pred_abs_center, _pred_abs_center)

        # compute matching at offsets level (offsets occupancy label and 3d offset position)
        #   matched offsets labels (padded with empty label)
        #   matched regression targets (padded with zeros)
        #   positive (occupied) offsets mask (1 indicates a gt match)
        #   compute mask1 of offsets with are inside the grid (1 indicates inside the grid)
        #   compute mask2 of valid offsets (either matched OR inside the grid)
        gt_offset_occ_label = [
            torch.ones((len(voxels_in_obj),), dtype=torch.long, device=voxels_in_obj.device)
            for voxels_in_obj in gt_obj_occ_matched
        ]

        matched_offsets_occ, matched_offsets_center3d, pos_offsets_mask, _, _ = self._match_gt(
            pred_offsets_occ_logits, pred_abs_center,
            gt_offset_occ_label,
            gt_obj_occ_matched,
            self.obj_assigner,
            0,
        )

        loss_offsets_occ, loss_offsets_center3d = self._loss_offset_level(
            pred_offsets_occ_logits,
            pred_abs_center,
            matched_offsets_occ,
            matched_offsets_center3d,
            pos_offsets_mask,
            avg_num_pos_obj
        )
        losses['loss_offsets_occ'] = loss_offsets_occ
        losses['loss_offsets_center3d'] = loss_offsets_center3d

        return losses #, pred_obj_cls_logits_valid, pred_obj_occ_logits_valid

    def _get_gt(
            self, data_samples: list[OccDataSample], batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]]]:
        batch_size = batch_size
        if self.mask_camera:
            device = data_samples[0].gt_instances_3d.labels_3d.device
            pred_grid_size = torch.tensor([[self.occ_l, self.occ_w, self.occ_h]], device=device)

        # (bs, num_visible_objects)
        gt_visible_obj_idx = [
            data_samples[i].gt_occ_instances.occ_map_visible_obj_idx
            for i in range(batch_size)
        ]

        # (bs, num_visible_objects * num_voxels_of_each_obj, 3)
        gt_obj_occ_grid_flat = [
            data_samples[i].gt_occ_grid_map.occ_map_visible_obj_occ_grid
            for i in range(batch_size)
        ]

        # (bs, num_visible_objects)
        gt_obj_occ_grid_start_idx = [
            data_samples[i].gt_occ_instances.occ_map_visible_obj_occ_grid_start_idx
            for i in range(batch_size)
        ]

        # rebuild objects using start idx
        gt_obj_occ_grid = []
        for i in range(batch_size):
            gt_obj_occ_grid.append([])
            for j in range(len(gt_obj_occ_grid_start_idx[i])):
                s = gt_obj_occ_grid_start_idx[i][j]
                e = gt_obj_occ_grid_start_idx[i][j + 1] if j + 1 < len(gt_obj_occ_grid_start_idx[i]) else None
                obj_voxels = gt_obj_occ_grid_flat[i][s:e]
                if self.mask_camera:
                    # keep only voxels that are visible from the cameras
                    obj_voxels_ = torch.clone(obj_voxels)
                    mask_camera = data_samples[i].gt_occ_map.occ_map_mask_camera
                    obj_voxels_[..., 0] = (obj_voxels_[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                    obj_voxels_[..., 1] = (obj_voxels_[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                    obj_voxels_[..., 2] = (obj_voxels_[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                    obj_voxels_indices = torch.floor(obj_voxels_ * pred_grid_size).long()
                    obj_visible_voxels_mask = mask_camera[obj_voxels_indices[..., 0], obj_voxels_indices[..., 1], obj_voxels_indices[..., 2]]
                    obj_voxels = obj_voxels[obj_visible_voxels_mask]
                gt_obj_occ_grid[i].append(obj_voxels)

        # (bs, num_visible_objects)
        gt_obj_label = [
            data_samples[i].gt_instances_3d.labels_3d[gt_visible_obj_idx[i]] for
            i in range(batch_size)]

        # (bs, num_visible_objects, 3)
        gt_obj_center3d = [
            data_samples[i].gt_instances_3d.bboxes_3d
            .center[gt_visible_obj_idx[i]]
            for i in range(batch_size)
        ]

        # (bs, num_visible_objects, 3)
        gt_obj_lwh = [
            data_samples[i].gt_instances_3d.bboxes_3d
            .dims[gt_visible_obj_idx[i]]
            for i in range(batch_size)
        ]

        return gt_obj_label, gt_obj_center3d, gt_obj_lwh, gt_obj_occ_grid

    def _match_gt(
            self,
            pred_logits: torch.Tensor,
            pred_loc: torch.Tensor,
            gt_cls: list[torch.Tensor],
            gt_loc: list[torch.Tensor],
            assigner: HungarianAssigner,
            bg_cls_label: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        gt_instances = [
            InstanceData(labels=cls, center3d=loc)
            for cls, loc in zip(gt_cls, gt_loc)
        ]

        matched_cls, matched_loc, pos_mask, pos_gt_inds, pos_inds = multi_apply(
            self._match_gt_single,
            pred_logits, pred_loc, gt_instances,
            assigner=assigner,
            bg_cls_label=bg_cls_label
        )

        matched_cls = torch.stack(matched_cls, 0)
        matched_loc = torch.stack(matched_loc, 0)
        pos_mask = torch.stack(pos_mask, 0).bool()

        return matched_cls, matched_loc, pos_mask, pos_gt_inds, pos_inds

    def _match_gt_single(
            self,
            pred_logits: torch.Tensor,
            pred_loc: torch.Tensor,
            gt_instance: InstanceData,
            assigner: HungarianAssigner,
            bg_cls_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pred = pred_loc.size(0)

        pred_instance = InstanceData(scores=pred_logits, center3d=pred_loc)
        assign_result = assigner.assign(
            pred_instances=pred_instance, gt_instances=gt_instance)

        gt_loc = gt_instance.center3d
        gt_cls = gt_instance.labels.long()
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # Mask
        matched_mask = torch.zeros(
            num_pred, dtype=pred_logits.dtype, device=pred_logits.device)
        matched_mask[pos_inds] = 1.0

        # Class targets
        cls_targets = gt_loc.new_full((num_pred,), bg_cls_label, dtype=torch.long)
        cls_targets[pos_inds] = gt_cls[pos_assigned_gt_inds]

        # Location targets
        loc_targets = torch.zeros_like(pred_loc, dtype=gt_loc.dtype)
        loc_targets[pos_inds] = gt_loc[pos_assigned_gt_inds.long(), :]

        return cls_targets, loc_targets, matched_mask, pos_assigned_gt_inds, pos_inds

    def _loss_obj_level(
            self,
            pred_logits: torch.Tensor,
            pred_loc: torch.Tensor,
            pred_obj_lwh: torch.Tensor,
            matched_labels: torch.Tensor,
            matched_loc: torch.Tensor,
            matched_lwh: torch.Tensor,
            pos_obj_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute object level classification and regression losses"""
        num_pos_obj = pos_obj_mask.sum().float()
        # average factor from all gpus
        if self.sync_avg_factor:
            avg_num_pos_obj = torch.clamp(reduce_mean(num_pos_obj), min=1).item()
        else:
            avg_num_pos_obj = max(num_pos_obj, 1)
        avg_num_neg_obj = len(pos_obj_mask) * pos_obj_mask.shape[1] - avg_num_pos_obj

        # Classification loss
        cls_avg_factor = avg_num_pos_obj * 1.0 + avg_num_neg_obj * self.bg_cls_weight
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_obj_cls = self.loss_scene_cls(
            pred_logits.reshape(-1, self.cls_out_channels),
            matched_labels.reshape(-1),
            avg_factor=cls_avg_factor
        )

        # Regression loss (center 3D)
        loss_obj_center3d = self.loss_scene_reg(
            pred_loc.reshape(-1, 3),
            matched_loc.reshape(-1, 3),
            pos_obj_mask.float().reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        loss_obj_lwh = self.loss_scene_lwh(
            pred_obj_lwh.reshape(-1, 3),
            matched_lwh.reshape(-1, 3),
            pos_obj_mask.float().reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        return loss_obj_cls, loss_obj_center3d, loss_obj_lwh, avg_num_pos_obj

    def _loss_offset_level(
            self,
            pred_logits: torch.Tensor,
            pred_abs_loc: torch.Tensor,
            matched_labels: torch.Tensor,
            matched_loc: torch.Tensor,
            pos_offsets_mask: torch.Tensor,
            avg_num_pos_obj: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute offsets level classification and regression losses."""

        in_grid_mask = torch.logical_and(
            pred_abs_loc[..., 0] >= self.pc_range[0],
            pred_abs_loc[..., 0] <= self.pc_range[3],
        ) & torch.logical_and(
            pred_abs_loc[..., 1] >= self.pc_range[1],
            pred_abs_loc[..., 1] <= self.pc_range[4],
        ) & torch.logical_and(
            pred_abs_loc[..., 2] >= self.pc_range[2],
            pred_abs_loc[..., 2] <= self.pc_range[5],
        )

        valid_offsets_mask = pos_offsets_mask | in_grid_mask

        # label smoothing
        matched_labels = (1 - self.obj_occ_label_smoothing) * matched_labels + self.obj_occ_label_smoothing * (1 - matched_labels)

        # Voxel occupancy loss
        off_occ_weights = valid_offsets_mask / valid_offsets_mask.sum(
            dim=-1).unsqueeze(-1)
        loss_offsets_occ = self.loss_obj_cls(
            pred_logits.reshape(-1),
            matched_labels.reshape(-1),
            off_occ_weights.reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        if self.obj_occ_voxel_center_noise:
            # add noise to the center of the voxel
            noise = torch.rand(matched_loc.size(), device=matched_loc.device)
            noise[..., 0] *= self.voxel_dim_x
            noise[..., 1] *= self.voxel_dim_y
            noise[..., 2] *= self.voxel_dim_z
            noise[..., 0] -= self.voxel_dim_x / 2
            noise[..., 1] -= self.voxel_dim_y / 2
            noise[..., 2] -= self.voxel_dim_z / 2
            matched_loc += noise

        # Voxel coordinates loss
        off_center3d_weights = pos_offsets_mask / pos_offsets_mask.sum(dim=-1).unsqueeze(-1)
        loss_offsets_center3d = self.loss_obj_occ(
            pred_abs_loc.reshape(-1, 3),
            matched_loc.reshape(-1, 3),
            off_center3d_weights.reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        return loss_offsets_occ, loss_offsets_center3d

    # def discretize_for_fusion(self, pred_obj_cls_logits_batch: torch.Tensor, pred_obj_center3d_batch: torch.Tensor,
    #                pred_offsets_occ_logits_batch: torch.Tensor, pred_offsets_center_batch: torch.Tensor):
    #
    #     # not really used for now, but yaw should be adapted here
    #
    #     bs = pred_obj_cls_logits_batch.size(0)
    #     num_queries = pred_obj_cls_logits_batch.size(1)
    #     device = pred_obj_cls_logits_batch.device
    #     dtype = pred_obj_cls_logits_batch.dtype
    #
    #     # pred_obj_label_scores_batch = F.softmax(pred_obj_cls_logits_batch, dim=-1).detach()
    #     pred_obj_label_scores_batch = pred_obj_cls_logits_batch.sigmoid().detach()    # focalloss
    #     pred_obj_cls_labels_batch = pred_obj_label_scores_batch.argmax(dim=-1, keepdim=True)
    #     pred_obj_absolute_batch = (pred_obj_center3d_batch.unsqueeze(2) + pred_offsets_center_batch).detach()
    #
    #     # normalize given pc_range
    #     pred_obj_absolute_batch[..., 0] = (pred_obj_absolute_batch[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
    #     pred_obj_absolute_batch[..., 1] = (pred_obj_absolute_batch[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
    #     pred_obj_absolute_batch[..., 2] = (pred_obj_absolute_batch[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
    #
    #     pred_grid_size = torch.tensor([[[[self.occ_l, self.occ_w, self.occ_h]]]], device=device)
    #     pred_obj_occ_logits_batch = torch.ones((bs, num_queries, self.occ_l, self.occ_w, self.occ_h), device=device, dtype=dtype) * float('-inf')
    #     pred_obj_occ_indices = pred_obj_absolute_batch * pred_grid_size
    #     pred_obj_occ_indices = torch.floor(pred_obj_occ_indices).long()
    #     # keep only valid indices (within the occupancy grid)
    #     pred_obj_occ_indices_mask = torch.logical_and(pred_obj_occ_indices[..., 0] >= 0,
    #                                                   pred_obj_occ_indices[..., 0] < self.occ_l) & \
    #                                 torch.logical_and(pred_obj_occ_indices[..., 1] >= 0,
    #                                                   pred_obj_occ_indices[..., 1] < self.occ_w) & \
    #                                 torch.logical_and(pred_obj_occ_indices[..., 2] >= 0,
    #                                                   pred_obj_occ_indices[..., 2] < self.occ_h)
    #
    #     # TODO: this code does not handle the special case where multiple offsets from an object point to the same voxel.
    #     #  In principle it should neve happen given how the the loss is computed, however their logits or scores
    #     #  should be averaged -----------> CHECK THIS
    #
    #     for i in range(bs):
    #         for j in range(num_queries):
    #             pred_obj_occ_indices_curr_obj = pred_obj_occ_indices[i, j]
    #             pred_obj_offsets_logits_batch_curr_obj = pred_offsets_occ_logits_batch[i, j]
    #             pred_obj_occ_indices_mask_curr_obj = torch.logical_and(pred_obj_occ_indices_mask[i, j], (pred_obj_offsets_logits_batch_curr_obj.sigmoid() > 0.5).bool())
    #             pred_obj_occ_indices_curr_obj = pred_obj_occ_indices_curr_obj[pred_obj_occ_indices_mask_curr_obj]
    #             pred_obj_offsets_logits_batch_curr_obj = pred_obj_offsets_logits_batch_curr_obj[pred_obj_occ_indices_mask_curr_obj]
    #             pred_obj_occ_logits_batch[i, j][pred_obj_occ_indices_curr_obj[:, 0], pred_obj_occ_indices_curr_obj[:, 1], pred_obj_occ_indices_curr_obj[:, 2]] = pred_obj_offsets_logits_batch_curr_obj
    #
    #     pred_obj_cls_logits_valid_batch = []
    #     pred_obj_occ_logits_valid_batch = []
    #
    #     for i in range(bs):
    #
    #         # (num_queries, num_classes + 1)
    #         pred_obj_cls_logits = pred_obj_cls_logits_batch[i]
    #         # (num_queries,)
    #         pred_obj_cls_labels = pred_obj_cls_labels_batch[i]
    #         # (num_queries, occ_l, occ_w, occ_h)
    #         pred_obj_occ_logits = pred_obj_occ_logits_batch[i]
    #
    #         # TODO: to be fixed later but is needed for visualization purposes atm (used by the fusion head)
    #         # keep only non-background
    #         valid_mask_nonbg = pred_obj_cls_labels != self.num_classes
    #         # and non-stuff classified objects
    #         valid_mask_nonst = torch.zeros_like(valid_mask_nonbg).bool()
    #         for j in self.obj_classes_indices:
    #             valid_mask_nonst = torch.logical_or(valid_mask_nonst, (pred_obj_cls_labels == j))
    #         valid_mask = torch.logical_and(valid_mask_nonbg, valid_mask_nonst)
    #
    #         valid_mask = valid_mask.squeeze(-1)
    #         # (num_non_bg_queries, num_classes + 1)
    #         pred_obj_cls_logits_valid = pred_obj_cls_logits[valid_mask]
    #         # (num_non_bg_queries, occ_l, occ_w, occ_h)
    #         pred_obj_occ_logits_valid = pred_obj_occ_logits[valid_mask]
    #
    #         pred_obj_cls_logits_valid_batch.append(pred_obj_cls_logits_valid)
    #         pred_obj_occ_logits_valid_batch.append(pred_obj_occ_logits_valid)
    #
    #     return pred_obj_cls_logits_valid_batch, pred_obj_occ_logits_valid_batch

    def predict(self, queries: torch.Tensor,
                data_samples: Optional[List[Optional[OccDataSample]]] = None) -> List[OccDataSample]:
        # convert predicted labels from bbox type to occ type
        (pred_obj_cls_logits_batch, pred_obj_center3d_batch, pred_obj_lwh_batch,
         pred_offsets_occ_logits_batch, pred_offsets_center_batch) = self(queries)

        bs = pred_obj_cls_logits_batch.size(0)
        num_queries = pred_obj_cls_logits_batch.size(1)
        device = pred_obj_cls_logits_batch.device

        # pred_obj_label_scores_batch = F.softmax(pred_obj_cls_logits_batch, dim=-1).detach()
        pred_obj_label_scores_batch = pred_obj_cls_logits_batch.sigmoid().detach()    # focalloss
        pred_obj_cls_labels_batch = pred_obj_label_scores_batch.argmax(dim=-1, keepdim=True)
        pred_obj_absolute_batch = (pred_obj_center3d_batch.unsqueeze(2) + pred_offsets_center_batch).detach()

        # normalize given pc_range
        pred_obj_absolute_batch[..., 0] = (pred_obj_absolute_batch[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        pred_obj_absolute_batch[..., 1] = (pred_obj_absolute_batch[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        pred_obj_absolute_batch[..., 2] = (pred_obj_absolute_batch[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        pred_grid_size = torch.tensor([[[[self.occ_l, self.occ_w, self.occ_h]]]], device=device)
        pred_obj_occ_logits_batch = torch.ones((bs, num_queries, self.occ_l, self.occ_w, self.occ_h), device=device) * float('-inf')
        pred_obj_occ_indices = pred_obj_absolute_batch * pred_grid_size
        pred_obj_occ_indices = torch.floor(pred_obj_occ_indices).long()
        # keep only valid indices (within the occupancy grid)
        pred_obj_occ_indices_mask = torch.logical_and(pred_obj_occ_indices[..., 0] >= 0,
                                                      pred_obj_occ_indices[..., 0] < self.occ_l) & \
                                    torch.logical_and(pred_obj_occ_indices[..., 1] >= 0,
                                                      pred_obj_occ_indices[..., 1] < self.occ_w) & \
                                    torch.logical_and(pred_obj_occ_indices[..., 2] >= 0,
                                                      pred_obj_occ_indices[..., 2] < self.occ_h)

        # TODO: this code does not handle the special case where multiple offsets from an object point to the same voxel.
        #  In principle it should neve happen given how the the loss is computed, however their logits or scores
        #  should be averaged -----------> CHECK THIS

        for i in range(bs):
            for j in range(num_queries):
                pred_obj_occ_indices_curr_obj = pred_obj_occ_indices[i, j]
                pred_obj_offsets_logits_batch_curr_obj = pred_offsets_occ_logits_batch[i, j]
                pred_obj_occ_indices_mask_curr_obj = torch.logical_and(pred_obj_occ_indices_mask[i, j], (pred_obj_offsets_logits_batch_curr_obj.sigmoid() > 0.5).bool())
                pred_obj_occ_indices_curr_obj = pred_obj_occ_indices_curr_obj[pred_obj_occ_indices_mask_curr_obj]
                pred_obj_offsets_logits_batch_curr_obj = pred_obj_offsets_logits_batch_curr_obj[pred_obj_occ_indices_mask_curr_obj]
                pred_obj_occ_logits_batch[i, j][pred_obj_occ_indices_curr_obj[:, 0], pred_obj_occ_indices_curr_obj[:, 1], pred_obj_occ_indices_curr_obj[:, 2]] = pred_obj_offsets_logits_batch_curr_obj

        # pred_obj_occ_scores_batch = F.sigmoid(pred_obj_occ_logits_batch).detach()
        # pred_obj_occ_map_batch = (pred_obj_occ_scores_batch > 0.5).float()

        out_data_samples = []

        if data_samples is None:
            data_samples = [None for _ in range(bs)]

        for i in range(bs):

            data_sample = data_samples[i]
            if data_sample is None:
                data_sample = OccDataSample()

            # (num_queries, num_classes + 1)
            pred_obj_cls_logits = pred_obj_cls_logits_batch[i]
            # (num_queries,)
            pred_obj_cls_labels = pred_obj_cls_labels_batch[i]
            # (num_queries, 3)
            pred_obj_center3d = pred_obj_center3d_batch[i]
            # (num_queries, 3)
            pred_obj_lwh = pred_obj_lwh_batch[i]
            # (num_queries, num_offsets, 3)
            pred_obj_offsets = pred_offsets_center_batch[i]
            # (num_queries, num_offsets)
            pred_obj_offsets_occ_logits = pred_offsets_occ_logits_batch[i]
            # (num_queries, occ_l, occ_w, occ_h)
            pred_obj_occ_logits = pred_obj_occ_logits_batch[i]
            # (num_queries, occ_l, occ_w, occ_h)
            # pred_obj_occ_map = pred_obj_occ_map_batch[i]

            # TODO: to be fixed later but is needed for visualization purposes atm (used by the fusion head)
            # keep only non-background
            valid_mask_nonbg = pred_obj_cls_labels != self.num_classes
            # and non-stuff classified objects
            valid_mask_nonst = torch.zeros_like(valid_mask_nonbg).bool()
            for j in self.obj_classes_indices:
                valid_mask_nonst = torch.logical_or(valid_mask_nonst, (pred_obj_cls_labels == j))
            valid_mask = torch.logical_and(valid_mask_nonbg, valid_mask_nonst)

            valid_mask = valid_mask.squeeze(-1)
            # (num_non_bg_queries, num_classes + 1)
            pred_obj_cls_logits_valid = pred_obj_cls_logits[valid_mask]
            # (num_non_bg_queries,)
            # pred_obj_cls_labels_valid = pred_obj_cls_labels[valid_mask]
            # (num_non_bg_queries, occ_l, occ_w, occ_h)
            pred_obj_occ_logits_valid = pred_obj_occ_logits[valid_mask]
            # (num_non_bg_queries, occ_l, occ_w, occ_h)
            # pred_obj_occ_map_valid = pred_obj_occ_map[valid_mask]

            inst_data_pred_logits_valid = InstanceData(scores=pred_obj_cls_logits_valid)
            # inst_data_pred_labels_valid = InstanceData(labels=pred_obj_cls_labels_valid)
            inst_data_pred_occ_logits_valid = InstanceData(masks=pred_obj_occ_logits_valid)
            # inst_data_pred_occ_map_valid = InstanceData(masks=pred_obj_occ_map_valid)

            data_sample.pred_instance_cls_logits_valid = inst_data_pred_logits_valid
            # data_sample.pred_instance_cls_labels_valid = inst_data_pred_labels_valid
            data_sample.pred_instance_occ_logits_valid = inst_data_pred_occ_logits_valid
            # data_sample.pred_instance_occ_map_valid = inst_data_pred_occ_map_valid

            # pack predictions in original shapes to compute val loss
            data_sample.pred_obj_cls_logits = pred_obj_cls_logits
            data_sample.pred_obj_center3d = pred_obj_center3d
            data_sample.pred_obj_lwh = pred_obj_lwh
            data_sample.pred_obj_offsets = pred_obj_offsets
            data_sample.pred_obj_offsets_occ_logits = pred_obj_offsets_occ_logits

            out_data_samples.append(data_sample)

        return out_data_samples