from typing import List
from typing import Sequence

import torch
from mmdet.models.task_modules.assigners import HungarianAssigner
from mmdet.models.utils import multi_apply
from mmengine.evaluator import BaseMetric
from mmengine.structures import InstanceData

from offsetocc.registry import METRICS
from offsetocc.registry import MODELS
from offsetocc.utils import ConfigDict


@METRICS.register_module()
class PanopticLossMetric2Stage(BaseMetric):

    def __init__(self,
                 num_classes: int,
                 pc_range: List[float],
                 scene_assigner_cfg: ConfigDict,
                 obj_assigner_cfg: ConfigDict,
                 loss_scene_cls_cfg: ConfigDict,
                 loss_scene_reg_cfg: ConfigDict,
                 loss_scene_lwh_cfg: ConfigDict,
                 loss_obj_cls_cfg: ConfigDict,
                 loss_obj_occ_cfg: ConfigDict,
                 bg_cls_weight: float,
                 bg_occ_weight: float,
                 mask_camera_panoptic: bool,
                 occ_l: int,
                 occ_w: int,
                 occ_h: int,
                 obj_classes_indices: List[int],
                 ):

        super(PanopticLossMetric2Stage, self).__init__()

        self.num_classes = num_classes
        self.pc_range = pc_range
        self.cls_out_channels = num_classes + 1
        self.bg_cls_weight = bg_cls_weight
        self.bg_occ_weight = bg_occ_weight

        self.scene_assigner_cfg = HungarianAssigner(**scene_assigner_cfg)

        self.obj_assigner = HungarianAssigner(**obj_assigner_cfg)

        self.loss_scene_cls = MODELS.build(loss_scene_cls_cfg)
        self.loss_scene_reg = MODELS.build(loss_scene_reg_cfg)
        self.loss_scene_lwh = MODELS.build(loss_scene_lwh_cfg)
        self.loss_obj_cls = MODELS.build(loss_obj_cls_cfg)
        self.loss_obj_occ = MODELS.build(loss_obj_occ_cfg)

        self.mask_camera_pano = mask_camera_panoptic
        self.occ_l = occ_l
        self.occ_w = occ_w
        self.occ_h = occ_h

        self.obj_classes_indices = obj_classes_indices


    # def compute_panoptic_loss(self):
    #     pass

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.
        """

        # define prediction lists
        pred_logits_batch_list = []

        pred_fusion_logits_batch_list = []

        pred_obj_cls_logits_batch_list = []
        pred_obj_center3d_batch_list = []
        pred_obj_lwh_batch_list = []
        pred_obj_occ_logits_batch_list = []
        pred_offsets_center_batch_list = []

        # define gt lists
        gt_obj_labels = []
        gt_obj_centers3d = []
        gt_obj_lwhs = []
        gt_obj_occ = []

        for i, data_sample in enumerate(data_samples):

            mask_camera = data_batch['data_samples'][i].gt_occ_map.occ_map_mask_camera.squeeze()

            # extract occ prediction
            pred_logits = data_sample['pred_occ_map_logits']['logits'].squeeze()

            # extract fusion prediction
            pred_fusion_logits = data_sample['pred_occ_map_logits']['logits'].squeeze()

            # extract panoptic prediction
            pred_obj_cls_logits = data_sample['pred_obj_cls_logits']
            pred_obj_center3d = data_sample['pred_obj_center3d']
            pred_obj_lwh = data_sample['pred_obj_lwh']
            pred_offsets_occ_logits = data_sample['pred_obj_offsets_occ_logits']
            pred_offsets_center = data_sample['pred_obj_offsets']

            mask_camera = mask_camera.to(pred_logits.device)

            # append predictions to the list
            pred_logits_batch_list.append(pred_logits)

            pred_fusion_logits_batch_list.append(pred_fusion_logits)

            pred_obj_cls_logits_batch_list.append(pred_obj_cls_logits)
            pred_obj_center3d_batch_list.append(pred_obj_center3d)
            pred_obj_lwh_batch_list.append(pred_obj_lwh)
            pred_obj_occ_logits_batch_list.append(pred_offsets_occ_logits)
            pred_offsets_center_batch_list.append(pred_offsets_center)

            # extract panoptic target
            # (num_visible_objects)
            gt_visible_obj_idx = data_samples[i]['gt_occ_instances']['occ_map_visible_obj_idx']
            # (num_visible_objects * num_voxels_of_each_obj, 3)
            gt_obj_occ_grid_flat = data_samples[i]['gt_occ_grid_map']['occ_map_visible_obj_occ_grid']
            # (num_visible_objects)
            gt_obj_occ_grid_start_idx = data_samples[i]['gt_occ_instances']['occ_map_visible_obj_occ_grid_start_idx']

            # rebuild objects using start idx
            gt_obj_occ_grid = []
            for j in range(len(gt_obj_occ_grid_start_idx)):
                s = gt_obj_occ_grid_start_idx[j]
                e = gt_obj_occ_grid_start_idx[j + 1] if j + 1 < len(gt_obj_occ_grid_start_idx) else None
                obj_voxels = gt_obj_occ_grid_flat[s:e]
                if self.mask_camera_pano:
                    # keep only voxels that are visible from the cameras
                    pred_grid_size = torch.tensor([[self.occ_l, self.occ_w, self.occ_h]], device=obj_voxels.device)
                    obj_voxels_ = torch.clone(obj_voxels)
                    obj_voxels_[..., 0] = (obj_voxels_[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                    obj_voxels_[..., 1] = (obj_voxels_[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                    obj_voxels_[..., 2] = (obj_voxels_[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                    obj_voxels_indices = torch.floor(obj_voxels_ * pred_grid_size).long()
                    obj_visible_voxels_mask = mask_camera[
                        obj_voxels_indices[..., 0], obj_voxels_indices[..., 1], obj_voxels_indices[..., 2]]
                    obj_voxels = obj_voxels[obj_visible_voxels_mask]
                gt_obj_occ_grid.append(obj_voxels)

            # (num_visible_objects)
            gt_obj_label = data_samples[i]['gt_instances_3d']['labels_3d'][gt_visible_obj_idx]
            gt_obj_center3d = data_samples[i]['gt_instances_3d']['bboxes_3d'].center[gt_visible_obj_idx]
            gt_obj_lwh = data_samples[i]['gt_instances_3d']['bboxes_3d'].dims[gt_visible_obj_idx]

            gt_obj_labels.append(gt_obj_label)
            gt_obj_centers3d.append(gt_obj_center3d)
            gt_obj_lwhs.append(gt_obj_lwh)
            gt_obj_occ.append(gt_obj_occ_grid)

        bs = len(data_samples)

        pred_obj_cls_logits_batch = torch.stack(pred_obj_cls_logits_batch_list)
        pred_obj_center3d_batch = torch.stack(pred_obj_center3d_batch_list)
        pred_obj_lwh_batch = torch.stack(pred_obj_lwh_batch_list)
        pred_offsets_occ_logits_batch = torch.stack(pred_obj_occ_logits_batch_list)
        pred_offsets_center_batch = torch.stack(pred_offsets_center_batch_list)

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
            matched_obj_labels, matched_obj_centers3d, pos_obj_mask, pos_assigned_gt_inds, pos_inds
         ) = self._match_gt(pred_obj_cls_logits_batch, pred_obj_center3d_batch, gt_obj_labels, gt_obj_centers3d,
                            self.scene_assigner_cfg, self.num_classes)

        matched_obj_lwh = [torch.zeros_like(pred_obj_lwh_batch[i]) for i in range(bs)]
        for i in range(bs):
            matched_obj_lwh[i][pos_inds[i]] = gt_obj_lwhs[i][pos_assigned_gt_inds[i]]

        matched_obj_lwh = torch.stack(matched_obj_lwh, dim=0)

        (loss_obj_cls, loss_obj_center3d, loss_obj_lwh, err_obj_center3d, err_obj_center3d_x, err_obj_center3d_y, err_obj_center3d_z,
         avg_num_pos_obj) = self._loss_obj_level(
            pred_obj_cls_logits_batch,
            pred_obj_center3d_batch,
            pred_obj_lwh_batch,
            matched_obj_labels,
            matched_obj_centers3d,
            matched_obj_lwh,
            pos_obj_mask
        )
        losses['loss_obj_cls_val'] = loss_obj_cls.cpu()
        losses['loss_obj_center3d_val'] = loss_obj_center3d.cpu()
        losses['loss_obj_lwh_val'] = loss_obj_lwh.cpu()

        if avg_num_pos_obj > 0:
            losses['err_obj_center3d_val'] = err_obj_center3d.cpu()
            losses['err_obj_center3d_x_val'] = err_obj_center3d_x.cpu()
            losses['err_obj_center3d_y_val'] = err_obj_center3d_y.cpu()
            losses['err_obj_center3d_z_val'] = err_obj_center3d_z.cpu()

        # Index positive objects offsets predictions
        # relevant_obj = pos_obj_mask
        # relevant_obj = pos_obj_mask * (pred_obj_cls_logits_batch.argmax(dim=-1) == matched_obj_labels) # TODO: (1)

        if pos_obj_mask.sum() == 0:
            losses['loss_offsets_occ_val'] = pred_offsets_occ_logits_batch.sum().cpu() * 0
            losses['loss_offsets_center3d_val'] = pred_offsets_center_batch.sum().cpu() * 0
        else:
            # pred_offsets_occ_logits = pred_offsets_occ_logits_batch[relevant_obj]
            # pred_offsets_center = pred_offsets_center_batch[relevant_obj]

            pred_offsets_occ_logits = torch.cat([pred_offsets_occ_logits_batch[i][pos_inds[i]] for i in range(bs)])
            pred_offsets_center = torch.cat([pred_offsets_center_batch[i][pos_inds[i]] for i in range(bs)])

            # obj_center3d = pred_obj_center3d_batch[relevant_obj]
            # obj_center3d = matched_obj_centers3d[relevant_obj]  # TODO: (2)
            # gt_obj_occ_matched = [
            #     gt_obj_occ[i][j]
            #     for i, pos_gt_inds_in_frame in enumerate(pos_assigned_gt_inds)
            #     for j in pos_gt_inds_in_frame
            # ]

            gt_obj_occ_matched = [gt_obj_occ[i][j] for i in range(bs) for j in pos_assigned_gt_inds[i]]

            obj_center3d = torch.cat([gt_obj_centers3d[i][pos_assigned_gt_inds[i]] for i in range(bs)])

            # compute absolute offsets positions by summing with object center3d
            #  (TODO: predicted or gt center3d - test) (2)
            pred_abs_center = obj_center3d.unsqueeze(1) + pred_offsets_center

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
            losses['loss_offsets_occ_val'] = loss_offsets_occ.cpu()
            losses['loss_offsets_center3d_val'] = loss_offsets_center3d.cpu()

        losses['loss_val'] = torch.sum(torch.stack([v for k, v in losses.items() if 'loss' in k]))

        self.results.append(losses)

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

    def _loss_obj_level(
            self,
            pred_logits: torch.Tensor,
            pred_loc: torch.Tensor,
            pred_obj_lwh: torch.Tensor,
            matched_labels: torch.Tensor,
            matched_loc: torch.Tensor,
            matched_lwh: torch.Tensor,
            pos_obj_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute object level classification and regression losses"""
        num_pos_obj = pos_obj_mask.sum().float()
        # average factor from all gpus
        num_pos_obj = torch.clamp(num_pos_obj, min=1).item()
        num_neg_obj = len(pos_obj_mask) * pos_obj_mask.shape[1] - num_pos_obj

        # Classification loss
        cls_avg_factor = num_pos_obj * 1.0 + num_neg_obj * self.bg_cls_weight
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
            avg_factor=num_pos_obj
        )

        loss_obj_lwh = self.loss_scene_lwh(
            pred_obj_lwh.reshape(-1, 3),
            matched_lwh.reshape(-1, 3),
            pos_obj_mask.float().reshape(-1),
            avg_factor=num_pos_obj
        )

        err_obj_center3d = (torch.norm(pred_loc - matched_loc, dim=-1) * pos_obj_mask).sum() / num_pos_obj
        err_obj_center3d_x = (torch.abs(pred_loc[..., 0] - matched_loc[..., 0]) * pos_obj_mask).sum() / num_pos_obj
        err_obj_center3d_y = (torch.abs(pred_loc[..., 1] - matched_loc[..., 1]) * pos_obj_mask).sum() / num_pos_obj
        err_obj_center3d_z = (torch.abs(pred_loc[..., 2] - matched_loc[..., 2]) * pos_obj_mask).sum() / num_pos_obj

        return loss_obj_cls, loss_obj_center3d, loss_obj_lwh, err_obj_center3d, err_obj_center3d_x, err_obj_center3d_y, err_obj_center3d_z, num_pos_obj

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

        # Voxel occupancy loss
        off_occ_weights = valid_offsets_mask / valid_offsets_mask.sum(
            dim=-1).unsqueeze(-1)
        loss_offsets_occ = self.loss_obj_cls(
            pred_logits.reshape(-1),
            matched_labels.reshape(-1),
            off_occ_weights.reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        # Voxel coordinates loss
        off_center3d_weights = pos_offsets_mask / pos_offsets_mask.sum(dim=-1).unsqueeze(-1)
        loss_offsets_center3d = self.loss_obj_occ(
            pred_abs_loc.reshape(-1, 3),
            matched_loc.reshape(-1, 3),
            off_center3d_weights.reshape(-1),
            avg_factor=avg_num_pos_obj
        )

        return loss_offsets_occ, loss_offsets_center3d

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

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.
        """

        metrics = {
            'loss_val': [],
            'loss_obj_cls_val': [],
            'loss_obj_center3d_val': [],
            'loss_obj_lwh_val': [],
            'err_obj_center3d_val': [],
            'err_obj_center3d_x_val': [],
            'err_obj_center3d_y_val': [],
            'err_obj_center3d_z_val': [],
            'loss_offsets_occ_val': [],
            'loss_offsets_center3d_val': [],
        }

        for result in self.results:
            for key, value in result.items():
                metrics[key].append(value)

        return {key: sum(value) / len(value) for key, value in metrics.items()}