import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lib.criterions
from lib.utils.builder import MODEL, LOSS, build_loss
from lib.utils.config import CN
from lib.utils.net_utils import load_weights
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.recorder import Recorder
from lib.utils.transform import gen_random_rotation, gen_random_direction
from lib.metrics import LossMetric

from .backbones import StackedHourglass
from .heads import PointNetContactHead
from .model_abc import ModelABC


@MODEL.register_module()
class PiCRI(ModelABC):

    def __init__(self, cfg: CN):
        super(PiCRI, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.hg_stacks = cfg.HG_STACKS
        self.hg_blocks = cfg.HG_BLOCKS
        self.hg_classes = cfg.HG_CLASSES
        self.mean_offset = cfg.MEAN_OFFSET
        self.std_offset = cfg.STD_OFFSET
        self.maximal_angle = cfg.get("MAXIMAL_ANGLE", 180 / 24)
        self.vc_thresh = cfg.VC_THRESH
        self.mid_ft_size = cfg.HG_CLASSES  # intermediate feature size
        self.img_size = cfg.DATA_PRESET.IMAGE_SIZE  # (W, H)
        self.obj_scale_factor = cfg.OBJ_SCALE_FACTOR
        self.obj_verts_mode = cfg.OBJ_VERTS_MODE
        assert self.obj_verts_mode in ["gt", "pred"]

        self.base_net = StackedHourglass(cfg.HG_STACKS, cfg.HG_BLOCKS, cfg.HG_CLASSES)
        self.contact_head = PointNetContactHead(feat_dim=self.mid_ft_size + 1, n_region=17, n_anchor=4)

        if cfg.get("LOSS", None) is not None:
            self.criterion = build_loss(cfg.LOSS, data_preset=cfg.DATA_PRESET)
            self.loss_metric = LossMetric(cfg.LOSS)

        load_weights(self, pretrained=cfg.PRETRAINED, strict=False)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq

    def inference_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch, is_train=False, **kwargs)
        output = preds[f"stack{self.hg_stacks - 1}"]  # @NOTE: only use last stack as final output.

        if "callback" in kwargs and callable(kwargs["callback"]):
            kwargs["callback"](output, batch, step_idx, **kwargs)
        return output

    def training_step(self, batch, step_idx, **kwargs):
        batch_size = batch["image"].shape[0]
        results = self._forward_impl(batch, is_train=True, **kwargs)
        loss, loss_dict = self.compute_loss(gts=batch, preds=results, **kwargs)

        self.loss_metric.feed(loss_dict[f"stack{self.hg_stacks - 1}"], batch_size)  # only record last lvl.

        if step_idx % self.log_freq == 0:
            self.summary.add_scalar("train/loss", loss, step_idx)
            for k, v in loss_dict[f"stack{self.hg_stacks - 1}"].items():  # only summary last lvl.
                self.summary.add_scalar(f"train/loss_{k}", v, step_idx)

        return results, loss_dict

    def testing_step(self, batch, batch_idx, **kwargs):
        prd = self.inference_step(batch, batch_idx, **kwargs)
        return prd, {}

    def validation_step(self, batch, batch_idx, **kwargs):
        prd, acc = self.inference_step(batch, batch_idx, **kwargs)
        return prd, {}

    def on_train_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def on_val_finished(self, recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        pass

    def _forward_impl(self, batch, is_train=False, **kwargs):
        """forward pass of PiCRI

        Args:
            batch (dict): input of batch data
            is_train (bool, optional): Defaults to False.

        Returns:
            dict: results of all stacks
        """
        image = batch["image"]
        cam_intr = batch["target_cam_intr"]
        ls_hg_feature, _ = self.base_net(image)  # prefix [ ls_ ] = list

        if self.obj_verts_mode == "pred":
            # @NOTE: from preceding model prediction.
            obj_verts_3d = batch["recov_obj_verts3d"]
        elif self.obj_verts_mode == "gt":
            obj_verts_3d = batch["target_obj_verts_3d"].float()
        else:
            raise ValueError(f"Unknown obj_verts_mode {self.obj_verts_mode}")

        lvl_results = {}
        for i_stack in range(self.base_net.nstacks):  # RANGE 2
            i_hg_feature = ls_hg_feature[i_stack]  # TENSOR[NBATCH, 64, 1/4 ?, 1/4 ?]
            i_contact_results = self._picri_forward(cam_intr, obj_verts_3d, i_hg_feature, is_train)
            lvl_results[f"stack{i_stack}"] = i_contact_results

        return lvl_results

    def _picri_forward(self, cam_intr, object_vert_3d, low_level_feature_map, is_train):
        """forward pass of PiCRI per stack

        Args:
            cam_intr (Tensor): (B, 3, 3) 
            object_vert_3d (Tensor): (B, NO, 3)
            low_level_feature_map (Tensor): (B, 64, 1/4 IMGH, 1/4 IMGW)
            is_train (bool): whether in training mode
        Returns:
            dict: results of each stack
        """
        # * ===== random perturb object pose >>>>>>>>
        if is_train:
            # generate a random_rotation
            batch_size = object_vert_3d.shape[0]
            rand_rot = gen_random_rotation(np.deg2rad(self.maximal_angle)).expand(batch_size, -1,
                                                                                  -1).to(object_vert_3d.device)
            mean_obj_v = torch.mean(object_vert_3d, dim=1, keepdim=True)  # TENSOR[NBATCH, 1, 3]
            object_vert_3d = (torch.bmm(rand_rot,
                                        (object_vert_3d - mean_obj_v).permute(0, 2, 1)).permute(0, 2, 1) + mean_obj_v)

            # generate a random_direction
            dir_vec = gen_random_direction()
            rand_dist = torch.normal(torch.Tensor([self.mean_offset]), torch.Tensor([self.std_offset]))
            offset = rand_dist * dir_vec
            offset = offset.to(object_vert_3d.device)
            object_vert_3d = object_vert_3d + offset

        # * ===== STAGE 1, index the features >>>>>>>>
        reprojected_vert = torch.bmm(cam_intr, object_vert_3d.transpose(1, 2)).transpose(1, 2)
        reprojected_vert = reprojected_vert[:, :, :2] / reprojected_vert[:, :, 2:]  # TENSOR[NBATCH, NPOINT, 2]

        image_center_coord = torch.tensor(self.img_size, device=object_vert_3d.device).float() / 2  # TENSOR[2]
        image_center_coord = image_center_coord.view((1, 1, 2))  # TENSOR[1, 1, 2]
        reprojected_grid = (reprojected_vert - image_center_coord) / image_center_coord  # TENSOR[NBATCH, NPOINT, 2]
        # compute the in image mask, so that the points fall out of the image can be filtered when calculating loss
        in_image_mask = ((reprojected_grid[:, :, 0] >= -1.0) & (reprojected_grid[:, :, 0] <= 1.0) &
                         (reprojected_grid[:, :, 1] >= -1.0) & (reprojected_grid[:, :, 1] <= 1.0))
        in_image_mask = in_image_mask.float()
        # reshape reprojected_grid so that it fits the torch grid_sample interface
        reprojected_grid = reprojected_grid.unsqueeze(2)  # TENSOR[NBATCH, NPOINT, 1, 2]
        # by default. grid sampling have zero padding
        # those points get outside of current featmap will have feature vector all zeros
        collected_features = F.grid_sample(low_level_feature_map, reprojected_grid,
                                           align_corners=True)  # [NBATCH, 64, NPOINT, 1]

        # * ===== STAGE 2, concatenate the geometry features >>>>>>>>
        # @NOTE: z_normed = (z - 0.4)/focal, we uses the focal normalized z value.
        focal = cam_intr[:, :1, :1]
        object_vert_3d_z = object_vert_3d[:, :, 2:]  # TENSOR(B, N, 1)
        normed_object_vert_3d_z = ((object_vert_3d_z - 0.4) / focal) / self.obj_scale_factor
        normed_object_vert_3d_z = normed_object_vert_3d_z.unsqueeze(1)  # TENSOR(B, 1, N, 1)
        collected_features = torch.cat((collected_features, normed_object_vert_3d_z), dim=1)  # TENSOR(B, 65, N, 1)

        # * ===== STAGE 3, pass to contact head for VC, CR and AE >>>>>>>>>
        vertex_contact_pred, contact_region_pred, anchor_elasti_pred = self.contact_head(collected_features)

        # for computing loss
        results = {
            "vertex_contact": vertex_contact_pred,
            "contact_in_image_mask": in_image_mask,
            "contact_region": contact_region_pred,
            "anchor_elasti": anchor_elasti_pred,
        }

        # for output as results
        # @NOTE: argmax is not differentiable
        results[f'recov_vertex_contact'] = (torch.sigmoid(vertex_contact_pred) > self.vc_thresh).bool()  # TENSOR[B, N]
        results[f'recov_contact_region'] = torch.argmax(contact_region_pred, dim=2)  # TENSOR[B, N]
        results[f"recov_anchor_elasti"] = anchor_elasti_pred

        return results

    def compute_loss(self, preds, gts, **kwargs):
        """compute loss for each stack and sum them up

        Args:
            gts (dict): batch data
            preds (dict): results from forward
        """
        loss = 0
        loss_dict = {}
        for i in range(self.hg_stacks):
            i_loss, i_loss_dict = self.criterion(preds[f"stack{i}"], gts)
            loss += i_loss
            loss_dict[f"stack{i}"] = i_loss_dict

        loss_dict["loss"] = loss  # @NOTE: this `loss` is used for computing gradient (backward)
        return loss, loss_dict

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
