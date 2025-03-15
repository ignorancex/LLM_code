import copy
from math import pi

import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayer, AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from lib.utils.config import CN
from lib.utils import builder
from time import sleep
from lib.utils.transform import rotmat_to_aa, aa_to_rotmat, batch_Rt_transform
from lib.viztools.draw import ColorMode, get_color_map

from lib.utils.etqdm import etqdm
from lib.utils.logger import logger


class GeOptimizer(nn.Module):

    def __init__(
        self,
        cfg: CN,
        cfg_preset: CN,
        lr=0.001,
        n_iter=1000,
        runtime_viz=False,
        viz_contact_mode="CR",
        optimize_hand_tsl=True,
        optimize_hand_pose=True,
        optimize_obj_tsl=True,
        optimize_obj_rot=True,
        use_obj_gt=False,
        use_contact_gt=False,
    ):

        super().__init__()
        from lib.viztools.viz_o3d_utils import VizContext
        self.name = type(self).__name__

        self.cfg = cfg
        self.cfg_preset = cfg_preset
        self.lr = lr
        self.n_iter = n_iter
        self.runtime_viz = runtime_viz
        self.n_region = cfg_preset.N_REGION

        self.honet_module = builder.build_model(cfg.HONET, data_preset=cfg_preset)
        self.picr_module = builder.build_model(cfg.PICR, data_preset=cfg_preset)
        self.honet_module.eval()
        self.picr_module.eval()

        self.criterion = builder.build_loss(cfg.GEO_LOSS, data_preset=cfg_preset)
        self.mano_layer = ManoLayer(rot_mode="axisang",
                                    side="right",
                                    center_idx=cfg_preset.CENTER_IDX,
                                    mano_assets_root="assets/mano_v1_2",
                                    use_pca=False,
                                    flat_hand_mean=True)
        self.axis_layer_fk = AxisLayerFK(mano_assets_root="assets/mano_v1_2")
        self.hand_faces = self.mano_layer.get_mano_closed_faces().numpy()  # (NF, 3)

        self.opt_val = {}  # @NOTE: value to optimize
        self.cnst_val = {}  # @NOTE: constant value
        self.ctrl_val = {  # @NOTE: control value, dict of bool
            "optimize_hand_tsl": optimize_hand_tsl,
            "optimize_hand_pose": optimize_hand_pose,
            "optimize_obj_tsl": optimize_obj_tsl,
            "optimize_obj_rot": optimize_obj_rot,
            "use_obj_gt": use_obj_gt,
            "use_contact_gt": use_contact_gt,
            # ...
            "viz_contact_mode": viz_contact_mode
        }

        self.optimizing = False  # @FLAG: control whether to optimize

        if self.runtime_viz:
            self.viz_ctx = VizContext(non_block=True)
            self.viz_ctx.init(point_size=15.0)

        logger.warning(f"{self.name} initialized. "
                       f"lr={self.lr}, n_iter={self.n_iter}")

    def forward(self, inputs, batch_idx, n_batchs, **kwargs):
        return self.testing_step(inputs, batch_idx, n_batchs, **kwargs)

    def testing_step(self, inputs, batch_idx, n_batchs, **kwargs):
        params = self.prepare_essentials_optim(inputs, batch_idx, **kwargs)

        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=1.0)

        proc_bar = etqdm(range(self.n_iter), desc="GEOI", leave=(batch_idx == n_batchs - 1))

        self.optimizing = True  # @FLAG: start optimizing
        for i, _ in enumerate(proc_bar):
            sleep(0.01)
            # if self.optimizing:
            #     optimizer.zero_grad()

            # loss, loss_dict = self.criterion(self.opt_val,
            #                                  self.cnst_val,
            #                                  curr_step=i,
            #                                  n_steps=self.n_iter,
            #                                  mano_layer=self.mano_layer,
            #                                  **kwargs)
            # if self.optimizing:
            #     loss.backward()
            #     optimizer.step()
            #     scheduler.step()

            # proc_bar.set_description(f"GEO_loss: {loss.item():.5f}")

            if self.runtime_viz:
                eid = 0  # element id in batch to viz, @NOTE: we only viz one in batch !

                hand_verts_curr = self.mano_layer(
                    self.opt_val["hand_pose"],
                    self.cnst_val["hand_shape"]).verts + self.opt_val["hand_tsl"].unsqueeze(1)
                self.viz_ctx.update_by_mesh("hand_mesh_init",
                                            self.cnst_val["hand_verts_init"][eid],
                                            self.hand_faces,
                                            vcolors=[0.9, 0.9, 0.9],
                                            update=False)
                self.viz_ctx.update_by_mesh("hand_mesh_curr",
                                            hand_verts_curr[eid],
                                            self.hand_faces,
                                            vcolors=[0.9, 0, 0],
                                            update=True)

                verts_mask = self.cnst_val["obj_verts_padding"][eid].bool()  # bool is required
                faces_mask = self.cnst_val["obj_faces_padding"][eid].bool()
                obj_verts_init = self.cnst_val["obj_verts_init"][eid][verts_mask]
                obj_normals_init = (self.cnst_val["obj_normals_init"][eid])[verts_mask]
                obj_faces_cnst = (self.cnst_val["obj_faces"][eid])[faces_mask]

                if self.ctrl_val["viz_contact_mode"] == "VC":
                    vc_for_viz = copy.deepcopy(self.cnst_val["vertex_contact"])
                    contact_color = get_color_map(vc_for_viz[eid], ColorMode.VERTEX_CONTACT)
                elif self.ctrl_val["viz_contact_mode"] == "CR":
                    cr_for_viz = copy.deepcopy(self.cnst_val["contact_region_id"])
                    cr_for_viz[self.cnst_val["vertex_contact"] == 0] = self.n_region
                    contact_color = get_color_map(cr_for_viz[eid], ColorMode.CONTACT_REGION)
                elif self.ctrl_val["viz_contact_mode"] == "AE":
                    ae_for_viz = copy.deepcopy(self.cnst_val["anchor_elasti"])
                    ae_for_viz[self.cnst_val["vertex_contact"] == 0] = 0
                    contact_color = get_color_map(ae_for_viz[eid], ColorMode.ANCHOR_ELASTI)

                self.viz_ctx.update_by_mesh("obj_mesh_init",
                                            obj_verts_init,
                                            obj_faces_cnst,
                                            obj_normals_init,
                                            vcolors=contact_color,
                                            update=False)

                if not self.ctrl_val["use_obj_gt"]:
                    rotmat_curr = aa_to_rotmat(self.opt_val["obj_rot"])
                    tsl_curr = self.opt_val["obj_tsl"]
                    obj_verts_can = self.cnst_val["obj_verts_can"]
                    obj_normals_can = self.cnst_val["obj_normals_can"]
                    obj_verts_curr = batch_Rt_transform(obj_verts_can, R=rotmat_curr, t=tsl_curr)  # (B, NO, 3)
                    obj_normals_curr = rotmat_curr.matmul(obj_normals_can.transpose(1, 2)).transpose(1, 2)  # (B, NO, 3)

                    obj_verts_curr = (obj_verts_curr[eid])[verts_mask]
                    obj_normals_curr = (obj_normals_curr[eid])[verts_mask]

                    self.viz_ctx.update_by_mesh("obj_mesh_curr",
                                                obj_verts_curr,
                                                obj_faces_cnst,
                                                obj_normals_curr,
                                                vcolors=contact_color,
                                                update=True)

                self.viz_ctx.step()

        self.optimizing = False  # @FLAG: stop optimizing
        # reset the viz
        self.viz_ctx.remove_all_geometry()

        hand_pose_final = self.opt_val["hand_pose"]
        hand_shape = self.cnst_val["hand_shape"]
        hand_tsl_final = self.opt_val["hand_tsl"]
        hand_verts_final = self.mano_layer(hand_pose_final, hand_shape).verts + hand_tsl_final.unsqueeze(1)
        obj_rot_final = self.opt_val["obj_rot"]
        obj_tsl_final = self.opt_val["obj_tsl"]
        obj_verts_final = batch_Rt_transform(self.cnst_val["obj_verts_can"], R=obj_rot_final, t=obj_tsl_final)
        obj_normals_final = obj_rot_final.matmul(self.cnst_val["obj_normals_can"].transpose(1, 2)).transpose(1, 2)

        res = dict(
            sample_id=inputs["sample_id"],
            # * ===== for hand >>>>>
            hand_shape=self.cnst_val["hand_shape"],
            hand_pose_init=self.cnst_val["hand_pose_init"],
            hand_tsl_init=self.cnst_val["hand_tsl_init"],
            hand_verts_init=self.cnst_val["hand_verts_init"],
            hand_pose_final=hand_pose_final,
            hand_tsl_final=hand_tsl_final,
            hand_verts_final=hand_verts_final,
            # * ===== for obj >>>>>
            obj_rot_init=self.cnst_val["obj_rot_init"],
            obj_tsl_init=self.cnst_val["obj_tsl_init"],
            obj_verts_init=self.cnst_val["obj_verts_init"],
            obj_normals_init=self.cnst_val["obj_normals_init"],
            obj_rot_final=obj_rot_final,
            obj_tsl_final=obj_tsl_final,
            obj_verts_final=obj_verts_final,
            obj_normals_final=obj_normals_final,
            obj_verts_padding_mask=self.cnst_val["obj_verts_padding"],
            obj_faces_padding_mask=self.cnst_val["obj_faces_padding"],
            obj_id=inputs["obj_id"],
        )

        for k, v in res.items():
            res[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v

        return res

    def prepare_essentials_to_optim(self, inputs, step_idx, **kwargs):
        params = []
        HOPE_res = self.honet_module(inputs, step_idx, **kwargs)
        CONT_res = self.picr_module({**inputs, **HOPE_res}, step_idx, **kwargs)

        # region 1. ===== for obj pose >>>>>
        obj_verts_can = inputs["obj_verts_can"]
        obj_normals_can = inputs["obj_normals_can"]
        obj_faces = inputs["obj_faces"]
        obj_verts_padding = inputs["obj_verts_padding_mask"]
        obj_faces_padding = inputs["obj_faces_padding_mask"]
        self.cnst_val["obj_verts_can"] = obj_verts_can.detach().requires_grad_(False)
        self.cnst_val["obj_normals_can"] = obj_normals_can.detach().requires_grad_(False)
        self.cnst_val["obj_verts_padding"] = obj_verts_padding.detach().requires_grad_(False)
        self.cnst_val["obj_faces"] = obj_faces.detach().requires_grad_(False)
        self.cnst_val["obj_faces_padding"] = obj_faces_padding.detach().requires_grad_(False)

        if self.ctrl_val["use_obj_gt"]:
            obj_transf_gt = inputs["target_obj_transf"]
            obj_verts_gt = inputs["target_obj_verts_3d"]
            obj_normals_gt = inputs["target_obj_normals_3d"]
            obj_rot_gt = rotmat_to_aa(obj_transf_gt[:, :3, :3])  # axis-angle
            obj_tsl_gt = obj_transf_gt[:, :3, 3]
            self.cnst_val["obj_rot_init"] = obj_rot_gt.detach().requires_grad_(False)
            self.cnst_val["obj_tsl_init"] = obj_tsl_gt.detach().requires_grad_(False)
            self.cnst_val["obj_verts_init"] = obj_verts_gt.detach().requires_grad_(False)
            self.cnst_val["obj_normals_init"] = obj_normals_gt.detach().requires_grad_(False)

            self.opt_val["obj_rot"] = obj_rot_gt.detach().requires_grad_(False)
            self.opt_val["obj_tsl"] = obj_tsl_gt.detach().requires_grad_(False)
        else:
            # @NOTE: obj_verts = obj_rot @ obj_verts_can + obj_tsl
            obj_rot_pred = rotmat_to_aa(HOPE_res["obj_rotmat"])  # axis-angle
            obj_tsl_pred = HOPE_res["obj_center3d"]
            obj_verts_pred = HOPE_res["recov_obj_verts3d"]
            obj_normals_pred = HOPE_res["obj_rotmat"].bmm(obj_normals_can.transpose(1, 2)).transpose(1, 2)

            self.cnst_val["obj_rot_init"] = obj_rot_pred.detach().requires_grad_(False)
            self.cnst_val["obj_tsl_init"] = obj_tsl_pred.detach().requires_grad_(False)
            self.cnst_val["obj_verts_init"] = obj_verts_pred.detach().requires_grad_(False)
            self.cnst_val["obj_normals_init"] = obj_normals_pred.detach().requires_grad_(False)

            if self.ctrl_val["optimize_obj_tsl"]:
                self.opt_val["obj_tsl"] = obj_tsl_pred.clone().detach().requires_grad_(True)
                params.append({"params": [self.opt_val["obj_tsl"]], "lr": 0.1 * self.lr})
            else:
                self.opt_val["obj_tsl"] = obj_tsl_pred.detach().requires_grad_(False)

            if self.ctrl_val["optimize_obj_rot"]:
                self.opt_val["obj_rot"] = obj_rot_pred.clone().detach().requires_grad_(True)
                params.append({"params": [self.opt_val["obj_rot"]]})
            else:
                self.opt_val["obj_rot"] = obj_rot_pred.detach().requires_grad_(False)
        # endregion <<<<<<

        # region 2. ===== for hand pose >>>>>
        hand_verts_pred = HOPE_res["recov_hand_verts3d"]  # @NOTE: pred in cam space
        hand_pose_pred = HOPE_res["full_pose"]  # (B, 16x3)
        hand_shape = HOPE_res["shape"]  # (B, 10)
        verts_rel = self.mano_layer(hand_pose_pred, hand_shape).verts  # @NOTE: pred in root-rel space
        hand_tsl_pred = torch.mean(hand_verts_pred - verts_rel, dim=1)  # @NOTE: pred transl in cam space
        self.cnst_val["hand_verts_init"] = hand_verts_pred.detach().requires_grad_(False)
        self.cnst_val["hand_tsl_init"] = hand_tsl_pred.detach().requires_grad_(False)
        self.cnst_val["hand_pose_init"] = hand_pose_pred.detach().requires_grad_(False)
        self.cnst_val["hand_shape"] = hand_shape.detach().requires_grad_(False)  # shape is constant

        if self.ctrl_val["optimize_hand_tsl"]:
            self.opt_val["hand_tsl"] = hand_tsl_pred.clone().detach().requires_grad_(True)
            params.append({"params": [self.opt_val["hand_tsl"]], "lr": 0.1 * self.lr})
        else:
            self.opt_val["hand_tsl"] = hand_tsl_pred.detach().requires_grad_(False)

        if self.ctrl_val["optimize_hand_pose"]:
            self.opt_val["hand_pose"] = hand_pose_pred.clone().detach().requires_grad_(True)
            params.append({"params": [self.opt_val["hand_pose"]]})
        else:
            self.opt_val["hand_pose"] = hand_pose_pred.detach().requires_grad_(False)
        # endregion <<<<<<

        # region 3. ===== for contact >>>>>>
        if self.ctrl_val["use_contact_gt"]:
            vertex_contact = inputs["vertex_contact"].unsqueeze(2)  # (B, NO, 1)
            contact_region_id = inputs["contact_region_id"].unsqueeze(2)  # (B, NO, 1)
            contact_region_full = torch.zeros_like(contact_region_id).repeat(1, 1, self.n_region + 1)\
                                                    .scatter(2, contact_region_id, 1.0)  # (B, NO, 18)
            contact_region = contact_region_full[:, :, :self.n_region]  # (B, NO, 17) @NOTE one-hot
            anchor_elasti = inputs["anchor_elasti"]  # (B, NO, 4)
            self.cnst_val["vertex_contact"] = vertex_contact.detach().requires_grad_(False)  # gt
            self.cnst_val["contact_region_id"] = contact_region_id.detach().requires_grad_(False)  # gt
            self.cnst_val["contact_region"] = contact_region.detach().requires_grad_(False)  # gt
            self.cnst_val["anchor_elasti"] = anchor_elasti.detach().requires_grad_(False)  # gt
        else:
            self.cnst_val["vertex_contact"] = CONT_res["recov_vertex_contact"].detach().requires_grad_(False)
            self.cnst_val["contact_region_id"] = CONT_res["recov_contact_region"].detach().requires_grad_(False)
            self.cnst_val["contact_region"] = CONT_res["contact_region"].detach().requires_grad_(False)
            self.cnst_val["anchor_elasti"] = CONT_res["recov_anchor_elasti"].detach().requires_grad_(False)
        # endregion <<<<<<

        return params
