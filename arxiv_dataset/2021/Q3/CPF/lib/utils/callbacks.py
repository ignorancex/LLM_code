import os
import pickle

import cv2
import numpy as np
import torch
import math
import imageio
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver, get_rev_anchor_mapping

from lib.utils.logger import logger
from lib.utils.builder import CALLBACK
from lib.utils.transform import denormalize
from lib.viztools.opendr_renderer import OpenDRRenderer
from lib.viztools.utils import get_color_map, ColorMode, ColorsMap


@CALLBACK.register_module()
class IdelCallback():

    def __init__(self):
        pass

    def __call__(self, preds, inputs, step_idx, **kwargs):
        pass

    def on_finished(self):
        pass

    def reset(self):
        pass


@CALLBACK.register_module()
class DrawCallback(IdelCallback):

    def __init__(self, cfg):

        self.img_draw_dir = cfg.IMG_DRAW_DIR
        self.draw_contact = cfg.DRAW_CONTACT
        os.makedirs(self.img_draw_dir, exist_ok=True)

        mano_layer = ManoLayer(mano_assets_root="assets/mano_v1_2")
        self.mano_faces = mano_layer.get_mano_closed_faces().numpy()
        self.renderer = OpenDRRenderer()

    def __call__(self, preds, inputs, step_idx, **kwargs):
        tensor_image = inputs["image"]  # (B, 3, H, W) 5 channels
        batch_size = tensor_image.size(0)
        image = denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False)
        image = image.permute(0, 2, 3, 1)
        image = image.mul_(255.0).detach().cpu()  # (B, H, W, 3)
        image = image.numpy().astype(np.uint8)

        for i in range(batch_size):
            curr_image = image[i]
            hand_verts = preds["recov_hand_verts3d"][i].detach().cpu().numpy()  # (B, 778, 3)
            obj_verts = preds["recov_obj_verts3d"][i].detach().cpu().numpy()  # (B, NO, 3)

            cam_intr = inputs["target_cam_intr"][i].detach().cpu().numpy()  # (B, 3, 3)
            obj_faces = inputs["obj_faces"][i].detach().cpu().numpy()  # (B, NF, 3)

            h_color = np.array(ColorsMap['lightblue'])
            o_color = inputs["obj_verts_color"][i].detach().cpu().numpy() if "obj_verts_color" in inputs \
                                                                        else np.array(ColorsMap['lime'])

            frame_h = self.renderer([hand_verts], [self.mano_faces],
                                    cam_intr,
                                    img=curr_image.copy(),
                                    vertex_color=[h_color])
            frame_o = self.renderer([obj_verts], [obj_faces], cam_intr, img=curr_image.copy(), vertex_color=[o_color])

            frame_list = [curr_image, frame_h, frame_o]
            if self.draw_contact:
                vertex_contact = preds["recov_vertex_contact"][i].detach().cpu().numpy()
                contact_region_id = preds["recov_contact_region"][i].detach().cpu().numpy()
                contact_region_id[vertex_contact == 0] = 17
                anchor_elasti = preds["recov_anchor_elasti"][i].detach().cpu().numpy()
                anchor_elasti = np.max(anchor_elasti, axis=1)  # find the maximum
                vc_color = get_color_map(vertex_contact, ColorMode.VERTEX_CONTACT)
                cr_color = get_color_map(contact_region_id, ColorMode.CONTACT_REGION)
                ae_color = get_color_map(anchor_elasti, ColorMode.ANCHOR_ELASTI)
                frame_vc = self.renderer([obj_verts], [obj_faces],
                                         cam_intr,
                                         img=curr_image.copy(),
                                         vertex_color=[vc_color])
                frame_cr = self.renderer([obj_verts], [obj_faces],
                                         cam_intr,
                                         img=curr_image.copy(),
                                         vertex_color=[cr_color])
                frame_ae = self.renderer([obj_verts], [obj_faces],
                                         cam_intr,
                                         img=curr_image.copy(),
                                         vertex_color=[ae_color])
                frame_list.extend([frame_vc, frame_cr, frame_ae])

            frame = np.hstack(frame_list)
            img_save_path = inputs["sample_id"][i] + ".png"
            imageio.imwrite(os.path.join(self.img_draw_dir, img_save_path), frame)

        return

    def on_finished(self):
        pass


@CALLBACK.register_module()
class DumperCallback(IdelCallback):

    def __init__(self, cfg):
        self.type = "PicrDumper"
        self.dump_prefix = cfg.DUMP_PREFIX
        _, _, _, anchor_mapping = anchor_load_driver(os.path.normpath("assets"))
        self.rev_anchor_mapping = get_rev_anchor_mapping(anchor_mapping)
        self.counter = 0

        self.honet_field = [
            "image_path"
            "hand_tsl",
            "hand_joints_3d",
            "hand_verts_3d",
            "hand_full_pose",
            "hand_shape",
            "obj_tsl",
            "obj_rot",
            "obj_verts_3d",
        ]

    def info(self):
        res = f"{self.type}\n"
        res += f"  prefix: {self.dump_prefix}\n"
        res += f"  count: {self.counter}"
        return res

    def on_finished(self):
        print(self.info())
        self.reset()

    def reset(self):
        self.counter = 0

    def __call__(self, preds, inputs, step_idx, **kwargs):

        assert "recov_vertex_contact" in preds, f"{self.type}: vertex_contact not found"
        assert "recov_contact_region" in preds, f"{self.type}: contact_region not found"
        assert "recov_anchor_elasti" in preds, f"{self.type}: anchor_elasti not found"

        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                preds[k] = v.detach().cpu()

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.detach().cpu()

        batch_size = inputs["image"].size(0)
        for idx in range(batch_size):
            sample_id = inputs["sample_id"][idx]

            # ==================== dump contact related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            verts_collate_mask = inputs["obj_verts_padding_mask"][idx, ...].bool()  # (NO,)
            prd_vertex_contact = preds["recov_vertex_contact"][idx, ...]  # (NO,)
            contact_in_image_mask = preds["contact_in_image_mask"][idx, ...]  # (NO,)
            prd_vertex_contact = prd_vertex_contact.bool() & contact_in_image_mask.bool()
            prd_vertex_contact = prd_vertex_contact[verts_collate_mask]  # TENSOR[X, ]

            prd_contact_region = preds["recov_contact_region"][idx, ...]  # (NO,)
            prd_contact_region = prd_contact_region[verts_collate_mask]  # TENSOR[X, ]

            prd_anchor_elasti = preds["recov_anchor_elasti"][idx, ...]  # (NO, 4)
            prd_anchor_elasti = prd_anchor_elasti[verts_collate_mask, ...]  # TENSOR[X, 4]

            # iterate over all points
            sample_res = []
            n_points = prd_vertex_contact.shape[0]  # X
            for p_idx in range(n_points):
                p_contact = int(prd_vertex_contact[p_idx])
                if p_contact == 0:
                    p_res = {
                        "contact": 0,
                    }
                else:  # p_contact == 1
                    p_region = int(prd_contact_region[p_idx])
                    p_anchor_id = self.rev_anchor_mapping[p_region]
                    p_n_anchor = len(p_anchor_id)
                    p_anchor_elasti = prd_anchor_elasti[p_idx, :p_n_anchor].tolist()
                    p_res = {
                        "contact": 1,
                        "region": p_region,
                        "anchor_id": p_anchor_id,
                        "anchor_elasti": p_anchor_elasti,
                    }
                sample_res.append(p_res)

            # save picr's sample_res
            contact_save_path = os.path.join(self.dump_prefix, f"{sample_id}_contact.pkl")
            save_dir = os.path.dirname(contact_save_path)
            os.makedirs(save_dir, exist_ok=True)
            with open(contact_save_path, "wb") as fstream:
                pickle.dump(sample_res, fstream)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # ==================== dump mano and object related info >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            honet_res = {}
            honet_res["hand_tsl"] = preds["hand_center3d"][idx, ...]
            honet_res["hand_joints_3d"] = preds["recov_joints3d"][idx, ...]
            honet_res["hand_verts_3d"] = preds["recov_hand_verts3d"][idx, ...]
            honet_res["hand_full_pose"] = preds["full_pose"][idx, ...]
            honet_res["hand_shape"] = preds["shape"][idx, ...]
            honet_res["obj_tsl"] = preds["obj_center3d"][idx, ...]
            honet_res["obj_rotmat"] = preds["obj_rotmat"][idx, ...]  # (3, 3) rotation matrix
            honet_res["obj_rot"] = preds["obj_rotaa"][idx, ...]  # (3,) axis-angle
            honet_res["obj_verts_3d"] = preds["recov_obj_verts3d"][idx, ...][verts_collate_mask, :]
            honet_res["image_path"] = inputs["image_path"][idx]

            # save honet_res
            honet_save_path = os.path.join(self.dump_prefix, f"{sample_id}_honet.pkl")
            with open(honet_save_path, "wb") as fstream:
                pickle.dump(honet_res, fstream)

            self.counter += 1
