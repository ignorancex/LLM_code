import os
import numpy as np
import pickle
import torch
from .ho3d import HO3D
from ..utils.builder import DATASET
from ..utils.transform import (build_SE3_from_Rt, SE3_transform, aa_to_rotmat)
from ..utils.contact import process_contact_info


@DATASET.register_module()
class HO3DSynthT(HO3D):

    def __init__(self, cfg):
        super(HO3DSynthT, self).__init__(cfg)
        assert self.mode_split == "official", f"Unknown mode_split: {self.mode_split}"
        assert self.data_split == "train", "HO3DSynthT is only for training"
        self.synth_round = int(cfg.SYNTH_ROUND)
        assert self.synth_round in [0, 1, 2], f"Unknown synth_round: {self.synth_round}"

        self.CAM_INTR = np.array([
            [617.343, 0.0, 312.42],
            [0.0, 617.343, 241.42],
            [0.0, 0.0, 1.0],
        ]).astype(np.float32)

    def _preload(self):
        super(HO3DSynthT, self)._preload()
        self.synth_root = os.path.join(self.data_root, "HO3D_syntht")

    def get_image_path(self, idx):
        orig_img_path = super(HO3DSynthT, self).get_image_path(idx)
        img_name = os.path.basename(orig_img_path)  # e.g., "0000.png"
        img_dir = os.path.dirname(orig_img_path)
        img_id = img_name.split(".png")[0]  # e.g., "0000"
        syn_img_id = f"{img_id}_{self.synth_round}.png"  # e.g., "0000_{0,1,2}.png"
        syn_img_dir = img_dir.replace(self.root, self.synth_root)  # HO3D -> HO3D_syntht
        syn_img_path = os.path.join(syn_img_dir, syn_img_id)
        return syn_img_path

    def get_mano_pose(self, idx):
        syn_img_path = self.get_image_path(idx)
        syn_pose_path = syn_img_path.replace("rgb", "meta").replace("png", "npy")
        ''' syn_img_path: "HO3D_syntht/0000/rgb/0000_{0,1,2}.png"
            syn_pose_path: "HO3D_syntht/0000/meta/0000_{0,1,2}.npy"
        '''
        pose_48 = np.load(syn_pose_path)  # (48,)
        return pose_48

    def get_cam_intr(self, idx):
        return self.CAM_INTR

    def get_joints_3d(self, idx):
        ori_joints = super(HO3DSynthT, self).get_joints_3d(idx)
        ori_obj_transf = super(HO3DSynthT, self).get_obj_transf(idx)

        # ori_obj_R = ori_obj_transf[:3, :3]
        # ori_obj_t = ori_obj_transf[:3, 3:]  # (3, 1)
        inv_obj_transf = np.linalg.inv(ori_obj_transf)

        # obj_relative_joints = ori_joints - ori_obj_t.T # (21, 3)
        # obj_relative_joints = (ori_obj_R.T @ obj_relative_joints.T).T
        obj_relative_joints = SE3_transform(ori_joints, inv_obj_transf)

        syn_obj_transf = self.get_obj_transf(idx)
        syn_joints = SE3_transform(obj_relative_joints, syn_obj_transf)
        return syn_joints

    def get_verts_3d(self, idx):
        ori_v = super(HO3DSynthT, self).get_verts_3d(idx)
        ori_obj_transf = super(HO3DSynthT, self).get_obj_transf(idx)
        inv_obj_transf = np.linalg.inv(ori_obj_transf)
        obj_relative_v = SE3_transform(ori_v, inv_obj_transf)

        syn_obj_transf = self.get_obj_transf(idx)
        syn_v = SE3_transform(obj_relative_v, syn_obj_transf)
        return syn_v

    def get_obj_transf(self, idx):
        ori_transf = super(HO3DSynthT, self).get_obj_transf(idx)
        rr, rt = self.__get_synth_params(idx)
        inv_extr = np.linalg.inv(self.cam_extr)
        syn_transf = ori_transf @ inv_extr @ build_SE3_from_Rt(aa_to_rotmat(rr), rt) @ self.cam_extr
        return syn_transf

    def __get_synth_params(self, idx):
        syn_img_path = self.get_image_path(idx)
        syn_meta_path = syn_img_path.replace("rgb", "meta").replace("png", "pkl")
        with open(syn_meta_path, "rb") as p_f:
            syn_meta = pickle.load(p_f)

        rr = np.array(syn_meta["rr"], dtype=np.float32)
        rt = np.array(syn_meta["rt"], dtype=np.float32)
        return rr, rt

    def get_processed_contact_info(self, idx):
        ## e.g. img_path:  data/HO3D_syntht/train/ABF10/rgb/0000_0.png
        ##      ci_path:   data/HO3D_supp_v2/train/ABF10/contact_info/0000.pkl"
        img_path = self.get_image_path(idx)
        img_id, suffix = os.path.basename(img_path).split("_")  # e.g., "0000", "{0,1,2,..}.png"

        ci_path = img_path.replace("rgb", "contact_info")
        ci_path = ci_path.replace("HO3D_syntht", "HO3D_supp_v2")
        ci_path = ci_path.replace(os.path.basename(ci_path), f"{img_id}.pkl")

        with open(ci_path, "rb") as bytestream:
            contact_info_list = pickle.load(bytestream)

        (
            vertex_contact,
            contact_region_id,
            anchor_id,
            anchor_dist,
            anchor_elasti,
            anchor_padding_mask,
        ) = process_contact_info(
            contact_info_list,
            self.anchor_mapping,
            pad_vertex=True,
            pad_anchor=True,
        )

        res = {
            "vertex_contact": vertex_contact,
            "contact_region_id": contact_region_id,
            "anchor_id": anchor_id,
            "anchor_dist": anchor_dist,
            "anchor_elasti": anchor_elasti,
            "anchor_padding_mask": anchor_padding_mask,
        }
        return res
