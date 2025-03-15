import hashlib
import json
import os
import pickle
from collections import defaultdict
from typing import List
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import trimesh
import yaml
from dex_ycb_toolkit.dex_ycb import DexYCBDataset, _YCB_CLASSES
from dex_ycb_toolkit.factory import get_dataset
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver
from termcolor import colored
from scipy.spatial.distance import cdist

from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import (
    build_SE3_from_Rt,
    SE3_transform,
    aa_to_rotmat,
    get_annot_center,
    get_annot_scale,
    persp_project,
    rotmat_to_aa,
)
from ..utils.contact import process_contact_info
from .hdata import HODataset, HDataset, auto_delegate_get

YCB_MODEL_DIR = "data/YCB_models_supp_v2"
OBJ_PROCESS_FNAME = "downsample_vcolor_normal.obj"


@DATASET.register_module()
class DexYCB(HODataset):

    def __init__(self, cfg):
        super(HODataset, self).__init__(cfg)
        # ======== DexYCB params >>>>>>>>>>>>>>>>>>
        self.setup = cfg.SETUP  # s0, s1, s2, s3

        # ======== DexYCB default >>>>>>>>>>>>>>>>>\
        self.dex_ycb = None
        self.raw_size = (640, 480)  # (W, H)
        self.load_obj_on_init = cfg.get("LOAD_OBJ_ON_INIT", True)
        self.use_obj_outsrc = cfg.get("USE_OBJ_OUTSRC", False)
        self.inner_mano_layer_right = ManoLayer(flat_hand_mean=False,
                                                side="right",
                                                mano_assets_root="assets/mano_v1_2",
                                                use_pca=True,
                                                ncomps=45)
        self.inner_mano_layer_left = ManoLayer(flat_hand_mean=False,
                                               side="left",
                                               mano_assets_root="assets/mano_v1_2",
                                               use_pca=True,
                                               ncomps=45)

        self.load_dataset()

    def _preload(self):
        self.name = "DexYCB"
        self.root = os.path.join(self.data_root, self.name)
        os.environ["DEX_YCB_DIR"] = self.root

    def load_dataset(self):
        self._preload()

        dexycb_name = f"{self.setup}_{self.data_split}"
        logger.info(f"DexYCB use split: {dexycb_name}")
        self.dex_ycb: DexYCBDataset = get_dataset(dexycb_name)

        if "obj" in self.data_mode:
            self.obj_id_to_path = {}
            __obj_file = self.dex_ycb.obj_file
            for k, v in __obj_file.items():
                obj_id = _YCB_CLASSES[k]
                if self.use_obj_outsrc is True:
                    obj_path = os.path.join(YCB_MODEL_DIR, f"{obj_id}", OBJ_PROCESS_FNAME)
                    # if file not exists
                    if not os.path.exists(obj_path):
                        obj_path = v
                else:
                    obj_path = v  # use the obj from DexYCB
                self.obj_id_to_path[obj_id] = obj_path

            # reserving for future use
            self.obj_id_to_mesh = {}
            self.obj_id_to_center_offset = {}
            self.obj_id_to_vox = {}
            if self.load_obj_on_init:
                logger.info("Loading objects...")
                for obj_id in etqdm(self.obj_id_to_path.keys()):
                    self.__dexycb_load_and_process_obj(obj_id)
                logger.info("Loading objects done.")

        if "contact" in self.data_mode:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_assets)

    def __dexycb_load_and_process_obj(self, obj_id):
        obj_path = self.obj_id_to_path[obj_id]
        obj_mesh = trimesh.load(obj_path, process=False)

        verts_0 = np.asarray(obj_mesh.vertices).astype(np.float32)
        offset = (verts_0.min(0) + verts_0.max(0)) / 2.0
        verts_1 = verts_0 - offset
        obj_mesh.vertices = verts_1  # @NOTE: assign back to obj_mesh

        self.obj_id_to_mesh[obj_id] = obj_mesh
        self.obj_id_to_center_offset[obj_id] = offset

    def __len__(self):
        return len(self.dex_ycb)

    def __get_label(self, label_file: str):
        return np.load(label_file)

    def get_image(self, idx):
        path = self.get_image_path(idx)
        img = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return img

    def get_image_path(self, idx):
        sample = self.dex_ycb[idx]
        return sample["color_file"]

    def get_joints_3d(self, idx):
        sample = self.dex_ycb[idx]
        label = self.__get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_3d"].squeeze(0)

    def get_verts_3d(self, idx):
        sample = self.dex_ycb[idx]
        label = self.__get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.inner_mano_layer_right if sample["mano_side"] == "right" else self.inner_mano_layer_left
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts + pose_m[:, 48:]
        return hand_verts.squeeze(0).numpy().astype(np.float32)

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_joints_2d(self, idx):
        sample = self.dex_ycb[idx]
        label = self.__get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_2d"].squeeze(0)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_cam_intr(self, idx):
        sample = self.dex_ycb[idx]
        fx = sample["intrinsics"]["fx"]
        fy = sample["intrinsics"]["fy"]
        ppx = sample["intrinsics"]["ppx"]
        ppy = sample["intrinsics"]["ppy"]
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float32)
        return K

    def get_side(self, idx):
        sample = self.dex_ycb[idx]
        return sample["mano_side"]

    def get_mano_pose(self, idx):
        sample = self.dex_ycb[idx]
        label = self.__get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        mano_layer = self.inner_mano_layer_right if sample["mano_side"] == "right" else self.inner_mano_layer_left
        pose_m = torch.from_numpy(label["pose_m"])
        pose = mano_layer.rotation_by_axisang(pose_m[:, :48])["full_poses"]  # (1, 48)
        pose = pose.squeeze(0).numpy().astype(np.float32)
        return pose

    def get_mano_shape(self, idx):
        sample = self.dex_ycb[idx]
        shape = sample["mano_betas"]
        shape = np.array(shape, dtype=np.float32)
        return shape

    def get_bbox_center_scale(self, idx):
        if self.use_full_image:
            center = np.array([self.raw_size[0] // 2, self.raw_size[1] // 2], dtype=np.float32)
            scale = self.raw_size[0]
            return center, scale

        joints2d = self.get_joints_2d(idx)  # (21, 2)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)
        return center, scale

    def get_sample_identifier(self, idx):
        res = f"{self.name}_{self.data_split}_{self.setup}__{idx:07d}"
        return res

    def get_obj_id(self, idx):
        sample = self.dex_ycb[idx]
        ycb_id = sample["ycb_ids"][sample["ycb_grasp_ind"]]
        obj_id = _YCB_CLASSES[ycb_id]
        return obj_id

    def get_obj_verts_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self.__dexycb_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertices).astype(np.float32)

    def get_obj_verts_color(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self.__dexycb_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]

        vertex_color = np.asarray(obj_mesh.visual.vertex_colors).astype(np.float32)
        vertex_color = vertex_color[:, :3] / 255.0
        return vertex_color

    def get_obj_corners_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self.__dexycb_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        corners = trimesh.bounds.corners(obj_mesh.bounds)
        return np.asarray(corners).astype(np.float32)

    def get_obj_faces(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self.__dexycb_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.faces).astype(np.longlong)

    def get_obj_transf(self, idx):
        sample = self.dex_ycb[idx]
        label = self.__get_label(sample["label_file"])
        transf = label["pose_y"][sample["ycb_grasp_ind"]]
        obj_id = self.get_obj_id(idx)
        v_0 = self.obj_id_to_center_offset[obj_id]

        R, t = transf[:3, :3], transf[:, 3]
        new_t = R @ v_0 + t
        new_transf = build_SE3_from_Rt(R, new_t)
        return new_transf.astype(np.float32)

    def get_obj_verts_3d(self, idx):
        verts_c = self.get_obj_verts_can(idx)
        obj_transf = self.get_obj_transf(idx)
        verts_3d = SE3_transform(verts_c, obj_transf)
        return verts_3d

    def get_obj_corners_3d(self, idx):
        cornes_c = self.get_obj_corners_can(idx)
        obj_transf = self.get_obj_transf(idx)
        corners_3d = SE3_transform(cornes_c, obj_transf)
        return corners_3d

    def get_obj_normals_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self.__dexycb_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertex_normals).astype(np.float32)

    def get_obj_normals_3d(self, idx):
        normals_c = self.get_obj_normals_can(idx)
        obj_transf = self.get_obj_transf(idx)
        R = obj_transf[:3, :3]

        normals_3d = (R @ normals_c.T).T
        return normals_3d

    def get_processed_contact_info(self, idx):
        root_contact_info = self.root.replace(self.name, f"{self.name}_supp")
        image_path = self.get_image_path(idx)
        contact_path = image_path.replace(self.root, root_contact_info)
        contact_path = contact_path.replace("color", "contact_info")
        contact_path = contact_path.replace("jpg", "pkl")
        with open(contact_path, "rb") as bytestream:
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


@auto_delegate_get(incls=DexYCB)
@DATASET.register_module()
class FilteredDexYCB(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.name = type(self).__name__
        self.base_dataset = DexYCB(cfg)
        # selected obj id:
        self.selected_obj_ids = cfg.get("SELECTED_OBJ_IDS", None)
        all_obj_ids = set([v for k, v in _YCB_CLASSES.items()])
        if self.selected_obj_ids is not None:
            self.selected_obj_ids = set(self.selected_obj_ids)
        else:
            self.selected_obj_ids = all_obj_ids

        self.use_cache = bool(cfg.DATA_PRESET.USE_CACHE)
        self.filter_no_contact = True
        self.contact_threshold = 0.005
        self.filter_left = True
        self.filter_invisibile = True
        self.filter_object = (self.selected_obj_ids != all_obj_ids)

        do_filter = self.filter_no_contact or \
                    self.filter_left or \
                    self.filter_invisibile or \
                    self.filter_object

        self.inlier_indices = None
        if do_filter:
            self.inlier_indices = self._get_inlier_indices()

        logger.info(f"{self.name} Got {colored(len(self), 'yellow', attrs=['bold'])}"
                    f"/{len(self.base_dataset)} samples for data_split {self.base_dataset.data_split}")

    def __getitem__(self, idx):
        if self.inlier_indices is not None:
            # Mode: contact - return filtered subset element
            return self.base_dataset[self.inlier_indices[idx]]
        else:
            # Mode: full - return fullset element
            return self.base_dataset[idx]

    def __len__(self):
        if self.inlier_indices is not None:
            # Mode: contact - return filtered subset length
            return len(self.inlier_indices)
        else:
            # Mode: full - return fullset length
            return len(self.base_dataset)

    def _get_inlier_indices(self):
        self.indices_id_dict = {
            "data_split": self.base_dataset.data_split,
            "setup": self.base_dataset.setup,
            "filter_no_contact": self.filter_no_contact,
            "filter_left": self.filter_left,
            "filter_invisibile": self.filter_invisibile,
        }
        if self.filter_object:
            self.indices_id_dict["filter_object"] = True
            self.indices_id_dict["selected_obj_ids"] = sorted(list(self.selected_obj_ids))

        indices_id = json.dumps(self.indices_id_dict, sort_keys=True)
        indices_id = hashlib.md5(indices_id.encode("ascii")).hexdigest()[:8]
        indices_cache_path = os.path.join("common", "cache", self.base_dataset.name, self.base_dataset.setup,
                                          self.base_dataset.data_split, f"{indices_id}.pkl")

        os.makedirs(os.path.dirname(indices_cache_path), exist_ok=True)
        if os.path.exists(indices_cache_path) and self.use_cache:
            with open(indices_cache_path, "rb") as p_f:
                inlier_indices = pickle.load(p_f)
            logger.info(f"Loaded inlier indices for dataset {self.base_dataset.name} from {indices_cache_path}")
        else:
            inlier_indices = []
            for idx in etqdm(range(len(self.base_dataset)), desc="Filtering DexYCB"):
                side = self.base_dataset.get_side(idx)
                if self.filter_left and side == "left":
                    continue

                if self.filter_object:
                    obj_id = self.base_dataset.get_obj_id(idx)
                    if obj_id not in self.selected_obj_ids:
                        continue

                joints_2d = self.base_dataset.get_joints_2d(idx)
                if self.filter_invisibile and np.any(joints_2d == -1):
                    continue

                if self.filter_no_contact:
                    verts_o = self.base_dataset.get_obj_verts_3d(idx)
                    joints_h = self.base_dataset.get_joints_3d(idx)
                    all_dists = cdist(joints_h, verts_o)
                    if np.min(all_dists) > self.contact_threshold:
                        continue

                inlier_indices.append(idx)
            # dump cache
            with open(indices_cache_path, "wb") as f:
                pickle.dump(inlier_indices, f)
            with open(indices_cache_path.replace(".pkl", ".json"), "w") as f:
                json.dump(self.indices_id_dict, f)
            logger.info(f"Wrote filter indices for dataset {self.base_dataset.name} to {indices_cache_path}")
        return inlier_indices
