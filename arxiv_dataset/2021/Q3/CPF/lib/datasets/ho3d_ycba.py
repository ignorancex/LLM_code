import hashlib
import json
import os
import pickle
import random
import time
import warnings
from collections import defaultdict
from typing import List

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import trimesh
import yaml
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import mano_to_openpose
from ..utils.contact import process_contact_info
from .ho3d import HO3D, CPF_TRAIN_SEQS, CPF_TEST_SEQS, ho3d_get_seq_object


@DATASET.register_module()
class HO3DYCBA(HO3D):

    def __init__(self, cfg):
        super(HO3DYCBA, self).__init__(cfg)
        assert self.mode_split in ["official"], f"Unknown mode_split: {self.mode_split}"

        del self.mano_layer
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=True,
            ncomps=45,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,  # @NOTE: HO3D's gt require this center_idx=None.
            flat_hand_mean=True,
        )
        # this dataset is rendered using a constant K
        self.CAM_INTR = np.array([
            [617.343, 0.0, 312.42],
            [0.0, 617.343, 241.42],
            [0.0, 0.0, 1.0],
        ]).astype(np.float32)

    def _preload(self):
        self.name = "HO3D_ycba"
        self.root = os.path.join(self.data_root, self.name)
        self.cache_path = os.path.join("common", "cache", self.name, f"{self.name}.pkl")

    def load_dataset(self):
        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annos = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.mode_split} from {self.cache_path}")

        else:
            seq_names = CPF_TRAIN_SEQS + CPF_TEST_SEQS
            obj_names = ho3d_get_seq_object(seq_names)

            annos = {}
            annos["img_path"] = []
            annos["transf"] = []
            annos["hand_verts"] = []
            annos["hand_pose"] = []
            annos["hand_tsl"] = []
            annos["hand_shape"] = []
            annos["obj_id"] = []

            for on in obj_names:
                img_dir = os.path.join(self.root, on, "rgb")
                for sample in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, sample)
                    meta_path = img_path.replace("rgb", "meta").replace("png", "pkl")
                    meta = pickle.load(open(meta_path, "rb"))

                    annos["img_path"].append(img_path)
                    annos["transf"].append(meta["transf"].astype(np.float32))
                    annos["hand_verts"].append(meta["hand_v"].astype(np.float32))
                    annos["hand_pose"].append(meta["hand_p"].astype(np.float32))
                    annos["hand_tsl"].append(meta["hand_t"].astype(np.float32))
                    annos["hand_shape"].append(meta["hand_s"].astype(np.float32))
                    annos["obj_id"].append(on)

            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annos, p_f)
            logger.info(f"Wrote cache for dataset {self.name} to {self.cache_path}")

        self.annos = annos
        if "obj" in self.data_mode:
            self.obj_id_to_dir = self._parse_obj_dir(os.path.join(self.data_root, "YCB_models_supp"))
            # reserving for future use
            self.obj_id_to_mesh = {}
            self.obj_id_to_center_offset = {}
            self.obj_id_to_vertex_color = {}
            self.obj_id_to_vox = {}
            if self.load_obj_on_init:
                for obj_id in self.obj_id_to_dir.keys():
                    self._ho3d_load_and_process_obj(obj_id)

        if "contact" in self.data_mode:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_assets)

        logger.info(f"{self.name} Got {colored(len(self.annos['img_path']), 'yellow', attrs=['bold'])} "
                    f"samples for data_split {self.data_split}")

    def __len__(self):
        return len(self.annos["img_path"])

    def get_sample_identifier(self, idx):
        obj_id = self.annos["obj_id"][idx]
        img_path = self.annos["img_path"][idx]
        res = f"{self.name}_{idx}_{obj_id}_{img_path.split('/')[-1].split('.')[0]}"
        return res

    def get_image_path(self, idx):
        return self.annos["img_path"][idx]

    def get_cam_intr(self, idx):
        return self.CAM_INTR

    def get_joints_3d(self, idx):
        hand_verts = torch.from_numpy(self.annos["hand_verts"][idx]).unsqueeze(0)
        J = self.mano_layer.th_J_regressor
        joints_3d = mano_to_openpose(J, hand_verts).squeeze(0).numpy()
        return joints_3d

    def get_mano_pose(self, idx):
        pose = self.annos["hand_pose"][idx]  # (48)
        pose = torch.from_numpy(pose[None, ...])  # (1, 48)
        new_pose = self.mano_layer.rotation_by_axisang(pose)["full_poses"]  # (1, 16, 3)
        new_pose = new_pose.squeeze(0).reshape(48).numpy()
        return new_pose

    def get_mano_shape(self, idx):
        shape = self.annos["hand_shape"][idx]
        return shape

    def get_verts_3d(self, idx):
        hand_verts = self.annos["hand_verts"][idx]
        return hand_verts

    def get_obj_id(self, idx):
        return self.annos["obj_id"][idx]

    def get_obj_transf(self, idx):
        obj_id = self.annos["obj_id"][idx]
        tranf = self.annos["transf"][idx]
        new_trasnf = tranf @ self.cam_extr  # (4, 4)
        return new_trasnf

    def get_processed_contact_info(self, idx):
        image_path = self.get_image_path(idx)
        contact_path = image_path.replace("rgb", "contact_info")
        contact_path = contact_path.replace("png", "pkl")
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
