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
from .hdata import HODataset

OBJ_PROCESS_FNAME = "downsample_vcolor_normal.obj"
OBJ_VOX_FNAME = "solid.binvox"

HO3D_SEQ_TO_OBJ = {
    "ABF10": "021_bleach_cleanser",
    "ABF11": "021_bleach_cleanser",
    "ABF12": "021_bleach_cleanser",
    "ABF13": "021_bleach_cleanser",
    "ABF14": "021_bleach_cleanser",
    "BB10": "011_banana",
    "BB11": "011_banana",
    "BB12": "011_banana",
    "BB13": "011_banana",
    "BB14": "011_banana",
    "GPMF10": "010_potted_meat_can",
    "GPMF11": "010_potted_meat_can",
    "GPMF12": "010_potted_meat_can",
    "GPMF13": "010_potted_meat_can",
    "GPMF14": "010_potted_meat_can",
    "GSF10": "037_scissors",
    "GSF11": "037_scissors",
    "GSF12": "037_scissors",
    "GSF13": "037_scissors",
    "GSF14": "037_scissors",
    "MC1": "003_cracker_box",
    "MC2": "003_cracker_box",
    "MC4": "003_cracker_box",
    "MC5": "003_cracker_box",
    "MC6": "003_cracker_box",
    "MDF10": "035_power_drill",
    "MDF11": "035_power_drill",
    "MDF12": "035_power_drill",
    "MDF13": "035_power_drill",
    "MDF14": "035_power_drill",
    "ND2": "035_power_drill",
    "SB10": "021_bleach_cleanser",
    "SB12": "021_bleach_cleanser",
    "SB14": "021_bleach_cleanser",
    "SM2": "006_mustard_bottle",
    "SM3": "006_mustard_bottle",
    "SM4": "006_mustard_bottle",
    "SM5": "006_mustard_bottle",
    "SMu1": "025_mug",
    "SMu40": "025_mug",
    "SMu41": "025_mug",
    "SMu42": "025_mug",
    "SS1": "004_sugar_box",
    "SS2": "004_sugar_box",
    "SS3": "004_sugar_box",
    "ShSu10": "004_sugar_box",
    "ShSu12": "004_sugar_box",
    "ShSu13": "004_sugar_box",
    "ShSu14": "004_sugar_box",
    "SiBF10": "011_banana",
    "SiBF11": "011_banana",
    "SiBF12": "011_banana",
    "SiBF13": "011_banana",
    "SiBF14": "011_banana",
    "SiS1": "004_sugar_box",
    # test
    "SM1": "006_mustard_bottle",
    "MPM10": "010_potted_meat_can",
    "MPM11": "010_potted_meat_can",
    "MPM12": "010_potted_meat_can",
    "MPM13": "010_potted_meat_can",
    "MPM14": "010_potted_meat_can",
    "SB11": "021_bleach_cleanser",
    "SB13": "021_bleach_cleanser",
    "AP10": "019_pitcher_base",
    "AP11": "019_pitcher_base",
    "AP12": "019_pitcher_base",
    "AP13": "019_pitcher_base",
    "AP14": "019_pitcher_base",
}

CPF_TRAIN_SEQS = [
    "ABF10",
    "ABF11",
    "ABF12",
    "ABF13",
    "ABF14",
    "GPMF10",
    "GPMF11",
    "GPMF12",
    "GPMF13",
    "GPMF14",
    "SB10",
    "SB12",
    "SB14",
    "SM2",
    "SM3",
    "SM4",
    "SM5",
]

CPF_TEST_SEQS = [
    "SM1",
    "MPM10",
    "MPM11",
    "MPM12",
    "MPM13",
    "MPM14",
    "SB11",
    "SB13",
]

CPF_GRASP_LIST = {
    "SM1": [i for i in range(0, 889 + 1)],
    "SM2": [i for i in range(0, 897 + 1)],
    "SM3": [i for i in range(0, 895 + 1)],
    "SM4": [i for i in range(0, 879 + 1)],
    "SM5": [i for i in range(0, 867 + 1)],
    "MPM10": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
    "MPM11": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
    "MPM12": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
    "MPM13": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
    "MPM14": [i for i in range(30, 450 + 1)] + [i for i in range(585, 685 + 1)],
    "SB10": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    "SB11": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    "SB12": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    "SB13": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    "SB14": [i for i in range(340, 1355 + 1)] + [i for i in range(1415, 1686 + 1)],
    "GPMF10": [i for i in range(60, 400 + 1)] + [i for i in range(460, 877 + 1)],
    "GPMF11": [i for i in range(60, 400 + 1)] + [i for i in range(460, 877 + 1)],
    "GPMF12": [i for i in range(60, 400 + 1)] + [i for i in range(460, 877 + 1)],
    "GPMF13": [i for i in range(60, 400 + 1)] + [i for i in range(460, 877 + 1)],
    "GPMF14": [i for i in range(60, 400 + 1)] + [i for i in range(460, 877 + 1)],
}


def ho3d_get_seq_object(seq):
    obj_set = set()
    for s in seq:
        obj_set.add(HO3D_SEQ_TO_OBJ[s])
    return list(obj_set)


@DATASET.register_module()
class HO3D(HODataset):

    def __init__(self, cfg):
        super(HODataset, self).__init__(cfg)
        # ======== HO3D params >>>>>>>>>>>>>>>>>>
        self.mode_split = cfg.MODE_SPLIT
        assert self.mode_split in ["official", "ho3d_paper"], f"Unknown mode_split: {self.mode_split}"
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # ======== HO3D default >>>>>>>>>>>>>>>>>
        self.version = cfg.VERSION
        self.raw_size = (640, 480)  # (W, H)
        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        # this camera extrinsic has no translation
        # and this is the reason transforms in following code just use rotation part
        self.cam_extr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,  # @NOTE: HO3D's gt require this center_idx=None.
            flat_hand_mean=True,
        )
        self.load_obj_on_init = cfg.get("LOAD_OBJ_ON_INIT", False)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.load_dataset()

    def _preload(self):
        # deal with all the naming and path convention
        self.name = "HO3D"
        if self.version == "V2":
            self.root = os.path.join(self.data_root, self.name)
        elif self.version == "V3":
            self.root = os.path.join(self.data_root, f"{self.name}_v3")
        self.root_assets = os.path.normpath("assets")

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "mode_split": self.mode_split,
            "version": self.version,
            "data_mode": self.data_mode,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()[:8]
        self.cache_path = os.path.join("common", "cache", self.name, self.version, self.data_split,
                                       f"{self.cache_identifier}.pkl")

    def load_dataset(self):
        self._preload()

        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        if self.mode_split == "ho3d_paper":
            """
            @NOTE：original HO3D official split (V2 and V3). For details:
            * V2: https://competitions.codalab.org/competitions/22485
            * V3: https://competitions.codalab.org/competitions/33267
            """
            seq_frames, subfolder = self.__load_seq_frames()
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        elif self.mode_split == "official":
            # paper CPF splits, requires HO pair with minimumn distance for contact.
            seq_frames, subfolder = self.__load_cpf_frames(self.data_split, self.root)
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        else:
            raise NotImplementedError()

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.version}_{self.data_split}_{self.mode_split} "
                        f"from {self.cache_path}")

        else:
            annot_mapping, seq_idx = self._load_annots(seq_frames=seq_frames, subfolder=subfolder)
            annotations = {"seq_idx": seq_idx, "annot_mapping": annot_mapping}
            with open(self.cache_path, "wb") as p_f:
                pickle.dump(annotations, p_f)
            logger.info(f"Wrote cache for {self.name}_{self.version}_{self.data_split}_{self.mode_split} "
                        f"to {self.cache_path}")

        self.seq_idx = annotations["seq_idx"]
        self.annot_mapping = annotations["annot_mapping"]
        self.sample_idxs = list(range(len(self.seq_idx)))

        if "obj" in self.data_mode:
            self.obj_id_to_dir = self._parse_obj_dir(os.path.join(self.data_root, "YCB_models_supp_v2"))
            # reserving for future use
            self.obj_id_to_mesh = {}
            self.obj_id_to_center_offset = {}
            self.obj_id_to_vertex_color = {}
            self.obj_id_to_vox = {}
            if self.load_obj_on_init:
                for obj_id in self.obj_id_to_dir.keys():
                    self._ho3d_load_and_process_obj(obj_id)

        if "contact" in self.data_mode:
            self.contact_infos = self._load_contact_infos(self.seq_idx, self.annot_mapping)
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_assets)

        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{len(self.seq_idx)} samples for data_split {self.data_split}")

    def _load_contact_infos(self, seq_idx, annot_mapping):
        contact_info = []
        root_contact_info = self.root.replace(self.name, f"{self.name}_supp_v2")
        for i in range(len(seq_idx)):
            seq, idx = seq_idx[i]
            path = annot_mapping[seq][idx]["img"]
            path = path.replace(self.root, root_contact_info)
            path = path.replace("rgb", "contact_info")
            path = path.replace("png", "pkl")
            contact_info.append(path)
        return contact_info

    def _ho3d_load_and_process_obj(self, obj_id):
        obj_dir = self.obj_id_to_dir[obj_id]
        obj_mesh = trimesh.load(os.path.join(obj_dir, OBJ_PROCESS_FNAME), process=False)
        verts_0 = np.asarray(obj_mesh.vertices).astype(np.float32)
        offset = (verts_0.min(0) + verts_0.max(0)) / 2.0
        verts_1 = verts_0 - offset
        obj_mesh.vertices = verts_1  # @NOTE: assign back to obj_mesh

        self.obj_id_to_mesh[obj_id] = obj_mesh
        self.obj_id_to_center_offset[obj_id] = offset

    def _ho3d_load_obj_voxel(self, obj_id):
        obj_dir = self.obj_id_to_dir[obj_id]
        obj_path = os.path.join(obj_dir, OBJ_VOX_FNAME)
        vox = trimesh.load(obj_path)

        voxpt = np.asarray(vox.points).astype(np.float32)
        offset = (voxpt.min(0) + voxpt.max(0)) / 2.0
        voxpt = voxpt - offset

        self.obj_id_to_vox[obj_id] = {
            "points": np.array(vox.points),
            "matrix": np.array(vox.matrix),
            "element_volume": vox.element_volume,
        }
        return vox

    def _parse_obj_dir(self, obj_root_dir):
        obj_ids = [oid for oid in os.listdir(obj_root_dir) if ".tgz" not in oid]
        obj_id_to_dir = {}
        for oid in obj_ids:
            obj_dir = os.path.join(obj_root_dir, oid)
            obj_id_to_dir[oid] = obj_dir
        return obj_id_to_dir

    def __load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        """
        trainval_idx (int): How many frames to include in training split when
                using trainval/val/test split
        """
        if self.mode_split == "ho3d_paper":
            if self.data_split == "train":
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                raise ValueError(f"Unknown data split: {self.data_split} in mode [paper]")
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
        else:
            raise ValueError(f"This func only used in mode: [ho3d_paper] , got mode: {self.mode_split}")
        return seq_frames, subfolder

    def __load_cpf_frames(self, split, root, trainval_idx=60000):
        if split in ["train", "trainval", "val"]:
            info_path = os.path.join(root, "train.txt")
            subfolder = "train"
        elif split == "test":
            info_path = os.path.join(root, "evaluation.txt")
            subfolder = "evaluation"
        else:
            raise ValueError(f"Unknown split {split}")

        with open(info_path, "r") as f:
            lines = f.readlines()
        txt_seq_frames = [line.strip().split("/") for line in lines]
        if split == "trainval":
            txt_seq_frames = txt_seq_frames[:trainval_idx]
        elif split == "val":
            txt_seq_frames = txt_seq_frames[trainval_idx:]
        seqs = {}
        for sf in txt_seq_frames:
            if sf[0] not in CPF_TRAIN_SEQS and sf[0] not in CPF_TEST_SEQS:
                continue
            if sf[0] in CPF_GRASP_LIST and not (int(sf[1]) in CPF_GRASP_LIST[sf[0]]):
                continue
            if sf[0] in seqs:
                seqs[sf[0]].append(sf[1])
            else:
                seqs[sf[0]] = [sf[1]]
        seq_frames = []
        for s in seqs:
            seqs[s].sort()
            for f in range(len(seqs[s])):
                seq_frames.append([s, seqs[s][f]])
        return seq_frames, subfolder

    def _load_annots(self, seq_frames=[], subfolder="train", **kwargs):
        seq_idx = []
        annot_mapping = defaultdict(list)
        seq_counts = defaultdict(int)
        for idx_count, (seq, frame_idx) in enumerate(etqdm(seq_frames)):
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")

            with open(meta_path, "rb") as p_f:
                annot = pickle.load(p_f)

            fext = "png" if self.version == "V2" else "jpg"
            img_path = os.path.join(rgb_folder, f"{frame_idx}.{fext}")
            annot["img"] = img_path
            annot["frame_idx"] = frame_idx

            annot_mapping[seq].append(annot)
            seq_idx.append((seq, seq_counts[seq]))
            seq_counts[seq] += 1

        return annot_mapping, seq_idx

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idxs(self) -> List[int]:
        return self.sample_idxs

    def get_sample_identifier(self, idx):
        res = f"{self.name}_{self.version}_{self.data_split}_{self.mode_split}"
        res = res + f"__{idx:07d}"
        return res

    def get_annot(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        return annot

    def get_seq_frame(self, idx):
        seq, img_idx = self.seq_idx[idx]
        annot = self.annot_mapping[seq][img_idx]
        frame_idx = annot["frame_idx"]
        return seq, frame_idx

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = np.array(imageio.imread(img_path, pilmode="RGB"), dtype=np.uint8)
        return img

    def get_image_path(self, idx):
        img_path = self.get_annot(idx)["img"]
        return img_path

    def get_side(self, idx):
        return "right"

    def get_cam_intr(self, idx):
        cam_intr = self.get_annot(idx)["camMat"]
        return cam_intr.astype(np.float32)

    def get_joints_3d(self, idx):
        hand_info = self.__ho3d_get_hand_info(idx)
        joints_3d = hand_info[0]
        joints_3d = self.cam_extr[:3, :3].dot(joints_3d.transpose()).transpose()
        joints_3d = joints_3d[self.reorder_idxs]
        return joints_3d.astype(np.float32)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_joints_2d(self, idx):
        joints_3d = self.get_joints_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(joints_3d, cam_intr)

    def get_mano_pose(self, idx):
        hand_info = self.__ho3d_get_hand_info(idx)
        pose = hand_info[1]
        root, remains = pose[:3], pose[3:]
        root = rotmat_to_aa(self.cam_extr[:3, :3] @ aa_to_rotmat(root))
        transf_pose = np.concatenate((root, remains), axis=0)
        return transf_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        hand_info = self.__ho3d_get_hand_info(idx)
        shape = hand_info[2]
        shape = np.array(shape, dtype=np.float32)
        return shape

    def get_verts_3d(self, idx):
        _, pose, shape, tsl = self.__ho3d_get_hand_info(idx)
        mano_out = self.mano_layer(torch.from_numpy(pose[None]), torch.from_numpy(shape[None]))
        verts = mano_out.verts[0].numpy() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        return transf_verts.astype(np.float32)

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_2d(self, idx):
        verts_3d = self.get_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(verts_3d, cam_intr)

    def get_hand_faces(self, idx):
        faces = np.array(self.mano_layer.get_mano_closed_faces()).astype(np.longlong)
        return faces

    def get_bbox_center_scale(self, idx):
        if self.use_full_image:
            center = np.array([self.raw_size[0] // 2, self.raw_size[1] // 2], dtype=np.float32)
            scale = self.raw_size[0]
            return center, scale

        # Only use hand joints or hand bbox
        if self.data_split in ["train", "trainval", "val"]:
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            center = get_annot_center(joints2d)
            scale = get_annot_scale(joints2d)
        else:  # self.data_split == "test", No gt joints annot, using handBoundingBox
            hand_bbox_coord = self.get_annot(idx)["handBoundingBox"]  # (x0, y0, x1, y1)
            hand_bbox_2d = np.array(
                [
                    [hand_bbox_coord[0], hand_bbox_coord[1]],
                    [hand_bbox_coord[2], hand_bbox_coord[3]],
                ],
                dtype=np.float32,
            )
            center = get_annot_center(hand_bbox_2d)
            scale = get_annot_scale(hand_bbox_2d)
            scale = scale

        return center, scale

    def __ho3d_get_hand_info(self, idx):
        if self.data_split == "test":
            # raise ValueError("No gt hand info in test split")
            # @NOTE: propagate the essentials with dummy values
            annot = self.get_annot(idx)
            tsl = annot["handJoints3D"]  # (3, ) only hand root provided in HO3D testset.
            joints = annot["handJoints3D"][np.newaxis, :].repeat(21, 0)  # repeat root 21 times
            pose = np.zeros(48, dtype=np.float32)  # dummy
            shape = np.zeros(10, dtype=np.float32)  # mean shape
            return joints, pose, shape, tsl

        annot = self.get_annot(idx)
        # Retrieve hand info
        joints = annot["handJoints3D"]
        pose = annot["handPose"]
        tsl = annot["handTrans"]
        shape = annot["handBeta"]
        pose = pose.astype(np.float32)
        return joints, pose, shape, tsl

    # region >>>>>> Object >>>>>>
    def get_obj_id(self, idx):
        annot = self.get_annot(idx)
        obj_id = annot["objName"]
        return obj_id

    def get_obj_verts_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._ho3d_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertices).astype(np.float32)

    def get_obj_vox_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._ho3d_load_obj_voexl(obj_id)

        vox_points = self.obj_id_to_vox[obj_id]["points"]
        return np.asarray(vox_points).astype(np.float32)

    def get_obj_vox_element_volume(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._ho3d_load_obj_voexl(obj_id)

        element_volume = self.obj_id_to_vox[obj_id]["element_volume"]
        return element_volume

    def get_obj_verts_color(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._ho3d_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        vertex_color = np.asarray(obj_mesh.visual.vertex_colors).astype(np.float32)
        vertex_color = vertex_color[:, :3] / 255.0
        return vertex_color

    def get_obj_corners_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._ho3d_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        corners = trimesh.bounds.corners(obj_mesh.bounds)
        return np.asarray(corners).astype(np.float32)

    def get_obj_faces(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._ho3d_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.faces).astype(np.longlong)

    def get_obj_transf(self, idx):
        annot = self.get_annot(idx)
        obj_id = annot["objName"]  # @NOTE: same as get_obj_id
        if obj_id not in self.obj_id_to_mesh:
            self._ho3d_load_and_process_obj(obj_id)

        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]

        v_0 = self.obj_id_to_center_offset[obj_id]
        """ HACK
        ## E is ho3d's extrinsic matrix, a 4x4 matrix with no translation.
        v_{can} = v_{raw} - v_0
        v_{cam} = E * (R * v_{raw} + t)

        => v_{raw} = v_{can} + v_0
        => v_{cam} = E * [ R * (v_{can} + v_0) + t]
        =>         = E * R * v_{can} + E * R * v_0 + E * t
        =>         = newR  * v_{can} + newt
        """

        extr_rot = self.cam_extr[:3, :3]
        rot_wrt_cam = extr_rot @ rot  # (3, 3)  newR
        tsl_wrt_cam = (extr_rot @ rot).dot(v_0) + extr_rot.dot(tsl)  # (3,)  newt
        obj_transf = build_SE3_from_Rt(rot_wrt_cam, tsl_wrt_cam)
        return obj_transf

    def get_obj_verts_3d_legacy(self, idx):
        annot = self.get_annot(idx)
        rot = cv2.Rodrigues(annot["objRot"])[0]
        tsl = annot["objTrans"]
        obj_id = annot["objName"]

        verts_c = self.get_obj_verts_can(idx)
        offset = self.obj_id_to_center_offset[obj_id]
        verts = verts_c + offset  # @NOTE this verts equal to the one initially load from .obj file

        transf_verts = rot.dot(verts.transpose()).transpose() + tsl
        transf_verts = self.cam_extr[:3, :3].dot(transf_verts.transpose()).transpose()
        return np.array(transf_verts).astype(np.float32)

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
            self._ho3d_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertex_normals).astype(np.float32)

    def get_obj_normals_3d(self, idx):
        normals_c = self.get_obj_normals_can(idx)
        obj_transf = self.get_obj_transf(idx)
        R = obj_transf[:3, :3]

        normals_3d = (R @ normals_c.T).T
        return normals_3d

    def get_obj_vox_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._ho3d_load_obj_voxel(obj_id)

        vox_points = self.obj_id_to_vox[obj_id]["points"]
        return np.asarray(vox_points).astype(np.float32)

    def get_obj_vox_element_volume(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._fphab_load_obj_voxel(obj_id)

        element_volume = self.obj_id_to_vox[obj_id]["element_volume"]
        return element_volume

    # endregion <<<<<< <<<<<<

    def get_processed_contact_info(self, idx):
        contact_info_pkl_path = self.contact_infos[idx]
        with open(contact_info_pkl_path, "rb") as bytestream:
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
