import traceback
from abc import ABC, abstractmethod
from typing import Dict, List
import warnings
import numpy as np
import torch
import os
import inspect

from ..utils.builder import build_transform
from ..utils.logger import logger
from ..utils.misc import CONST
from ..utils.transform import flip_2d, flip_3d, aa_to_rotmat, rotmat_to_aa
from torch.utils.data._utils.collate import default_collate\


class HDataset(ABC):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_mode = cfg.DATA_MODE
        self.data_root = cfg.DATA_ROOT
        self.data_split = cfg.DATA_SPLIT
        self.use_cache = cfg.DATA_PRESET.USE_CACHE
        self.use_full_image = bool(cfg.DATA_PRESET.USE_FULL_IMAGE)
        self.bbox_expand_ratio = float(cfg.DATA_PRESET.BBOX_EXPAND_RATIO)
        if self.use_full_image is True and self.bbox_expand_ratio != 1.0:
            warnings.warn("When using full image, bbox expand ratio should be 1.0")
            self.bbox_expand_ratio = 1.0

        self.image_size = cfg.DATA_PRESET.IMAGE_SIZE  # (W, H)
        self.data_preset = cfg.DATA_PRESET
        self.center_idx = int(cfg.DATA_PRESET.CENTER_IDX)
        self.is_inference = bool(cfg.get("IS_INFERENCE", False))
        self.root_assets = os.path.normpath("assets")
        self.transform = build_transform(cfg=cfg.TRANSFORM, data_preset=self.data_preset)
        logger.info(f"Initialized abstract class: HDataset")

    def __len__(self):
        return len(self)

    @staticmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_image(self, idx):
        pass

    @abstractmethod
    def get_image_path(self, idx):
        pass

    @abstractmethod
    def get_joints_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_uvd(self, idx):
        pass

    @abstractmethod
    def get_joints_2d(self, idx):
        pass

    @abstractmethod
    def get_joints_uvd(self, idx):
        pass

    @abstractmethod
    def get_cam_intr(self, idx):
        pass

    @abstractmethod
    def get_side(self, idx):
        pass

    @abstractmethod
    def get_bbox_center_scale(self, idx):
        pass

    @abstractmethod
    def get_sample_identifier(self, idx):
        pass

    @abstractmethod
    def get_mano_pose(self, idx):
        pass

    @abstractmethod
    def get_mano_shape(self, idx):
        pass

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    # visible in raw image
    def get_joints_2d_vis(self, joints_2d=None, img_size=None, **kwargs):
        joints_vis = np.all((0 <= joints_2d) & (joints_2d < img_size), axis=1)
        return joints_vis.astype(np.float32)

    def getitem_2d(self, idx):
        hand_side = self.get_side(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, img_size=raw_size)

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
        }

        return image, label

    def getitem_uvd(self, idx):
        # Support FreiHAND, HO3D, DexYCB, YT3D, TMANO,
        hand_side = self.get_side(idx)

        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, img_size=raw_size)

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "verts_uvd": verts_uvd,
            "joints_uvd": joints_uvd,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
        }

        return image, label

    def getitem_3d(self, idx):
        # Support FreiHAND, HO3D, DexYCB
        sample_id = self.get_sample_identifier(idx)
        hand_side = self.get_side(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        joints_3d = self.get_joints_3d(idx)
        verts_3d = self.get_verts_3d(idx)
        joints_2d = self.get_joints_2d(idx)

        image_path = self.get_image_path(idx)
        mano_pose = self.get_mano_pose(idx)
        mano_shape = self.get_mano_shape(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, img_size=raw_size)

        label = {
            "idx": idx,
            "sample_id": sample_id,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "mano_pose": mano_pose,
            "mano_shape": mano_shape,
            "image_path": image_path,
            "raw_size": raw_size,
            "hand_side": hand_side,
        }

        return image, label

    def __getitem__(self, idx):
        if self.data_mode not in ["2D", "UVD", "3D"]:
            raise NotImplementedError(f"Unknown data mode: {self.data_mode}")

        if self.data_mode == "2D":
            image, label = self.getitem_2d(idx)
        elif self.data_mode == "UVD":
            image, label = self.getitem_uvd(idx)
        elif self.data_mode == "3D":
            image, label = self.getitem_3d(idx)

        results = self.transform(image, label)  # @NOTE data augmentation
        results.update(label)

        return results


class HODataset(HDataset, ABC):

    def __init__(self, cfg):
        super(HDataset, self).__init__(cfg)
        logger.info(f"Initialized abstract class: HODataset")

    @abstractmethod
    def get_obj_id(self, idx):
        pass

    @abstractmethod
    def get_obj_faces(self, idx):
        pass

    @abstractmethod
    def get_obj_transf(self, idx):
        pass

    @abstractmethod
    def get_obj_normals_3d(self, idx):
        pass

    @abstractmethod
    def get_obj_verts_3d(self, idx):
        pass

    @abstractmethod
    def get_obj_verts_can(self, idx):
        pass

    @abstractmethod
    def get_obj_normals_can(self, idx):
        pass

    def get_processed_contact_info(self, idx):
        return {}

    def getitem_3d_hand_obj(self, idx):
        image, label = self.getitem_3d(idx)
        label["obj_verts_can"] = self.get_obj_verts_can(idx)
        label["obj_id"] = self.get_obj_id(idx)
        label["obj_verts_3d"] = self.get_obj_verts_3d(idx)
        label["obj_normals_3d"] = self.get_obj_normals_3d(idx)
        label["obj_transf"] = self.get_obj_transf(idx)
        label["obj_faces"] = self.get_obj_faces(idx)
        label["obj_verts_color"] = self.get_obj_verts_color(idx)

        return image, label

    def __getitem__(self, idx):
        if self.data_mode not in ["2d", "uvd", "3d", "3d_hand_obj", "3d_hand_obj_contact"]:
            raise NotImplementedError(f"Unknown data mode: {self.data_mode}")

        if self.data_mode == "2d":
            image, label = self.getitem_2d(idx)
        elif self.data_mode == "uvd":
            image, label = self.getitem_uvd(idx)
        elif self.data_mode == "3d":
            image, label = self.getitem_3d(idx)
        elif self.data_mode == "3d_hand_obj":
            image, label = self.getitem_3d_hand_obj(idx)
        elif self.data_mode == "3d_hand_obj_contact":
            image, label = self.getitem_3d_hand_obj(idx)
            contact_label = self.get_processed_contact_info(idx)
            label.update(contact_label)

        results = self.transform(image, label)  # @NOTE data augmentation
        results.update(label)

        return results


def ho_data_collate(batch: List[Dict]):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """
    extend_queries = {
        # before aug
        "obj_verts_can",
        "obj_normals_can",
        "obj_verts_3d",
        "obj_normals_3d",
        "obj_faces",

        # after aug
        "target_obj_verts_3d",
        "target_obj_normals_3d",

        # contact query
        "vertex_contact",
        "contact_region_id",
        "anchor_id",
        "anchor_dist",
        "anchor_elasti",
        "anchor_padding_mask"
    }
    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        padding_query_field = match_collate_queries(pop_query)
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            orig_len = pop_value.shape[0]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
            if padding_query_field not in sample:
                # generate a new field, contains padding mask
                # note that only the beginning pop_value.shape[0] points are in effect
                # so the mask will be a vector of length max_size, with origin_len ones in the beginning
                padding_mask = np.zeros(max_size, dtype=int)
                padding_mask[:orig_len] = 1
                sample[padding_query_field] = padding_mask

    # store the mask filtering the points
    batch = default_collate(batch)  # this function np -> torch
    return batch


def match_collate_queries(query_spin):
    object_vertex_queries = [
        # before aug
        "obj_verts_can",
        "obj_normals_can",
        "obj_verts_3d",
        "obj_normals_3d",

        # after aug
        "target_obj_verts_3d",
        "target_obj_normals_3d",

        # contact query
        "vertex_contact",
        "contact_region_id",
        "anchor_id",
        "anchor_dist",
        "anchor_elasti",
        "anchor_padding_mask"
    ]
    object_face_quries = [
        "obj_faces",
    ]

    if query_spin in object_vertex_queries:
        return "obj_verts_padding_mask"
    elif query_spin in object_face_quries:
        return "obj_faces_padding_mask"


def auto_delegate_get(incls):  # incls must be a HDataset class

    def decorator(cls):
        for name, member in inspect.getmembers(incls, predicate=inspect.isfunction):
            if name.startswith("get_"):

                def wrapped_get(self, idx, _name=name):
                    if self.inlier_indices is not None:
                        return getattr(self.base_dataset, _name)(self.inlier_indices[idx])
                    else:
                        return getattr(self.base_dataset, _name)(idx)

                setattr(cls, name, wrapped_get)
        return cls

    return decorator
