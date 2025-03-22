from enum import Enum, auto
import os
import torch
import pickle
from lib.utils.contact import dumped_process_contact_info
from manotorch.utils.anchorutils import anchor_load


class CIDumpedQueries(Enum):
    class_type = "ci_dumped"
    HAND_VERTS_3D = f"{class_type}.hand_verts_3d"
    HAND_JOINTS_3D = f"{class_type}.hand_joints_3d"
    HAND_TSL = f"{class_type}.hand_tsl"
    HAND_ROT = f"{class_type}.hand_rot"
    HAND_POSE = f"{class_type}.hand_pose"
    HAND_SHAPE = f"{class_type}.hand_shape"
    OBJ_VERTS_3D = f"{class_type}.obj_verts_3d"
    OBJ_TRANSF = f"{class_type}.obj_transf"
    OBJ_TSL = f"{class_type}.obj_tsl"
    OBJ_ROT = f"{class_type}.obj_rot"
    VERTEX_CONTACT = f"{class_type}.vertex_contact"  # NEW: for each vertex, whether it is in contact with hand
    CONTACT_REGION_ID = f"{class_type}.contact_region_id"  # NEW: returns region id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ID = f"{class_type}.contact_anchor_id"  # NEW: returns anchor id [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_ELASTI = f"{class_type}.contact_anchor_elasti"  # NEW: returns anchor elasti [[ NOTE ITS INTERACTION WITH PADDING ]]
    CONTACT_ANCHOR_PADDING_MASK = f"{class_type}.contact_anchor_padding_mask"  # NEW: if padding enabled, this field will be append to the query


class CIDumpedData():

    def __init__(self, sample_id: str, data_prefix: str):
        (
            self.anchor_face_vertex_index,
            self.anchor_weights,
            self.hand_vertex_merged_assignment,
            self.anchor_mapping,
        ) = anchor_load("assets/anchor")

        self.contact_pad_vertex = True
        self.contact_pad_anchor = True
        self.contact_elasti_th = 0.00
        self.contact_data_path = os.path.join(data_prefix, f"{sample_id}_contact.pkl")
        self.honet_data_path = os.path.join(data_prefix, f"{sample_id}_honet.pkl")

        # check if the files exist
        assert os.path.exists(self.contact_data_path), f"{self.contact_data_path} does not exist"
        assert os.path.exists(self.honet_data_path), f"{self.honet_data_path} does not exist"

    def get_dumped_contact_info(self):
        with open(self.contact_data_path, "rb") as bytestream:
            dumped_contact_info_list = pickle.load(bytestream)

        (vertex_contact, hand_region, anchor_id, anchor_elasti, anchor_padding_mask) = dumped_process_contact_info(
            dumped_contact_info_list,
            self.anchor_mapping,
            pad_vertex=self.contact_pad_vertex,
            pad_anchor=self.contact_pad_anchor,
            elasti_th=self.contact_elasti_th,
        )
        contact_info_dict = {
            "vertex_contact": torch.from_numpy(vertex_contact),
            "hand_region": torch.from_numpy(hand_region),
            "anchor_id": torch.from_numpy(anchor_id),
            "anchor_elasti": torch.from_numpy(anchor_elasti),
            "anchor_padding_mask": torch.from_numpy(anchor_padding_mask),
        }
        return contact_info_dict

    def get_dumped_honet_info(self):
        with open(self.honet_data_path, "rb") as bytestream:
            dumped_pose_dict = pickle.load(bytestream)

        return dumped_pose_dict

    def get(self):
        sample = {}
        contact_dict = self.get_dumped_contact_info()
        honet_dict = self.get_dumped_honet_info()

        sample[CIDumpedQueries.HAND_VERTS_3D] = honet_dict["hand_verts_3d"]
        sample[CIDumpedQueries.HAND_JOINTS_3D] = honet_dict["hand_joints_3d"]
        sample[CIDumpedQueries.HAND_TSL] = honet_dict["hand_tsl"].reshape((3,))
        sample[CIDumpedQueries.HAND_ROT] = honet_dict["hand_full_pose"][0:3]
        sample[CIDumpedQueries.HAND_POSE] = honet_dict["hand_full_pose"].reshape((16, 3))
        sample[CIDumpedQueries.HAND_SHAPE] = honet_dict["hand_shape"]
        sample[CIDumpedQueries.OBJ_TSL] = honet_dict["obj_tsl"].reshape((3,))
        sample[CIDumpedQueries.OBJ_ROT] = honet_dict["obj_rot"].reshape((3,))
        sample[CIDumpedQueries.OBJ_VERTS_3D] = honet_dict["obj_verts_3d"]

        sample[CIDumpedQueries.VERTEX_CONTACT] = contact_dict["vertex_contact"]
        sample[CIDumpedQueries.CONTACT_REGION_ID] = contact_dict["hand_region"]
        sample[CIDumpedQueries.CONTACT_ANCHOR_ID] = contact_dict["anchor_id"]
        sample[CIDumpedQueries.CONTACT_ANCHOR_ELASTI] = contact_dict["anchor_elasti"]
        sample[CIDumpedQueries.CONTACT_ANCHOR_PADDING_MASK] = contact_dict["anchor_padding_mask"]

        return sample
