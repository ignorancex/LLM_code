import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from scipy.spatial.distance import cdist

import imageio
import numpy as np
import torch
import trimesh

from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.anchorutils import anchor_load_driver

from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.contact import process_contact_info
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import (
    build_SE3_from_Rt,
    get_annot_center,
    get_annot_scale,
    persp_project,
    SE3_transform,
)
from .hdata import HODataset, auto_delegate_get


@DATASET.register_module()
class FPHAB(HODataset):

    def __init__(self, cfg):
        super(HODataset, self).__init__(cfg)
        self.mode_split = cfg.MODE_SPLIT  # action
        self.use_joints_magnetic = cfg.USE_JOINTS_MAGNETIC
        logger.info(f"FPHAB Using magnetic joints: {self.use_joints_magnetic}")

        if self.mode_split not in ["actions", "subjects", "objects"]:
            raise ValueError("Invalid split mode: {}".format(self.mode_split))

        self.reduce_size = cfg.REDUCE_SIZE
        self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]
        # Get camera info
        self.cam_extr = np.array([
            [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
            [0, 0, 0, 1],
        ],
                                 dtype=np.float32)
        self.cam_intr = np.array([
            [1395.749023, 0, 935.732544],
            [0, 1395.749268, 540.681030],
            [0, 0, 1],
        ],
                                 dtype=np.float32)
        self.reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,
            flat_hand_mean=True,
        )

        self.load_dataset()

    def _preload(self):
        self.name = "fhbhands"
        self.root = os.path.join(self.data_root, self.name)
        self.root_supp = os.path.join(self.data_root, f"{self.name}_supp")
        self.root_assets = os.path.normpath("assets")
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(self.root, "data_split_action_recognition.txt")
        small_rgb_root = os.path.join(self.root, "Video_files_480")
        if os.path.exists(small_rgb_root) and self.reduce_size:
            self.rgb_root = small_rgb_root
            self.reduce_factor = float(1 / 4)
        else:
            self.rgb_root = os.path.join(self.root, "Video_files")
            self.reduce_factor = float(1)
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")
        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite

        self.cache_info_dict = {
            "data_split": self.data_split,
            "mode_split": self.mode_split,
            "data_mode": self.data_mode,
            "rgb_root": self.rgb_root
        }
        cache_id_raw = json.dumps(self.cache_info_dict, sort_keys=True)
        self.cache_id = hashlib.md5(cache_id_raw.encode("ascii")).hexdigest()[:8]
        self.cache_path = os.path.join("common", "cache", self.name, self.data_split, f"{self.cache_id}.pkl")

    def load_dataset(self):

        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        self.all_objects_names = ["juice_bottle", "liquid_soap", "milk", "salt"]

        if os.path.exists(self.cache_path) and self.use_cache:
            with open(self.cache_path, "rb") as p_f:
                annotations = pickle.load(p_f)
            logger.info(f"Loaded cache information for dataset {self.name} from {self.cache_path}")
        else:
            subjects_infos = {}
            for subject in self.subjects:
                subject_info_path = os.path.join(self.info_root, "{}_info.txt".format(subject))
                subjects_infos[subject] = {}
                with open(subject_info_path, "r") as subject_f:
                    raw_lines = subject_f.readlines()

                for line in raw_lines[3:]:
                    line = " ".join(line.split())
                    action, action_idx, length = line.strip().split(" ")
                    subjects_infos[subject][(action, action_idx)] = length

            skel_info = self._get_skeletons(self.skeleton_root, subjects_infos)

            with open(self.info_split, "r") as annot_f:
                lines_raw = annot_f.readlines()

            train_list, test_list, all_infos = self._get_action_train_test(lines_raw, subjects_infos)

            obj_infos = self._load_object_infos(os.path.join(self.root, "Object_6D_pose_annotation_v1_1"))

            if self.mode_split == "actions":
                if self.data_split == "train":
                    sample_list = train_list
                elif self.data_split == "test":
                    sample_list = test_list
                elif self.data_split == "all":
                    sample_list = {**train_list, **test_list}
                else:
                    raise ValueError(f"Split {self.data_split} not in [train|test|all] for split_type actions")
            elif self.mode_split == "subjects":
                if self.data_split == "train":
                    subjects = ["Subject_1", "Subject_3", "Subject_4"]
                elif self.data_split == "test":
                    subjects = ["Subject_2", "Subject_5", "Subject_6"]
                else:
                    raise ValueError(f"Split {self.data_split} not in [train|test] for split_type subjects")
                self.subjects = subjects
                sample_list = all_infos
            elif self.mode_split == "objects":
                sample_list = all_infos
            else:
                raise ValueError(f"split_type should be in [action|objects|subjects], got {self.mode_split}")

            if self.mode_split != "subjects":
                self.subjects = ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"]

            image_paths = []
            magnetic_joints3d = []
            sample_infos = []
            objnames = []
            objtransforms = []

            # gather annotation
            for subject, action_name, seq_idx, frame_idx in sample_list:
                info = {"subject": subject, "action_name": action_name, "seq_idx": seq_idx, "frame_idx": frame_idx}
                triplet = (action_name, seq_idx, frame_idx)
                if subject not in self.subjects:
                    continue  # @NOTE: Skip samples from other subjects

                if subject not in obj_infos or triplet not in obj_infos[subject]:
                    continue  # @NOTE: Skip samples without objects

                obj_name, transf = obj_infos[subject][triplet]
                if obj_name not in self.all_objects_names:
                    continue  # @NOTE: Skip samples with objects not in our list

                img_path = os.path.join(self.rgb_root, subject, action_name, seq_idx, "color",
                                        self.rgb_template.format(frame_idx))
                skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
                skel = skel[self.reorder_idx]
                skel_3d = SE3_transform(skel, self.cam_extr) / 1000.0  # to meter

                # collect the results
                objtransforms.append(transf)
                objnames.append(obj_name)
                image_paths.append(img_path)
                sample_infos.append(info)
                magnetic_joints3d.append(skel_3d)

            mano_objs, mano_infos = self._load_manofits(sample_infos)
            # assemble annotation
            annotations = {
                "image_paths": image_paths,
                "magnetic_joints3d": magnetic_joints3d,
                "sample_infos": sample_infos,
                "mano_infos": mano_infos,
                "mano_objs": mano_objs,
                "objnames": objnames,
                "objtransforms": objtransforms,
            }

            # dump cache
            with open(self.cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            logger.info(f"Wrote cache for dataset {self.name} to {self.cache_path}")

        # register loaded information into object
        self.image_paths = annotations["image_paths"]
        self.magnetic_joints3d = annotations["magnetic_joints3d"]
        self.sample_infos = annotations["sample_infos"]
        self.mano_objs = annotations["mano_objs"]
        self.mano_infos = annotations["mano_infos"]
        self.objnames = annotations["objnames"]
        self.objtransforms = annotations["objtransforms"]

        if "obj" in self.data_mode:
            self.obj_root_dir = os.path.join(self.root_supp, "Object_models")
            # reserving for future use
            self.obj_id_to_mesh = {}
            self.obj_id_to_center_offset = {}
            self.obj_id_to_vox = {}

        if "contact" in self.data_mode:
            (
                self.anchor_face_vertex_index,
                self.anchor_weights,
                self.hand_vertex_merged_assignment,
                self.anchor_mapping,
            ) = anchor_load_driver(self.root_assets)
            seq_root = os.path.join(self.root_supp, "Object_contact_region_annotation_v512")
            self.contact_infos = self._load_contact_infos(seq_root=seq_root)

        # extra info: hand vertex & anchor stuff
        # this doesn't need to be cached, as it keeps the sampe for all samples
        # self.hand_palm_vertex_index = np.loadtxt(os.path.join(self.root_assets, "hand_palm_full.txt"), dtype=int)
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        self.raw_size = [int(1920 * self.reduce_factor), int(1080 * self.reduce_factor)]
        logger.info(f"{self.name} got {len(self.image_paths)} samples for data_split {self.data_split}")

        return

    def __len__(self):
        return len(self.image_paths)

    def get_side(self, idx):
        return "right"

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_image_mask(self, idx):
        raise NotImplementedError

    def get_image(self, idx):
        img_path = self.get_image_path(idx)
        img = np.array(imageio.imread(img_path, pilmode="RGB"), dtype=np.uint8)
        return img

    def get_hand_faces(self, idx):
        faces = np.array(self.mano_layer.th_faces).astype(np.longlong)
        return faces

    def get_joints_2d(self, idx):
        joints_3d = self.get_joints_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(joints_3d, cam_intr)

    def get_joints_3d_magnetic(self, idx):
        return self.magnetic_joints3d[idx]

    def get_joints_3d_manofit(self, idx):
        pose, trans, shape = self.__fhb_get_hand_info(idx)
        mano_out = self.mano_layer(torch.Tensor(pose).unsqueeze(0), torch.Tensor(shape).unsqueeze(0))
        joints = mano_out.joints.squeeze(0).numpy() + trans
        return np.array(joints).astype(np.float32)

    def get_joints_3d(self, idx):
        if self.use_joints_magnetic:
            return self.get_joints_3d_magnetic(idx)
        else:
            return self.get_joints_3d_manofit(idx)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_cam_intr(self, idx):
        return self.cam_intr

    def get_verts_3d(self, idx):
        pose, trans, shape = self.__fhb_get_hand_info(idx)
        mano_out: MANOOutput = self.mano_layer(torch.Tensor(pose).unsqueeze(0), torch.Tensor(shape).unsqueeze(0))
        verts = mano_out.verts.squeeze(0).numpy() + trans
        return np.array(verts).astype(np.float32)

    def get_verts_2d(self, idx):
        verts_3d = self.get_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(verts_3d, cam_intr)

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_mano_shape(self, idx):
        shape = self.mano_infos[idx]["shape"]
        shape = shape.astype(np.float32)
        return shape

    def get_mano_pose(self, idx):
        fullpose = self.mano_infos[idx]["fullpose"]
        mano_pose = fullpose.reshape(-1, 3).astype(np.float32)
        return mano_pose

    def __fhb_get_hand_info(self, idx):
        mano_info = self.mano_infos[idx]
        return mano_info["fullpose"], mano_info["trans"], mano_info["shape"]

    def get_bbox_center_scale(self, idx):
        if self.use_full_image:
            center = np.array([self.raw_size[0] // 2, self.raw_size[1] // 2], dtype=np.float32)
            scale = self.raw_size[0]
            return center, scale

        # Only use hand joints
        joints2d = self.get_joints_2d(idx)  # (21, 2)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)

        return center, scale

    def get_sample_identifier(self, idx):
        res = f"{self.name}_{self.data_split}_{self.mode_split}__{idx:06d}"
        return res

    def get_obj_id(self, idx):
        return self.objnames[idx]

    def get_obj_verts_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._fphab_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertices).astype(np.float32)

    def get_obj_verts_color(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._fphab_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        vertex_color = np.asarray(obj_mesh.visual.vertex_colors).astype(np.float32)
        vertex_color = vertex_color[:, :3] / 255.0
        return vertex_color

    def get_obj_faces(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._fphab_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.faces).astype(np.longlong)

    def get_obj_normals_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_mesh:
            self._fphab_load_and_process_obj(obj_id)

        obj_mesh = self.obj_id_to_mesh[obj_id]
        return np.asarray(obj_mesh.vertex_normals).astype(np.float32)

    def get_obj_vox_can(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._fphab_load_obj_voxel(obj_id)

        vox_points = self.obj_id_to_vox[obj_id]["points"]
        return np.asarray(vox_points).astype(np.float32)

    def get_obj_vox_element_volume(self, idx):
        obj_id = self.get_obj_id(idx)
        if obj_id not in self.obj_id_to_vox:
            self._fphab_load_obj_voxel(obj_id)

        element_volume = self.obj_id_to_vox[obj_id]["element_volume"]
        return element_volume

    def get_obj_transf(self, idx):
        obj_id = self.get_obj_id(idx)
        verts_can = self.get_obj_verts_can(idx)
        v_0 = self.obj_id_to_center_offset[obj_id]

        transf = self.objtransforms[idx]
        transf = self.cam_extr @ transf
        rot = transf[:3, :3]
        tsl = transf[:3, 3] / 1000.0  # to meter
        tsl_wrt_cam = rot.dot(v_0) + tsl
        tsl_wrt_cam = tsl_wrt_cam
        obj_transf = build_SE3_from_Rt(rot, tsl_wrt_cam).astype(np.float32)
        return obj_transf

    def get_obj_verts_3d(self, idx):
        verts_can = self.get_obj_verts_can(idx)
        obj_transf = self.get_obj_transf(idx)
        verts_3d = SE3_transform(verts_can, obj_transf)
        return verts_3d

    def get_obj_normals_3d(self, idx):
        normals_can = self.get_obj_normals_can(idx)
        obj_transf = self.get_obj_transf(idx)
        R = obj_transf[:3, :3]
        normals_3d = (R @ normals_can.T).T
        return normals_3d

    def get_processed_contact_info(self, idx):
        SI = self.sample_infos[idx]
        sbj, act, seq, frm = SI["subject"], SI["action_name"], SI["seq_idx"], SI["frame_idx"]
        contact_info_path = self.contact_infos[sbj][(act, seq, frm)]

        # ================================= CONTACT INFO HACK ZONE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        img_path = self.get_image_path(idx)
        if "fhbhands" in img_path:
            if "pour_milk" in img_path:
                contact_info_path = re.sub(r"Subject_\d", "Subject_1", contact_info_path)
                contact_info_path = re.sub(r"/\d/", "/2/", contact_info_path)
                contact_info_path = re.sub(r"\d{4}.pkl", "0159.pkl", contact_info_path)
            elif "/Subject_6/pour_liquid_soap" in img_path:
                contact_info_path = re.sub(r"Subject_\d", "Subject_3", contact_info_path)
                contact_info_path = re.sub(r"\d{4}.pkl", "0010.pkl", contact_info_path)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        with open(contact_info_path, "rb") as bytestream:
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

    def _fphab_load_and_process_obj(self, obj_id):
        obj_name = obj_id if obj_id != "juice_bottle" else "juice"
        obj_path = os.path.join(self.obj_root_dir, f"{obj_name}_model", f"{obj_name}_ds_normal.obj")
        obj_mesh = trimesh.load(obj_path, process=False)

        verts_0 = np.asarray(obj_mesh.vertices).astype(np.float32)
        offset = (verts_0.min(0) + verts_0.max(0)) / 2.0
        verts_1 = verts_0 - offset
        obj_mesh.vertices = verts_1  # @NOTE: assign back to obj_mesh

        self.obj_id_to_mesh[obj_id] = obj_mesh
        self.obj_id_to_center_offset[obj_id] = offset

    def _fphab_load_obj_voxel(self, obj_id):
        obj_name = obj_id if obj_id != "juice_bottle" else "juice"
        voxel_dir = self.obj_root_dir.replace("Object_models", "Object_models_binvox")
        obj_path = os.path.join(voxel_dir, f"{obj_name}_model", f"{obj_name}_model.binvox")
        vox = trimesh.load(obj_path)

        voxpt = np.asarray(vox.points).astype(np.float32)
        offset = (voxpt.min(0) + voxpt.max(0)) / 2.0
        voxpt = voxpt - offset

        self.obj_id_to_vox[obj_id] = {
            "points": voxpt,
            "matrix": np.array(vox.matrix),
            "element_volume": vox.element_volume,
        }
        return vox

    @staticmethod
    def _load_object_infos(seq_root="data/fhbhands/Object_6D_pose_annotation_v1_1"):
        subjects = os.listdir(seq_root)
        annots = {}
        clip_lengths = {}
        for subject in subjects:
            subject_dict = {}
            subj_path = os.path.join(seq_root, subject)
            actions = os.listdir(subj_path)
            clips = 0
            for action in actions:
                object_name = "_".join(action.split("_")[1:])
                action_path = os.path.join(subj_path, action)
                seqs = os.listdir(action_path)
                clips += len(seqs)
                for seq in seqs:
                    seq_path = os.path.join(action_path, seq, "object_pose.txt")
                    with open(seq_path, "r") as seq_f:
                        raw_lines = seq_f.readlines()
                    for raw_line in raw_lines:
                        line = raw_line.strip().split(" ")
                        frame_idx = int(line[0])
                        trans_matrix = np.array(line[1:]).astype(np.float32)
                        trans_matrix = trans_matrix.reshape(4, 4).transpose()
                        subject_dict[(action, seq, frame_idx)] = (object_name, trans_matrix)
            clip_lengths[subject] = clip_lengths
            annots[subject] = subject_dict
        return annots

    @staticmethod
    def _get_action_train_test(lines_raw, subjects_info):
        """
        Returns dicts of samples where key is
            subject: name of subject
            action_name: action class
            action_seq_idx: idx of action instance
            frame_idx
        and value is the idx of the action class
        """
        all_infos = []
        test_split = False
        test_samples = {}
        train_samples = {}
        for line in lines_raw[1:]:
            if line.startswith("Test"):
                test_split = True
                continue
            subject, action_name, action_seq_idx = line.split(" ")[0].split("/")
            action_idx = line.split(" ")[1].strip()  # Action classif index
            frame_nb = int(subjects_info[subject][(action_name, action_seq_idx)])
            for frame_idx in range(frame_nb):
                sample_info = (subject, action_name, action_seq_idx, frame_idx)
                if test_split:
                    test_samples[sample_info] = action_idx
                else:
                    train_samples[sample_info] = action_idx
                all_infos.append(sample_info)
        test_nb = len(np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in test_samples), axis=0))
        assert test_nb == 575, "Should get 575 test samples, got {}".format(test_nb)
        train_nb = len(np.unique(list((sub, act_n, act_seq) for (sub, act_n, act_seq, _) in train_samples), axis=0))
        # 600 - 1 Subject5/use_flash/6 discarded sample
        assert train_nb == 600 or train_nb == 599, "Should get 599 train samples, got {}".format(train_nb)
        assert len(test_samples) + len(train_samples) == len(all_infos)
        return train_samples, test_samples, all_infos

    @staticmethod
    def _load_contact_infos(seq_root="data/fhbhands_supp/Object_contact_region_annotation_v512"):
        re_strip_frame_idx = re.compile(r"contact_info_([0-9]*).pkl")
        subjects = os.listdir(seq_root)
        contact_blob = {}
        clip_lengths = {}
        for subject in subjects:
            subject_dict = {}
            subject_path = os.path.join(seq_root, subject)
            actions = os.listdir(subject_path)
            clips = 0
            for action in actions:
                object_name = "_".join(action.split("_")[1:])
                action_path = os.path.join(subject_path, action)
                seqs = os.listdir(action_path)
                clips += len(seqs)
                for seq in seqs:
                    sel_seq_path = os.path.join(action_path, seq)
                    all_pkl = sorted(os.listdir(sel_seq_path))
                    for pkl_name in all_pkl:
                        try:
                            current_frame_idx = re_strip_frame_idx.match(pkl_name).groups()[0]
                            current_frame_idx = int(current_frame_idx)
                            pkl_target = os.path.join(sel_seq_path, pkl_name)
                            subject_dict[(action, seq, current_frame_idx)] = pkl_target
                        except IndexError as e:
                            print(f"regular expression parsing error at {pkl_name}, location {subject}.{action}.{seq}")
                            print(e)
            clip_lengths[subject] = clip_lengths
            contact_blob[subject] = subject_dict
        return contact_blob

    @staticmethod
    def _load_manofits(sample_infos, fit_root="assets/fhbhands_fits"):
        obj_paths = []
        metas = []
        for sample_info in sample_infos:
            hand_seq_path = os.path.join(fit_root, sample_info["subject"], sample_info["action_name"],
                                         sample_info["seq_idx"], "pkls.pkl")
            with open(hand_seq_path, "rb") as p_f:
                mano_info = pickle.load(p_f)
            frame_name = f"{sample_info['frame_idx']:06d}.pkl"
            hand_info = mano_info[frame_name]
            metas.append(hand_info)
            hand_obj_path = os.path.join(
                fit_root,
                sample_info["subject"],
                sample_info["action_name"],
                sample_info["seq_idx"],
                "obj",
                f"{sample_info['frame_idx']:06d}.obj",
            )
            obj_paths.append(hand_obj_path)
        return obj_paths, metas

    @staticmethod
    def _get_skeletons(skeleton_root, subjects_info, use_cache=True):
        cache_path = os.path.join("common/cache/fhbhands/skels.pkl")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                skelet_dict = pickle.load(p_f)
            logger.info(f"Loaded fhb skel info from {cache_path}")
        else:
            skelet_dict = defaultdict(dict)
            for subject, samples in subjects_info.items():
                for (action, seq_idx) in samples:
                    skeleton_path = os.path.join(skeleton_root, subject, action, seq_idx, "skeleton.txt")
                    skeleton_vals = np.loadtxt(skeleton_path)
                    if len(skeleton_vals):
                        assert np.all(skeleton_vals[:, 0] == list(range(skeleton_vals.shape[0]))), \
                            "row idxs should match frame idx failed at {}".format(skeleton_path)
                        skelet_dict[subject][(action,seq_idx)] = skeleton_vals[:, 1:]\
                            .reshape(skeleton_vals.shape[0], 21, -1)
                    else:
                        # Handle sequences of size 0
                        skelet_dict[subject, action, seq_idx] = skeleton_vals
            with open(cache_path, "wb") as p_f:
                pickle.dump(skelet_dict, p_f)
        return skelet_dict


@auto_delegate_get(incls=FPHAB)
@DATASET.register_module()
class FilteredFPHAB(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.name = type(self).__name__
        self.base_dataset = FPHAB(cfg)
        self.data_split = self.base_dataset.data_split
        self.mode_split = self.base_dataset.mode_split
        self.use_cache = bool(cfg.DATA_PRESET.USE_CACHE)
        self.filter_no_contact = bool(cfg.DATA_PRESET.FILTER_NO_CONTACT)
        self.filter_dist_thresh = float(cfg.DATA_PRESET.FILTER_DIST_THRESH)

        self.inlier_indices = None
        if self.filter_no_contact:
            self.inlier_indices = self.build_inlier_indices()
        logger.warning(f"{self.name} {self.data_split} got {len(self)} samples after contact filter")

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

    def __getattr__(self, name):
        """
        Intercept calls to get_xxx methods and redirect them.
        """
        base_attr = getattr(self.base_dataset, name)
        if callable(base_attr) and name.startswith("get_"):

            def method(idx, *args, **kwargs):
                if self.inlier_indices is not None:
                    real_idx = self.inlier_indices[idx]
                else:
                    real_idx = idx
                return base_attr(real_idx, *args, **kwargs)

            return method
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def build_inlier_indices(self):
        self.indices_id_dict = {
            "data_split": self.base_dataset.data_split,
            "mode_split": self.base_dataset.mode_split,
            "filter_no_contact": self.filter_no_contact,
        }
        indices_id = json.dumps(self.indices_id_dict, sort_keys=True)
        indices_id = hashlib.md5(indices_id.encode("ascii")).hexdigest()[:8]
        indices_cache_path = os.path.join("common", "cache", self.base_dataset.name, self.base_dataset.data_split,
                                          f"{indices_id}.pkl")
        os.makedirs(os.path.dirname(indices_cache_path), exist_ok=True)

        if os.path.exists(indices_cache_path) and self.use_cache:
            with open(indices_cache_path, "rb") as p_f:
                inlier_indices = pickle.load(p_f)
            logger.info(f"Loaded filter indices for dataset {self.base_dataset.name} from {indices_cache_path}")
        else:
            # @NOTE: calculate filter indices
            inlier_indices = []
            for idx in etqdm(range(len(self.base_dataset))):
                verts = self.base_dataset.get_obj_verts_3d(idx)
                joints = self.base_dataset.get_joints_3d(idx)
                all_dists = cdist(verts, joints)
                if all_dists.min() > self.filter_dist_thresh:
                    continue
                inlier_indices.append(idx)
            # dump cache
            with open(indices_cache_path, "wb") as fid:
                pickle.dump(inlier_indices, fid)
            logger.info(f"Wrote filter indices for dataset {self.base_dataset.name} to {indices_cache_path}")
        return inlier_indices
