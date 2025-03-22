import os
import torch
import torch.nn as nn
import numpy as np
import pickle

from lib.utils.builder import MODEL, build_model
from lib.utils.config import CN
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.transform import rotmat_to_aa
from lib.models.model_abc import ModelABC


class HOPoseProvider:

    def __init__(self, data_path, center_idx, with_ho3d_extr) -> None:
        self.center_idx = center_idx
        self.with_ho3d_extr = with_ho3d_extr
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist")

        self.data = {}
        if os.path.isfile(data_path):
            with open(data_path, "rb") as f:
                raw = pickle.load(f)
                for i in range(len(raw)):
                    one_sample = raw[i]
                    self.data[one_sample["idx"]] = one_sample

        elif os.path.isdir(data_path):
            raise NotImplementedError("Not implemented yet")
        logger.info(f"Load HOPoseProvider from {data_path}, {len(self.data)} samples")

    def query(self, idx_list, image_path_list):
        # region >>>>> query essential data from HOPoseProvider
        '''curr_sample = {
            "idx": idx,
            "sample_id": sample_id,
            "image_path": image_path,
            "obj_faces": obj_faces,
            "obj_name": obj_name,
            "hand_pose": fitted_pose[i],
            "hand_shape": fitted_shape[i],
            "obj_rotmat": curr_obj_rotmat,
            "obj_tsl": curr_obj_tsl,
            "recov_hand_joints": fitted_joints[i],
            "recov_hand_verts": fitted_verts[i],
            "recov_obj_corners": curr_obj_corner,
            "recov_obj_verts": curr_obj_v,
        } '''
        recov_hand_verts3d = []
        recov_joints3d = []
        obj_center3d = []
        hand_center3d = []
        full_pose = []
        shape = []
        obj_rotaa = []
        obj_rotmat = []

        # endregion
        for i, idx in enumerate(idx_list):
            curr_sample = self.data[idx]
            curr_image_path = curr_sample["image_path"]
            assert curr_image_path.split("/HO3D/")[1] == image_path_list[i].split("/HO3D/")[1], \
                f"mismatch image path: {curr_image_path} vs {image_path_list[i]}"

            hand_center3d.append(curr_sample["recov_hand_joints"][self.center_idx, :])
            obj_center3d.append(curr_sample["obj_tsl"])
            curr_obj_rotmat = curr_sample["obj_rotmat"]
            if self.with_ho3d_extr:
                ho3d_extr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                curr_obj_rotmat = curr_obj_rotmat @ ho3d_extr
            obj_rotmat.append(curr_obj_rotmat)
            obj_rotaa.append(rotmat_to_aa(curr_obj_rotmat))
            full_pose.append(curr_sample["hand_pose"].reshape(-1))
            shape.append(curr_sample["hand_shape"])
            recov_joints3d.append(curr_sample["recov_hand_joints"])
            recov_hand_verts3d.append(curr_sample["recov_hand_verts"])

        hand_center3d = torch.from_numpy(np.stack(hand_center3d, axis=0)).float()
        obj_center3d = torch.from_numpy(np.stack(obj_center3d, axis=0)).float()
        full_pose = torch.from_numpy(np.stack(full_pose, axis=0)).float()
        shape = torch.from_numpy(np.stack(shape, axis=0)).float()
        obj_rotmat = torch.from_numpy(np.stack(obj_rotmat, axis=0)).float()  # (B, 3, 3)
        obj_rotaa = torch.from_numpy(np.stack(obj_rotaa, axis=0)).float()
        recov_joints3d = torch.from_numpy(np.stack(recov_joints3d, axis=0)).float()
        recov_hand_verts3d = torch.from_numpy(np.stack(recov_hand_verts3d, axis=0)).float()

        res = {
            "hand_center3d": hand_center3d,
            "obj_center3d": obj_center3d,
            "full_pose": full_pose,
            "shape": shape,
            "obj_rotmat": obj_rotmat,
            "obj_rotaa": obj_rotaa,
            "recov_joints3d": recov_joints3d,
            "recov_hand_verts3d": recov_hand_verts3d,
        }
        return res


@MODEL.register_module()
class HOPosePiCRPipeline(ModelABC):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.preset_cfg = cfg.DATA_PRESET
        self.ho_pose_provider = HOPoseProvider(cfg.HO_POSE_PATH, cfg.DATA_PRESET.CENTER_IDX, cfg.HO_POSE_WITH_HO3D_EXTR)
        self.picr_module = build_model(cfg.PICR, data_preset=cfg.DATA_PRESET)

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq
        self.picr_module.setup(summary_writer, log_freq, **kwargs)

    def forward(self, inputs, step_idx, mode="inference", **kwargs):
        if mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise NotImplementedError(f"Unsupport mode: {mode}")

    def inference_step(self, batch, step_idx, **kwargs):
        callback_fn = None
        if "callback" in kwargs:
            callback_fn = kwargs.pop("callback")

        batch_size = batch["idx"].shape[0]
        b_idx = batch["idx"].cpu().numpy().tolist()
        b_image_path = batch["image_path"]

        b_pred_res = self.ho_pose_provider.query(b_idx, b_image_path)
        obj_verts_can = batch["obj_verts_can"]
        for k, v in b_pred_res.items():
            if isinstance(v, torch.Tensor):
                b_pred_res[k] = v.to(obj_verts_can.device)
        obj_rotmat = b_pred_res["obj_rotmat"]
        obj_center3d = b_pred_res["obj_center3d"]
        recov_obj_verts3d = torch.einsum("bij, bkj -> bki", obj_rotmat, obj_verts_can) + obj_center3d
        b_pred_res["recov_obj_verts3d"] = recov_obj_verts3d
        batch.update(b_pred_res)

        contact_res = self.picr_module(batch, step_idx, mode="inference", **kwargs)
        results = {**b_pred_res, **contact_res}
        if callback_fn is not None and callable(callback_fn):
            callback_fn(results, batch, step_idx, **kwargs)

        return results

    def training_step(self, batch, step_idx):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")

    def validation_step(self, batch, step_idx):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")

    def compute_loss(self, gts, preds, **kwargs):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")


@MODEL.register_module()
class HONetPiCRPipeline(ModelABC):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.preset_cfg = cfg.DATA_PRESET
        self.honet_module = build_model(cfg.HONET, data_preset=cfg.DATA_PRESET)
        self.picr_module = build_model(cfg.PICR, data_preset=cfg.DATA_PRESET)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq
        self.honet_module.setup(summary_writer, log_freq, **kwargs)
        self.picr_module.setup(summary_writer, log_freq, **kwargs)

    def inference_step(self, batch, step_idx, **kwargs):
        callback_fn = None
        if "callback" in kwargs:
            callback_fn = kwargs.pop("callback")

        honet_res = self.honet_module(batch, step_idx, mode="inference", **kwargs)
        batch.update(honet_res)
        contact_res = self.picr_module(batch, step_idx, mode="inference", **kwargs)
        results = {**honet_res, **contact_res}

        if callback_fn is not None and callable(callback_fn):
            callback_fn(results, batch, step_idx, **kwargs)

        return results

    def training_step(self, batch, step_idx):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")

    def validation_step(self, batch, step_idx):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")

    def compute_loss(self, gts, preds, **kwargs):
        raise NotImplementedError("HONetPiCRPipeline is not trainable")

    def forward(self, inputs, step_idx, mode="inference", **kwargs):
        if mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise NotImplementedError(f"Unsupport mode: {mode}")
