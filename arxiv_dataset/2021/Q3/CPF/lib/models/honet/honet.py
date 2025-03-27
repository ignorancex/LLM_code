import os
import torch
import torch.nn as nn
import pickle

from lib.utils.builder import MODEL
from lib.utils.config import CN
from lib.utils.net_utils import load_weights, recurse_freeze
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.recorder import Recorder
from lib.utils.transform import batch_persp_project, aa_to_rotmat, rotmat_to_aa

from lib.models.model_abc import ModelABC
from lib.models.honet.manobranch import ManoAdaptor, ManoBranch
from lib.models.honet.transhead import TransHead, recover_3d_proj
from lib.models.backbones.resnet import resnet18, resnet50
from manotorch.manolayer import ManoLayer


@MODEL.register_module()
class HONet(ModelABC):

    def __init__(self, cfg: CN):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.mano_ncomps = cfg.MANO_NCOMPS
        self.resnet_version = cfg.RESNET_VERSION
        self.mano_neurons = cfg.MANO_NEURONS
        self.center_idx = int(cfg.DATA_PRESET.CENTER_IDX)
        self.use_fhb_adaptor = bool(cfg.USE_FHB_ADAPTOR)
        self.obj_trans_factor = float(cfg.OBJ_TRANS_FACTOR)
        self.obj_scale_factor = float(cfg.OBJ_SCALE_FACTOR)

        if int(self.resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet18(pretrained=True)
        elif int(self.resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet50(pretrained=True)
        else:
            logger.error("Resnet {} not supported".format(self.resnet_version))
            raise NotImplementedError()

        mano_base_neurons = [img_feature_size] + self.mano_neurons
        self.base_net = base_net
        # Predict translation and scaling for hand
        self.mano_transhead = TransHead(base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=3)
        # Predict translation, scaling and rotation for object
        self.obj_transhead = TransHead(base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=6)

        self.mano_branch = ManoBranch(
            ncomps=self.mano_ncomps,
            base_neurons=mano_base_neurons,
            center_idx=self.center_idx,
        )

        self.adaptor = None
        if self.use_fhb_adaptor:
            load_fhb_path = f"assets/mano/fhb_skel_centeridx{self.center_idx}.pkl"
            with open(load_fhb_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
            self.register_buffer("fhb_shape", torch.Tensor(exp_data["shape"]))
            self.adaptor = ManoAdaptor(self.mano_branch.mano_layer, load_fhb_path)
            recurse_freeze(self.adaptor)

        load_weights(self, pretrained=cfg.PRETRAINED, strict=True)

        self.pretrained_with_ho3d_extr = cfg.get("PRETRAINED_WITH_HO3D_EXTR", False)
        """ Explain. 1
        The early phases of the CPF project (ICCV2021) used a specific HO3D-obj's canonical frame as: E @ V_c. 
        Here, E represents the HO3D's inherent camera-coord transformation, 
        and V_c denotes the actual canonical mesh extracted from the scanned file.
        In such specific context, the rotation R of the object, predicted by the network, represents only a fraction 
        of the total rotation. The complete rotation should be represented as R' = R @ E. 
        """
        if self.pretrained_with_ho3d_extr:
            self.ho3d_extr = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32).unsqueeze(0)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

        self.mano_layer = ManoLayer(mano_assets_root="assets/mano_v1_2",
                                    flat_hand_mean=True,
                                    center_idx=self.center_idx)

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq

    def _recover_mano(self, sample, features):
        # Get hand projection, centered

        mano_results = self.mano_branch(features)
        if self.adaptor:
            """ Explain. 2
            Recall  that the GT joints of FPHAB dataset are obtained by the surfaces-attacted 
            megnatic sensors. To obtain the these kind of joints from the predicted MANO param: 
                \theta, \beta, 
            the original MANO \J regressor ( \J * V -> joints) is not suitable. 
            So this adaptor re-trian another \J regressor to fit the FPHAB dataset.
            """
            adapt_joints, _ = self.adaptor(mano_results["verts3d"])
            adapt_joints = adapt_joints.transpose(1, 2)
            mano_results["joints3d"] = adapt_joints - adapt_joints[:, self.center_idx].unsqueeze(1)
            mano_results["verts3d"] = mano_results["verts3d"] - adapt_joints[:, self.center_idx].unsqueeze(1)

        # Recover hand position in camera coordinates
        scaletrans = self.mano_transhead(features)
        trans = scaletrans[:, 1:]
        scale = scaletrans[:, :1]

        final_trans = trans.unsqueeze(1) * self.obj_trans_factor
        final_scale = scale.view(scale.shape[0], 1, 1) * self.obj_scale_factor
        height, width = tuple(sample["image"].shape[2:])
        camintr = sample["target_cam_intr"]
        recov_joints3d, hand_center3d = recover_3d_proj(mano_results["joints3d"],
                                                        camintr,
                                                        final_scale,
                                                        final_trans,
                                                        input_res=(width, height))
        recov_hand_verts3d = mano_results["verts3d"] + hand_center3d.unsqueeze(1)
        proj_joints2d = batch_persp_project(recov_joints3d, camintr)
        proj_verts2d = batch_persp_project(recov_hand_verts3d, camintr)

        # @NOTE: recov_joints3d = joints3d + hand_center3d
        mano_results["joints2d"] = proj_joints2d
        mano_results["hand_center3d"] = hand_center3d  # ===== To PICR =====
        mano_results["recov_joints3d"] = recov_joints3d  # ===== To PICR =====
        mano_results["recov_hand_verts3d"] = recov_hand_verts3d  # ===== To PICR =====
        mano_results["verts2d"] = proj_verts2d
        mano_results["hand_pretrans"] = trans
        mano_results["hand_prescale"] = scale
        mano_results["hand_trans"] = final_trans
        mano_results["hand_scale"] = final_scale

        return mano_results

    def _recover_object(self, sample, features):
        """
        Compute object vertex and corner positions in camera coordinates by predicting object translation
        and scaling, and recovering 3D positions given known object model
        """
        scaletrans_obj = self.obj_transhead(features)
        batch_size = scaletrans_obj.shape[0]
        scale = scaletrans_obj[:, :1]
        trans = scaletrans_obj[:, 1:3]
        rotaxisang = scaletrans_obj[:, 3:]  # (B, 3)

        rotmat = aa_to_rotmat(rotaxisang)  # # (B, 3, 3)
        if self.pretrained_with_ho3d_extr:
            # NOTE@Explain 1: R' = R @ E.
            ho3d_extr = self.ho3d_extr.to(rotmat.device)
            rotmat = rotmat.bmm(ho3d_extr.expand(batch_size, 3, 3))
            rotaxisang = rotmat_to_aa(rotmat)

        obj_verts_can = sample["obj_verts_can"]
        obj_verts_rot = rotmat.bmm(obj_verts_can.float().transpose(1, 2)).transpose(1, 2)

        final_trans = trans.unsqueeze(1) * self.obj_trans_factor
        final_scale = scale.view(batch_size, 1, 1) * self.obj_scale_factor
        height, width = tuple(sample["image"].shape[2:])
        camintr = sample["target_cam_intr"]
        recov_obj_verts3d, obj_center3d = recover_3d_proj(obj_verts_rot,
                                                          camintr,
                                                          final_scale,
                                                          final_trans,
                                                          input_res=(width, height))

        # Recover 2D positions given camera intrinsic parameters and object vertex
        # coordinates in camera coordinate reference
        pred_obj_verts2d = batch_persp_project(recov_obj_verts3d, camintr)

        obj_results = {
            "obj_verts2d": pred_obj_verts2d,
            "obj_verts3d": obj_verts_rot,
            "obj_center3d": obj_center3d,
            "recov_obj_verts3d": recov_obj_verts3d,
            # "recov_obj_corners3d": recov_obj_corners3d,
            "obj_scale": final_scale,
            "obj_prescale": scale,
            "obj_rotaa": rotaxisang,
            "obj_rotmat": rotmat,
            "obj_trans": final_trans,
            "obj_pretrans": trans,
            # "obj_corners2d": pred_obj_corners2d,
            # "obj_corners3d": rot_obj_corners,
        }

        return obj_results

    def _forward_impl(self, sample, **kwargs):
        results = {}
        image = sample["image"]
        features, _ = self.base_net(image)

        mano_results = self._recover_mano(sample, features)
        results.update(mano_results)

        obj_results = self._recover_object(sample, features)
        results.update(obj_results)

        return results

    def inference_step(self, batch, step_idx, **kwargs):
        prd = self._forward_impl(batch, **kwargs)

        if "callback" in kwargs:
            callback = kwargs.pop("callback")
            if callable(callback):
                callback(prd, batch, step_idx, **kwargs)

        return prd

    def testing_step(self, batch, step_idx, **kwargs):
        raise NotImplementedError()

    def training_step(self, batch, step_idx, **kwargs):
        raise NotImplementedError()

    def validation_step(self, batch, step_idx, **kwargs):
        raise NotImplementedError()

    def compute_loss(self):
        raise NotImplementedError()

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
