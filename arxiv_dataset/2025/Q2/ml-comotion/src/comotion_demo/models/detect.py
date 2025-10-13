# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
from collections import namedtuple
from dataclasses import dataclass

import einops as eo
import torch
from torch import nn

from ..utils import helper, smpl_kinematics
from . import backbones, layers

curr_dir = os.path.abspath(os.path.dirname(__file__))
PYTORCH_CHECKPOINT_PATH = f"{curr_dir}/../data/comotion_detection_checkpoint.pt"
COREML_CHECKPOINT_PATH = f"{curr_dir}/../data/comotion_detection.mlpackage"

DetectionOutput = namedtuple(
    "DetectOutput",
    [
        "image_features",
        "betas",
        "delta_root_orient",
        "delta_body_pose",
        "trans",
        "conf",
    ],
)


@dataclass
class CoMotionDetectConfig:
    backbone_choice: str = "ConvNextV2"
    pose_embed_dim: int = 256
    hidden_dim: int = 512
    rot_embed_dim: int = 8


class DetectionHead(nn.Module):
    """CoMotion detection head.

    Accepts as input image features and an intrinsics matrix to produce a large
    pool of candidate SMPL poses and corresponding confidences.
    """

    def __init__(
        self,
        input_dim,
        output_split,
        hidden_dim=512,
        num_slots=4,
        depth_adj=128,
        dropout=0.0,
    ):
        """Initialize CoMotion detection head."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.output_split = output_split
        self.depth_adj = depth_adj
        self.dropout = dropout

        # Blocks for feature pyramid encoding.
        self.enc1 = nn.Sequential(
            layers.DownsampleConvNextBlock(input_dim, hidden_dim),
            layers.DownsampleConvNextBlock(hidden_dim, 2 * hidden_dim, dropout=dropout),
        )
        self.enc2 = layers.DownsampleConvNextBlock(2 * hidden_dim, dropout=dropout)
        self.enc3 = layers.DownsampleConvNextBlock(2 * hidden_dim, dropout=dropout)

        out_dim = sum(self.output_split) * self.num_slots
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, 1),
                    layers.LayerNorm2d(2 * hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(2 * hidden_dim, out_dim, 1),
                )
                for _ in range(3)
            ]
        )

    def forward(self, px_feats, K, pooling=8, return_feats=False):
        """Detect poses from image features and intrinsics.

        Args:
        ----
            px_feats: image features from the backbone network.
            K: image intrinsics.
            pooling: downsampling factor from input image to features.
            return_feats: return image feature pyramid.

        Return:
        ------
            detections: 1344 candidates = 1024 + 256 + 64 from 3 levels.
            feat_pyramid: image feature pyramid.

        """
        # Rescale factor
        ht, wd = px_feats.shape[-2:]
        rescale = max(ht, wd) * pooling
        calib_adj = K[..., 0, 0][:, None, None]

        # Apply encoding and decoding layers
        x1 = self.enc1(px_feats)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        feat_pyramid = [x1, x2, x3]

        # Post-process to produce full set of detections
        pred_state = []
        for pyr_idx, feats in enumerate(feat_pyramid):
            feats = feats.float()

            ht, wd = feats.shape[-2:]
            ref_xy = helper.get_grid(ht, wd, device=feats.device) + 0.5 / max(ht, wd)
            ref_xy *= rescale
            ref_xy_slots = eo.repeat(ref_xy, "h w c -> (h w n) c", n=self.num_slots)

            pred = self.decoders[pyr_idx](feats)
            pred = eo.rearrange(pred, "b (n c) h w -> b (h w n) c", n=self.num_slots)

            (
                init_betas,
                init_pose,
                init_rot,
                init_xy,
                init_z,
                _,
                conf,
            ) = pred.split_with_sizes(dim=-1, split_sizes=self.output_split)

            # Adjust translation offset based on camera intrinsics
            default_depth = calib_adj / (self.depth_adj * 2**pyr_idx)
            init_z = default_depth / (torch.exp(init_z) + 0.05)
            init_xy = init_xy + ref_xy_slots
            init_xy = helper.px_to_world(K.unsqueeze(1), init_xy) * init_z
            init_trans = torch.cat([init_xy, init_z], -1)

            pred_state.append(
                {
                    "betas": init_betas,
                    "delta_root_orient": init_rot,
                    "pose_embedding": init_pose,
                    "trans": init_trans,
                    "conf": conf,
                }
            )

        detections = {}
        for k in pred_state[0].keys():
            detections[k] = torch.cat([p[k] for p in pred_state], 1).float()

        if return_feats:
            return detections, feat_pyramid
        else:
            return detections


class CoMotionDetect(nn.Module):
    """CoMotion detection module.

    Module responsible for initial feature extraction from a ConvNext backbone
    as well as producing candidate per-frame detections.
    """

    def __init__(
        self, cfg: CoMotionDetectConfig | None = None, pretrained: bool = True
    ):
        """Initialize CoMotion detection module.

        Args:
        ----
            cfg: Detection config defining various model hyperparameters.
            pretrained: Whether to load pretrained detection checkpoint.

        """
        super().__init__()
        cfg = CoMotionDetectConfig() if cfg is None else cfg

        self.cfg = cfg
        self.pos_embedding = layers.PosEmbed()
        self.rot_embedding = layers.RotaryEmbed(cfg.rot_embed_dim)
        self.kn = smpl_kinematics.SMPLKinematics()
        self.body_keys = ["betas", "pose", "trans"]

        # Instantiate an image backbone network
        self.image_backbone = backbones.initialize(cfg.backbone_choice)
        self.feat_dim = self.image_backbone.output_dim

        # Detection head
        # Output split: betas, pose embedding, root_orient, xy, z, scale, confidence
        output_split = [smpl_kinematics.BETA_DOF, cfg.pose_embed_dim, 3, 2, 1, 1, 1]
        self.detection_head = DetectionHead(
            input_dim=self.feat_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_split=output_split,
        )

        self.pose_decoder = nn.Sequential(
            nn.LayerNorm(cfg.pose_embed_dim),
            nn.Linear(cfg.pose_embed_dim, cfg.hidden_dim),
            layers.ResidualMLP(cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, smpl_kinematics.POSE_DOF - 3),
        )

        self.fuse_features = layers.FusePyramid(
            self.feat_dim,
            2 * cfg.hidden_dim,
        )

        if pretrained:
            checkpoint = torch.load(PYTORCH_CHECKPOINT_PATH, weights_only=True)
            self.load_state_dict(checkpoint)

    def _intrinsics_conditioning(self, feats, K, pooling=8, normalize_factor=1024):
        """Condition image features on pixel and world coordinate mapping.

        From input intrinsics matrix we define a coordinate grid and use rotary
        embeddings to update the extracted image features.
        """
        batch_size = feats.shape[0]
        feats = feats.clone()
        device = feats.device

        with torch.no_grad():
            # Get reference pixel positions
            ht, wd = feats.shape[-2:]
            ref_xy = helper.get_grid(ht, wd, device=device) + 0.5 / max(ht, wd)
            ref_xy *= max(ht, wd) * pooling

            # Reduce scale of pixel values
            K = K / normalize_factor
            ref_xy = ref_xy / normalize_factor

            # Adjust into world coordinate frame
            ref_xy_world = helper.px_to_world(K[:, None, None], ref_xy[None])

            # Get rotary embeddings
            xy_cs, xy_sn = self.rot_embedding(ref_xy)
            xy_world_cs, xy_world_sn = self.rot_embedding(ref_xy_world)

            # Rearrange from BHWC -> BCHW
            xy_cs = eo.repeat(xy_cs, f"h w d0 d1 -> {batch_size} (d0 d1) h w")
            xy_sn = eo.repeat(xy_sn, f"h w d0 d1 -> {batch_size} (d0 d1) h w")
            xy_world_cs = eo.rearrange(xy_world_cs, "b h w d0 d1 -> b (d0 d1) h w")
            xy_world_sn = eo.rearrange(xy_world_sn, "b h w d0 d1 -> b (d0 d1) h w")

        # Apply rotary embeddings
        cs = torch.cat([xy_cs, xy_world_cs, xy_cs, xy_world_cs], 1)
        sn = torch.cat([xy_sn, xy_world_sn, xy_sn, xy_world_sn], 1)
        embed_dim = sn.shape[1]
        f0 = feats[:, :embed_dim]
        f0 = layers.apply_rotary_pos_emb(f0, cs, sn)
        feats = torch.cat([f0, feats[:, embed_dim:]], 1)

        return feats

    @torch.inference_mode
    def forward(self, img, K) -> DetectionOutput:
        """Extract backbone features and detect poses.

        Args:
        ----
            img: input image tensor of shape (B, 3, 512, 512)
            K: input intrinsic tensor of shape (B, 2, 3)

        Return:
        ------
            DetectionOutput: NamedTuple that includes all output detection parameters.

        """
        outputs = {}

        # Get backbone features
        feats = self.image_backbone(img)
        feats = self._intrinsics_conditioning(feats, K)

        # Get detections
        detections, feature_pyramid = self.detection_head(feats, K, return_feats=True)
        for k, v in detections.items():
            if k == "pose_embedding":
                # Remap latent pose embedding to joint angles
                # Note: these are residual terms applied to a default pose
                outputs["delta_body_pose"] = self.pose_decoder(v) * 0.3
            else:
                outputs[k] = v

        # Fuse feature pyramid
        feature_pyramid = [feats] + feature_pyramid
        outputs["image_features"] = self.fuse_features(*feature_pyramid)

        return DetectionOutput(**outputs)


class CoMotionDetectCoreML:
    """A CoreML wrapper for CoMotion detection module."""

    def __init__(self):
        """Initialize the CoreML model."""
        import coremltools as ct

        self.model = ct.models.MLModel(COREML_CHECKPOINT_PATH)
        self.cfg = CoMotionDetectConfig()
        self.feat_dim = 256

    def __call__(self, img, K) -> DetectionOutput:
        """Run inference for the CoreML model."""
        outputs = self.model.predict({"image": img, "K": K})
        outputs = {k: torch.tensor(v) for k, v in outputs.items()}
        return DetectionOutput(**outputs)


def get_smpl_pose(delta_root_orient, delta_body_pose):
    """Apply predicted delta poses to default mean pose."""
    device = delta_root_orient.device
    default_pose = smpl_kinematics.extra_ref["mean_pose"].clone().to(device)
    delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
    return smpl_kinematics.update_pose(default_pose, delta_pose)


def decode_network_outputs(
    K: torch.Tensor,
    smpl_decoder: smpl_kinematics.SMPLKinematics,
    detect_out: DetectionOutput,
    sample_idx: int = 0,
    **nms_kwargs,
):
    """Postprocessing to get detections from network output."""
    # Decode output SMPL pose and joint coordinates
    pose = get_smpl_pose(detect_out.delta_root_orient, detect_out.delta_body_pose)

    pred_3d = smpl_decoder.forward(
        detect_out.betas,
        pose,
        detect_out.trans,
        output_format="joints_face",
    )
    pred_2d = helper.project_to_2d(K, pred_3d)

    # Perform non-maximum suppression
    nms_idxs = helper.nms_detections(
        pred_2d[sample_idx] / 1024, detect_out.conf[sample_idx].flatten(), **nms_kwargs
    )

    detections = {
        "betas": detect_out.betas,  # (1, 1344, 10)
        "pose": pose,  # (1, 1344, 72)
        "trans": detect_out.trans,  # (1, 1344, 3)
        "pred_3d": pred_3d,  # (1, 1344, 27, 3)
        "pred_2d": pred_2d,  # (1, 1344, 27, 2)
        "conf": detect_out.conf,  # (1, 1344, 1)
    }

    # Index into selected subset of detections, add singleton batch dimension
    for k, v in detections.items():
        detections[k] = v[sample_idx, nms_idxs][None]

    return detections
