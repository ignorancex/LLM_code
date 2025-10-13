# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
from collections import namedtuple
from dataclasses import dataclass

import einops as eo
import torch
from torch import nn

from ..utils import helper, smpl_kinematics
from . import layers

curr_dir = os.path.abspath(os.path.dirname(__file__))
PYTORCH_CHECKPOINT_PATH = f"{curr_dir}/../data/comotion_refine_checkpoint.pt"

RefineOutput = namedtuple(
    "RefineOutput",
    [
        "delta_root_orient",
        "delta_body_pose",
        "trans",
        "hidden",
    ],
)


@dataclass
class PoseRefinementConfig:
    num_tokens: int = 24
    num_heads: int = 4
    token_dim: int = 256
    hidden_dim: int = 512
    pose_embed_dim: int = 256
    normalizing_factor: int = 1024


class PoseRefinement(nn.Module):
    """CoMotion refinement module."""

    def __init__(
        self,
        image_feat_dim: int,
        pose_embed_dim: int,
        cfg: PoseRefinementConfig | None = None,
        pretrained: bool = True,
    ):
        """Initialize refinement module.

        Args:
        ----
            image_feat_dim: image backbone output feature dimension.
            pose_embed_dim: pose embedding dimension.
            cfg: pose refinement config.
            pretrained: whether to load pre-trained weights.

        """
        super().__init__()
        cfg = PoseRefinementConfig() if cfg is None else cfg
        self.cfg = cfg
        self.image_feat_dim = image_feat_dim
        self.pose_embed_dim = pose_embed_dim
        self.pos_embedding = layers.PosEmbed()
        self.encode_grid = nn.Linear(self.pos_embedding.out_dim, cfg.token_dim)
        token_kv_in, token_kv_out = image_feat_dim + cfg.token_dim, cfg.token_dim

        self.get_px_key = nn.Linear(token_kv_in, token_kv_out)
        self.get_px_value = nn.Linear(token_kv_in, token_kv_out)

        # Output reference
        self.split_ref = [
            smpl_kinematics.BETA_DOF,
            pose_embed_dim,
            3,
            smpl_kinematics.TRANS_DOF,
        ]

        # Hidden update layers
        self.tokens_to_hidden = layers.DecodeFromTokens(cfg.hidden_dim)
        self.feedback_gru_update = layers.GRU(cfg.hidden_dim, cfg.hidden_dim)

        def init_token_encoder(in_dim):
            return layers.ResidualMLP(
                in_dim,
                out_dim=cfg.num_tokens * cfg.hidden_dim,
                hidden_dim=2 * cfg.hidden_dim,
                pre_ln=False,
            )

        # Convert coordinates and hidden state to tokens
        cd_dim = smpl_kinematics.NUM_PARTS_AUX * (self.pos_embedding.out_dim + 1)
        dof = (
            smpl_kinematics.BETA_DOF
            + smpl_kinematics.POSE_DOF
            + smpl_kinematics.TRANS_DOF
        )

        self.encode_coords = init_token_encoder(cd_dim)
        self.encode_hidden = init_token_encoder(cfg.hidden_dim)
        self.encode_smpl = init_token_encoder(dof)

        cross_attention_kargs = {
            "num_tokens": cfg.num_tokens,
            "num_heads": cfg.num_heads,
            "hidden_dim": cfg.hidden_dim,
            "token_dim": cfg.token_dim,
        }

        self.cross_attention = nn.Sequential(
            layers.CrossAttention(**cross_attention_kargs),
            layers.CrossAttention(**cross_attention_kargs),
        )

        decode = layers.DecodeFromTokens

        self.get_px_feedback = decode(cfg.hidden_dim)
        self.tokens_to_hidden = nn.Sequential(
            decode(cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.project_hidden = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.get_pose_update = nn.Sequential(
            layers.ResidualMLP(cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, sum(self.split_ref)),
        )

        # This is the same module used in the detection step (with identical weights)
        # We reinstantiate it to support using the separate CoreML detection stage
        self.pose_decoder = nn.Sequential(
            nn.LayerNorm(cfg.pose_embed_dim),
            nn.Linear(cfg.pose_embed_dim, cfg.hidden_dim),
            layers.ResidualMLP(cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, smpl_kinematics.POSE_DOF - 3),
        )

        if pretrained:
            checkpoint = torch.load(PYTORCH_CHECKPOINT_PATH, weights_only=True)
            self.load_state_dict(checkpoint)

    @helper.fixed_dim_op(d=4, is_class_fn=True)
    def compute_image_kv(self, image_feats, pooling=8):
        """From image features, get flattened set of key, value token pairs."""
        batch_size = image_feats.shape[0]
        res = image_feats.shape[-2:]
        device = image_feats.device

        # Rearrange and flatten image features
        image_feats = eo.rearrange(image_feats, "b c h w -> b (h w) c")

        # Calculate reference grid of pixel positions
        with torch.no_grad():
            px_scale_factor = max(res) * pooling
            grid = helper.get_grid(res[0], res[1], device)
            grid = grid * px_scale_factor / self.cfg.normalizing_factor

        grid_embed = self.encode_grid(self.pos_embedding(grid))
        grid_embed = eo.repeat(grid_embed, "h w d -> b (h w) d", b=batch_size)
        image_feats = torch.cat([image_feats, grid_embed], -1)

        # Apply a linear layer to get keys and values
        pixel_k = eo.rearrange(
            self.get_px_key(image_feats),
            "... d0 (h d1) -> ... h d0 d1",
            h=self.cfg.num_heads,
        ).contiguous()
        pixel_v = eo.rearrange(
            self.get_px_value(image_feats),
            "... d0 (h d1) -> ... h d0 d1",
            h=self.cfg.num_heads,
        ).contiguous()

        return pixel_k, pixel_v

    def calib_adjusted_trans(self, K, trans, delta_trans, depth_adj=128, eps=0.05):
        """Update SMPL translation term based on provided intrinsics.

        This same operation is performed during detection, the output x, y
        terms are in pixel space and mapped to world coordinates using K.
        """
        delta_xy, delta_z = delta_trans.split_with_sizes(dim=-1, split_sizes=[2, 1])

        # Get new depth estimate
        default_depth = K[..., 0, 0][:, None, None] / depth_adj
        z = default_depth / (torch.exp(delta_z) + eps)

        # Apply delta to current x, y position in pixel space
        base_xy = helper.project_to_2d(K.unsqueeze(1), trans)
        xy = helper.px_to_world(K.unsqueeze(1), base_xy + delta_xy) * z
        return torch.cat([xy, z], -1)

    def encode_state(
        self,
        K,
        hidden,
        pred_3d,
        body_params,
    ):
        """Encode tracks into tokens.

        Hidden state, SMPL parameters, and 2D keypoint coordinates are all
        passed through an MLP to produce a set of tokens per-person.
        """
        # Project to 2D
        K = K[:, None, None] / self.cfg.normalizing_factor
        xy = helper.project_to_2d(K, pred_3d).clamp(-5, 5)
        z = pred_3d[..., 2]
        cd_embed = eo.rearrange(self.pos_embedding(xy), "... d0 d1 -> ... (d0 d1)")
        cd_embed = torch.cat([cd_embed, z], -1)

        # Encode tokens
        tokens = self.encode_coords(cd_embed)
        tokens = tokens + self.encode_hidden(hidden)
        tokens = tokens + self.encode_smpl(body_params)
        tokens = eo.rearrange(
            tokens, "... (d0 d1) -> ... d0 d1", d0=self.cfg.num_tokens
        )

        return tokens

    def perform_update(self, K, pixel_k, pixel_v, tokens, trans, hidden):
        """Attend to image features and calculate final outputs.

        Args:
        ----
            K: Input intrinsics
            pixel_k: Flattened set of image token keys.
            pixel_v: Flattened set of image token values.
            tokens: Feature encoding of current tracks.
            trans: SMPL translation term for each track.
            hidden: Hidden state for each track.

        """
        # Perform cross attention to get feedback from image features
        for ca in self.cross_attention:
            tokens = ca(pixel_k, pixel_v, tokens)

        # Update hidden state
        hidden_update = self.tokens_to_hidden(tokens)
        hidden = self.feedback_gru_update(hidden, hidden_update)

        # Update current state
        px_feedback = self.get_px_feedback(tokens)
        px_feedback = px_feedback + self.project_hidden(hidden)

        delta_smpl = self.get_pose_update(px_feedback)
        delta_smpl = delta_smpl.split_with_sizes(dim=-1, split_sizes=self.split_ref)
        _, pose_embedding, delta_root_orient, delta_trans = delta_smpl

        delta_body_pose = self.pose_decoder(pose_embedding) * 0.3
        trans = self.calib_adjusted_trans(K, trans, delta_trans)
        return RefineOutput(delta_root_orient, delta_body_pose, trans, hidden)

    def forward(self, image_feats, K, betas, pose, trans, pred_3d, hidden, pooling=8):
        """Predict new poses given image features and current tracks.

        Args:
        ----
            image_feats: Per-pixel image features.
            K: Input intrinsics.
            betas: SMPL beta parameters for each track.
            pose: SMPL pose parameters for each track.
            trans: SMPL translation term for each track.
            pred_3d: 3D keypoints in camera coordinate frame for each track.
            hidden: Hidden state for each track.
            pooling: Indicator of how image features have been pooled from input image.

        """
        body_params = torch.cat([betas, pose, trans], -1)
        pixel_k, pixel_v = self.compute_image_kv(image_feats, pooling=pooling)
        tokens = self.encode_state(K, hidden, pred_3d, body_params)
        return self.perform_update(K, pixel_k, pixel_v, tokens, trans, hidden)
