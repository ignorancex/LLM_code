# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import os
import pickle

import chumpy
import einops as eo
import numpy as np
import pypose
import torch
from torch import nn

BETA_DOF = 10
POSE_DOF = 72
TRANS_DOF = 3
FACE_VERTEX_IDXS = [332, 6260, 2800, 4071, 583]
NUM_PARTS_AUX = 27

smpl_dir = os.path.join(os.path.dirname(__file__), "../data/smpl")
extra_ref = torch.load(f"{smpl_dir}/extra_smpl_reference.pt", weights_only=True)
smpl_model_path = f"{smpl_dir}/SMPL_NEUTRAL.pkl"
assert os.path.exists(smpl_model_path), (
    "Please download the neutral SMPL body model from https://smpl.is.tue.mpg.de/ and"
    "rename it to SMPL_NEUTRAL.pkl, copying it into src/comotion_demo/data/smpl/"
)


def to_rotmat(theta):
    """Convert axis-angle to rotation matrix."""
    rotmat = pypose.so3(theta.view(-1, 3)).matrix()
    return rotmat.reshape(*theta.shape, 3)


@torch.compiler.disable
def update_pose(pose, delta_pose):
    """Apply residual to an existing set of joint angles.

    Instead of adding the update to the current pose directly, we apply the
    update as a rotation.
    """
    # Convert flattened pose to K x 3 set of axis-angle terms
    pose = eo.rearrange(pose, "... (k c) -> ... k c", c=3).contiguous()
    delta_pose = eo.rearrange(delta_pose, "... (k c) -> ... k c", c=3).contiguous()

    # Change to a rotation matrix and multiply the matrices together
    pose: pypose.SO3_type = pypose.so3(delta_pose).Exp() * pypose.so3(pose).Exp()

    # Map back to axis-angle representation and flatten again
    pose = pose.Log().tensor()
    return eo.rearrange(pose, "... k c -> ... (k c)")


class SMPLKinematics(nn.Module):
    """Parse SMPL parameters to get mesh and joint coordinates."""

    def __init__(self):
        """Initialize SMPLKinematics."""
        super().__init__()

        self.num_tf = 24

        # Load SMPL neutral model
        with open(smpl_model_path, "rb") as f:
            smpl_neutral = pickle.load(f, encoding="latin1")

        # Prepare model data and convert to torch tensors
        smpl_keys = [
            "J_regressor",
            "v_template",
            "posedirs",
            "shapedirs",
            "weights",
            "kintree_table",
        ]
        for k in smpl_keys:
            v = smpl_neutral[k]
            if isinstance(v, chumpy.ch.Ch):
                v = np.array(v)
            elif not isinstance(v, np.ndarray):
                v = v.toarray()
            v = torch.tensor(v).float()

            if k == "shapedirs":
                v = v[..., :BETA_DOF].view(-1, BETA_DOF).contiguous()
            elif k == "posedirs":
                v = v.view(-1, v.shape[-1]).T
            elif k == "kintree_table":
                k = "parents"
                v = v[0].long()
                v[0] = -1

            self.register_buffer(k, v, persistent=False)

        # Precompute intermediate matrices
        self.register_buffer(
            "joint_shapedirs",
            (self.J_regressor @ self.shapedirs.view(-1, 3 * BETA_DOF)).view(
                self.num_tf * 3, BETA_DOF
            ),
            persistent=False,
        )
        self.register_buffer(
            "joint_template", self.J_regressor @ self.v_template, persistent=False
        )

        # Load additional reference tensors
        self.register_buffer("mean_pose", extra_ref["mean_pose"], persistent=False)
        self.register_buffer("coco_map", extra_ref["coco_map"], persistent=False)

    def forward_kinematics(self, rot_mat, unposed_tx):
        # Get relative joint positions
        relative_tx = unposed_tx.clone()
        relative_tx[1:] -= relative_tx[self.parents[1:]]

        # Define transformation matrices
        tf = torch.cat([rot_mat, relative_tx.unsqueeze(-1)], -1)
        tf = torch.functional.F.pad(tf, [0, 0, 0, 1])
        tf[..., 3, 3] = 1

        # Follow kinematic chain
        tf_chain = [tf[0]]
        for child_idx in range(1, self.num_tf):
            parent_idx = self.parents[child_idx]
            tf_chain.append(tf_chain[parent_idx] @ tf[child_idx])
        tf_chain = torch.stack(tf_chain)

        # Get output joint positions
        joints = eo.rearrange(tf_chain[..., :3, 3], "k ... c -> ... k c")
        return tf_chain, joints

    def get_smpl_template(self, subsample_rate=1, idx_subset=None):
        """Return reference tensors to derive SMPL mesh.

        Optional flags allow us to subsample a subset of vertices from the mesh to
        avoid computation over the complete mesh.
        """
        posedir_dim = self.posedirs.shape[0]
        shapedirs = eo.rearrange(self.shapedirs, "(n c) m -> n c m", c=3)

        if idx_subset is None:
            v_template = self.v_template[..., ::subsample_rate, :]
            posedirs = self.posedirs.view(posedir_dim, -1, 3)[
                :, ::subsample_rate
            ].reshape(posedir_dim, -1)
            shapedirs = eo.rearrange(shapedirs[::subsample_rate], "n c m -> (n c) m")
            weights = self.weights[::subsample_rate]
        else:
            v_template = self.v_template[..., idx_subset, :]
            posedirs = self.posedirs.view(posedir_dim, -1, 3)[:, idx_subset].reshape(
                posedir_dim, -1
            )
            shapedirs = eo.rearrange(shapedirs[idx_subset], "n c m -> (n c) m")
            weights = self.weights[idx_subset]

        return v_template, posedirs, shapedirs, weights

    def forward(
        self,
        betas,
        pose,
        trans=None,
        output_format="joints",
        subsample_rate=1,
        idx_subset=None,
    ):
        """Get joints and mesh vertices.

        Valid output formats:
        - joints: 24x3 set of SMPL joints
        - joints_face: 27x3 joints where the first 22 values are original SMPL joints
            while the last 5 correspond to face keypoints
        - joints_coco: 17x3 joints in COCO order that match COCO annotation format
            e.g. hips are higher and wider than in SMPL
        - mesh: 6890x3 set of SMPL mesh vertices
            if idx_subset is not None, returns len(idx_subset) vertices
            if subsample rate > 1, returns (6078 // subsample_rate) vertices
        """
        # Convert pose to rotation matrices
        pose = eo.rearrange(pose, "... (k c) -> k ... c", c=3).contiguous()
        rot_mat = to_rotmat(pose)

        # Adjust mesh based on betas
        blend_shape = eo.rearrange(
            self.joint_shapedirs @ betas.unsqueeze(-1), "... (k c) 1 -> ... k c", c=3
        )
        unposed_tx = self.joint_template + blend_shape
        unposed_tx = eo.rearrange(unposed_tx, "... k c -> k ... c")

        # Run forward kinematics
        tf_chain, joints = self.forward_kinematics(rot_mat, unposed_tx)

        if output_format == "joints":
            smpl_output = joints

        else:
            if output_format == "joints_face":
                idx_subset = FACE_VERTEX_IDXS
            elif output_format == "joints_coco":
                subsample_rate = 1
                idx_subset = None

            v_template, posedirs, shapedirs, weights = self.get_smpl_template(
                subsample_rate, idx_subset
            )

            # Adjust mesh based on betas
            blend_shape = eo.rearrange(
                shapedirs @ betas.unsqueeze(-1), "... (k c) 1 -> ... k c", c=3
            )
            v_shaped = v_template + blend_shape

            # Get relative transforms (rotate unposed joints and subtract)
            tf_offset = (tf_chain[..., :3, :3] @ unposed_tx.unsqueeze(-1)).squeeze(-1)
            tf_relative = tf_chain.clone()
            tf_relative[..., :3, 3] -= tf_offset
            A = eo.rearrange(tf_relative, "n ... d0 d1 -> ... n (d0 d1)")

            # Flatten all rotation matrices (except root rotation), remove identity
            eye = torch.eye(3, device=rot_mat.device)
            pose_feature = eo.rearrange(
                rot_mat[1:] - eye, "k ... d0 d1 -> ... (k d0 d1)"
            )

            # Adjust base vertex positions
            pose_offsets = eo.rearrange(
                pose_feature @ posedirs, "... (n c) -> ... n c", c=3
            )
            v_posed = v_shaped + pose_offsets

            # Transform vertices based on pose
            T = weights @ A
            T = eo.rearrange(T, "... (d0 d1) -> ... d0 d1", d0=4)
            vertices = torch.functional.F.pad(v_posed, [0, 1], value=1)
            vertices = (T @ vertices.unsqueeze(-1)).squeeze(-1)
            vertices = vertices[..., :-1] / vertices[..., -1:]

            smpl_output = vertices
            if output_format == "joints_face":
                smpl_output = torch.cat([joints[..., :22, :], vertices], -2)
            elif output_format == "joints_coco":
                smpl_output = eo.einsum(
                    vertices, self.coco_map, "... i j, i k -> ... k j"
                )

        if trans is not None:
            # Apply translation offset
            smpl_output = smpl_output + trans.unsqueeze(-2)

        return smpl_output
