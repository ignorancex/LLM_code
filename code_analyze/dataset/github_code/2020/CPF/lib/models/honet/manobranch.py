import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer, MANOOutput
from torch import nn


class ManoAdaptor(nn.Module):

    def __init__(self, mano_layer, load_path=None):
        super().__init__()
        self.adaptor = torch.nn.Linear(778, 21, bias=False)
        if load_path is not None:
            with open(load_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
                weights = exp_data["adaptor"]
            regressor = torch.Tensor(weights)
            self.register_buffer("J_regressor", regressor)
        else:
            regressor = mano_layer._buffers["th_J_regressor"]
            tip_reg = regressor.new_zeros(5, regressor.shape[1])
            tip_reg[0, 745] = 1
            tip_reg[1, 317] = 1
            tip_reg[2, 444] = 1
            tip_reg[3, 556] = 1
            tip_reg[4, 673] = 1
            reordered_reg = torch.cat([regressor, tip_reg
                                      ])[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
            self.register_buffer("J_regressor", reordered_reg)
        self.adaptor.weight.data = self.J_regressor

    def forward(self, inp):
        fix_idxs = [0, 4, 8, 12, 16, 20]
        for idx in fix_idxs:
            self.adaptor.weight.data[idx] = self.J_regressor[idx]
        return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


class ManoBranch(nn.Module):

    def __init__(
        self,
        ncomps,
        base_neurons,
        center_idx,
        use_pca=True,
        use_shape=True,
        mano_root="assets/mano_v1_2",
        mano_side="right",
        dropout=0,
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
            center_idx: Joint idx on which to hand is centered (given joint has position
                [0, 0, 0]
            ncomps: Number of pose principal components that are predicted
        """
        super(ManoBranch, self).__init__()

        self.use_shape = use_shape
        self.use_pca = use_pca
        self.mano_side = mano_side

        if self.use_pca:
            # Final number of coefficients to predict for pose
            # is sum of PCA components and 3 global axis-angle params
            # for the global rotation
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 components per joint
            # rotation
            mano_pose_size = 16 * 9
        # Initial base layers of MANO decoder
        base_layers = []
        for inp_neurons, out_neurons in zip(base_neurons[:-1], base_neurons[1:]):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers to predict pose parameters
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            # Initialize all nondiagonal items on rotation matrix weights to 0
            self.pose_reg.bias.data.fill_(0)
            weight_mask = self.pose_reg.weight.data.new(np.identity(3)).view(9).repeat(16)
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, 256).float() * self.pose_reg.weight.data)

        # Shape layers to predict MANO shape parameters
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(base_neurons[-1], 10))

        # Mano layer which outputs the hand mesh given the hand pose and shape
        # paramters
        self.mano_layer = ManoLayer(ncomps=ncomps,
                                    center_idx=center_idx,
                                    side=mano_side,
                                    mano_assets_root=mano_root,
                                    use_pca=use_pca,
                                    flat_hand_mean=False)
        self.faces = self.mano_layer.th_faces

    def forward(self, inp):
        base_features = self.base_layer(inp)
        pose = self.pose_reg(base_features)  # TENSOR (B, N_PCA)

        if not self.use_pca:
            # Reshape to rotation matrixes
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)
        else:
            mano_pose = pose

        # Get shape
        if self.use_shape:
            shape = self.shape_reg(base_features)
        else:
            shape = None

        # Get MANO vertices and joints for left and right hands given
        # predicted mano parameters
        manoout: MANOOutput = self.mano_layer(mano_pose, betas=shape)

        # Gather results in metric space (vs MANO millimeter outputs)
        # pose: the 18 ncomps (3 global rot + 15 pca hand pose)
        # full_pose: the 48 (16 * 3) full relative axis-angles of all 16 joints rotations (from root to finger)
        results = {
            "shape": shape,
            "pose": pose,
            "verts3d": manoout.verts,
            "joints3d": manoout.joints,
            "full_pose": manoout.full_poses
        }

        return results
