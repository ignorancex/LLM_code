# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common 3D pose methods
"""

import torch
from jaxtyping import Float
from torch import Tensor


def to_homogeneous(input_tensor, dim=1):
    return torch.cat([input_tensor, torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))], dim=dim)

def torch_get_covis(c2w,hw_bounds,
              num_samples=1024,
                min_depth=0.1,
                max_depth=128,
                ):
    pose_covis=torch.zeros((len(c2w),len(c2w))).cuda()
    c2w=torch.tensor(c2w,dtype=torch.float32).cuda()
    w2c=torch.inverse(c2w)
    hw_bounds=torch.tensor(hw_bounds,dtype=torch.float32).cuda()
    # depth_samples= lognorm.rvs(0.7,2,16,size=num_samples)
    # depth_samples=torch.tensor(depth_samples,dtype=torch.float32).cuda()
    depth_samples=torch.rand(num_samples,dtype=torch.float32).cuda()*(max_depth-min_depth)+min_depth
    for i in range(len(c2w)):
        hw_bound=hw_bounds[i]

        h_samples=torch.rand(num_samples).cuda()*(hw_bound[0]*2)-hw_bound[0]
        w_samples=torch.rand(num_samples).cuda()*(hw_bound[1]*2)-hw_bound[1]

        src_coords=torch.stack([w_samples,h_samples,depth_samples],dim=1)

        # reproject to other images
        reproj_mat = w2c @ c2w[i]
        src_dir= reproj_mat[:,:3,:3] @ src_coords.T # M x 3 x N
        dst_coords = src_dir + reproj_mat[:,:3,3:4] # M x 3 x N
        dst_depths = dst_coords[:,2]
        dst_xy = dst_coords[:,:2,:]/dst_coords[:,2:]
        # cosine similarity
        score=torch.cosine_similarity(src_dir,dst_coords,dim=1)
        # score[score<0]=0

        # check inlier in -hw_bound to hw_bound for each image
        mask_depth=(dst_depths>min_depth) & (dst_depths<max_depth)
        mask=(dst_xy[:,0]>=-hw_bounds[:,1,None]) & (dst_xy[:,0]<hw_bounds[:,1,None]) & (dst_xy[:,1]>=-hw_bounds[:,0,None]) & (dst_xy[:,1]<hw_bounds[:,0,None]) & mask_depth
        mask=score*mask
        pose_covis[i]=mask.sum(dim=1)/num_samples

    return pose_covis.cpu().numpy()


def to4x4(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 4 4"]:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def multiply(pose_a: Float[Tensor, "*batch 3 4"], pose_b: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right.
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a.

    Returns:
        Camera pose matrix where pose_a was applied to pose_b.
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


def normalize(poses: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Normalize the XYZs of poses to fit within a unit cube ([-1, 1]). Note: This operation is not in-place.

    Args:
        poses: A collection of poses to be normalized.

    Returns;
        Normalized collection of poses.
    """
    pose_copy = torch.clone(poses)
    pose_copy[..., :3, 3] /= torch.max(torch.abs(poses[..., :3, 3]))

    return pose_copy
