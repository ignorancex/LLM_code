"""Transformer decoder definition.
Adapted from
    https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py
"""

from typing import Any

import numpy as np
import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from offsetocc.registry import MODELS


@MODELS.register_module()
class OffsetOccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        reference_system (str): The reference system of the reference points.
            Options are 'ego' and 'lidar'.
        pc_range (list[float]): The range of point cloud.
    """

    def __init__(self,
                 pc_range: list[float] = None,
                 return_intermediate: bool = False,
                 reference_system: str = 'lidar',
                 **kwargs: Any) -> None:

        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate
        assert reference_system is 'ego' or 'lidar', 'reference system should be either ego or lidar'
        self.reference_system = reference_system
        self.pc_range = pc_range

    @staticmethod
    @torch.cuda.amp.autocast(enabled=False)
    def get_reference_points(L: int, W: int, H: int, bs: int = 1, device: str | torch.device = 'cuda') -> torch.Tensor:

        """
        Get the reference points of the occupancy grid.

        Args:
            L (int): The length of the occupancy grid.
            W (int): The width of the occupancy grid.
            H (int): The height of the occupancy grid.
            bs (int): The batch size.
            device (str | torch.device): The device of the reference points.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        xs = torch.linspace(0.5, L - 0.5, L, dtype=torch.float32,
                            device=device).view(L, 1, 1).expand(L, W, H) / L
        ys = torch.linspace(0.5, W - 0.5, W, dtype=torch.float32,
                            device=device).view(1, W, 1).expand(L, W, H) / W
        zs = torch.linspace(0.5, H - 0.5, H, dtype=torch.float32,
                            device=device).view(1, 1, H).expand(L, W, H) / H
        ref_3d = torch.stack((xs, ys, zs), 3)
        ref_3d = ref_3d.flatten(0, 2)
        ref_3d = ref_3d[None].repeat(bs, 1, 1)  # (bs, num_queries, 3)

        return ref_3d


    @torch.cuda.amp.autocast(enabled=False)
    def point_sampling(self, reference_points: torch.Tensor, pc_range: list[float],
                       batch_input_metas: list[dict]) -> torch.Tensor:

        reference_points = reference_points.float()

        refsyst2img = []
        for meta in batch_input_metas:
                lidar2img = np.array(meta['lidar2img'])
                if self.reference_system == 'ego':
                    lidar2ego = np.array(meta['lidar_points']['lidar2ego'])
                    refsyst2img.append(lidar2img @ np.linalg.inv(lidar2ego))
                elif self.reference_system == 'lidar':
                    refsyst2img.append(lidar2img)
        refsyst2img = np.asarray(refsyst2img)
        refsyst2img = reference_points.new_tensor(refsyst2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), 2)

        B, num_query, _ = reference_points.size()
        num_cam = refsyst2img.size(1)

        reference_points = reference_points[:, None, :, :].repeat(1, num_cam, 1, 1)     # (B, num_cam, num_query, 4)
        reference_points = reference_points.unsqueeze(-1)                               # (B, num_cam, num_query, 4, 1)
        refsyst2img = refsyst2img[:, :, None, :, :].repeat(1, 1, num_query, 1, 1)       # (B, num_cam, num_query, 4, 4)

        # (B, num_cam, num_query, 4, 1) @ (B, num_cam, num_query, 4, 4) -> (B, num_cam, num_query, 4)
        reference_points_cam = torch.matmul(refsyst2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)

        eps = 1e-5

        occ_proj_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # TODO: following code handles batch elements with different image size but not different img sizes in the same
        #   batch element (i.e. different camera sizes mounted on the car)

        # normalize according to the original image size
        img_shape = torch.stack([torch.tensor(meta['img_shape'], device=reference_points_cam.device)
                                 for meta in batch_input_metas], dim=0)
        reference_points_cam[..., 0] /= img_shape[:, 1, None, None]
        reference_points_cam[..., 1] /= img_shape[:, 0, None, None]

        occ_proj_mask = (occ_proj_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        occ_proj_mask = occ_proj_mask.new_tensor(
            np.nan_to_num(occ_proj_mask.cpu().numpy()))

        # rescale reference_points to the size of the padded image
        input_shape = torch.stack([torch.tensor(meta['input_shape'], device=reference_points_cam.device)
                                   for meta in batch_input_metas], dim=0)
        reference_points_cam[..., 0] *= (img_shape[:, 1] / input_shape[:, 1])[:, None, None]
        reference_points_cam[..., 1] *= (img_shape[:, 0] / input_shape[:, 0])[:, None, None]

        occ_proj_mask = occ_proj_mask.squeeze(-1)

        return reference_points_cam, occ_proj_mask

    def forward(self,
                occ_query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_pos: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                **kwargs: Any) -> torch.Tensor:
        """Forward function for `TransformerDecoder`.
        Args:
            occ_query (Tensor): Input occ query with shape
                `(num_query, bs, embed_dims)`.
            key (Tensor): Input multi-camera features with shape
                (num_cam, num_value, bs, embed_dims)
            value (Tensor): Input multi-camera features with shape
                (num_cam, num_value, bs, embed_dims)
            query_pos (Tensor): Occupancy position with shape (bs, num_query, feat_dims).
            spatial_shapes (Tensor): Spatial shapes of multi-level features
                with shape (num_levels, 2).
            level_start_index (Tensor): Start index of each level in the
                concatenated features with shape (num_levels, ).
        Returns:
            Tensor: Results with shape (bs, num_query, embed_dims).
        """

        occ_l = kwargs['occ_l']
        occ_w = kwargs['occ_w']
        occ_h = kwargs['occ_h']

        bs = occ_query.size(0)

        output = occ_query
        intermediate = []

        # (bs, num_queries, 3)
        ref_3d = self.get_reference_points(occ_l, occ_w, occ_h, bs=bs, device=occ_query.device)
        # (bs, num_cams, num_queries, 2), (bs, num_cams, num_queries)
        reference_points_cam, occ_proj_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['batch_input_metas'])

        spatial_shapes_3d = torch.tensor([[occ_l, occ_w, occ_h]], device=occ_query.device)

        for lid, layer in enumerate(self.layers):
            output = layer(
                occ_query,
                key,
                value,
                query_pos=query_pos,
                reference_points_3d=ref_3d.unsqueeze(2),  # unsqueeze to create 1 "feat" level dimension,
                reference_points_cam=reference_points_cam,
                spatial_shapes=spatial_shapes,
                spatial_shapes_3d=spatial_shapes_3d,
                level_start_index=level_start_index,
                occ_proj_mask=occ_proj_mask,
                **kwargs)

            occ_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
