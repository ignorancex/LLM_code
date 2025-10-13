from typing import Any

import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from offsetocc.registry import MODELS


@MODELS.register_module()
class ObjDecoder3D(TransformerLayerSequence):

    """
    Attention with both self and cross
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        reference_system (str): The reference system of the reference points.
            Options are 'ego' and 'lidar'.
        pc_range (list[float]): The range of point cloud.
    """

    def __init__(self,
                 return_intermediate: bool = False,
                 **kwargs: Any) -> None:

        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_pos: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                **kwargs: Any) -> torch.Tensor:
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input occ query with shape
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

        output = query

        intermediate = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                query,
                key,
                value,
                query_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs)

            query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
