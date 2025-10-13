# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
import warnings
from typing import Optional, no_type_check

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning


def volumetric_deformable_attn_pytorch(value: torch.Tensor,
                                       value_spatial_shapes: torch.Tensor,
                                       sampling_locations: torch.Tensor,
                                       attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape (bs, num_levels, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of each feature map, has shape (num_levels, 3),
            last dimension 3 represent (L, W, H) ( or (X, Y, Z) )
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape (bs, num_queries, num_heads, num_levels, num_points, 3), the last dimension 3 represent (x, y, z).
        attention_weights (torch.Tensor): The weight of sampling points used when calculate the attention,
            has shape (bs, num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([L_ * W_ * H_ for L_, W_, H_ in value_spatial_shapes], dim=1)
    # Rescale sampling locations from [0 -- 1] to [-1 -- 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (L_, W_, H_) in enumerate(value_spatial_shapes):
        # bs, X_*Y_*Z_, num_heads, embed_dims ->
        # bs, X_*Y_*Z_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, X_*Y_*Z_ ->
        # bs*num_heads, embed_dims, X_, Y_, Z_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, L_, W_, H_)
        # bs, num_queries, num_heads, num_points, 3 ->
        # bs, num_heads, num_queries, num_points, 3 ->
        # bs*num_heads, _, num_queries, num_points, 3
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)[:, None, :, :, :]

        assert len(value_l_.shape) == 5, \
            f"Input must have 5 dimensions: [N, C, X_in, Y_in, Z_in], not {value_l_.shape}"
        assert len(sampling_grid_l_.shape) == 5, \
            f"Grid must have 5 dimensions: [N, X_out, Y_out, Z_out, 3], not {sampling_grid_l_.shape}"
        # NOTE: X_out, Y_out, Z_out represent a new 3D output, but in theory, these could be any dimensions, reshaped into 3. 
        assert sampling_grid_l_.shape[-1] == 3, \
            f"The last dimension of grid must be of size 3, not {sampling_grid_l_.shape[-1]}"

        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False).squeeze(2)
        # TODO: make sure the sampling process is correct (check pt documentation) (If True -> [-1,-1,-1]
        #  points to the center of pixel [0,0,0])
        # Output (N, C, H_out, W_out, Z_out) 
        # -> (bs*num_heads, embed_dims, _, num_queries, num_points) 
        # -> (bs*num_heads, embed_dims, num_queries, num_points)
        sampling_value_list.append(sampling_value_l_)

    # TODO: Jonas: haven't checked this attention yet
    
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = (attention_weights.transpose(1, 2).
                         reshape(bs * num_heads, 1, num_queries, num_levels * num_points))
    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).
              sum(-1).view(bs, num_heads * embed_dims, num_queries))
    return output.transpose(1, 2).contiguous()


@MODELS.register_module()
class VolumetricDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 3)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device

        def distribute_points_on_sphere(N, dtype=torch.float32, device=device):
            goldenRatio = (1 + 5 ** 0.5) / 2
            i = torch.arange(0, N, dtype=dtype, device=device)
            theta = 2 * torch.pi * i / goldenRatio
            phi = torch.acos(1 - 2 * (i + 0.5) / N)
            return theta, phi

        thetas, phis = distribute_points_on_sphere(self.num_heads, dtype=torch.float32, device=device)
        grid_init = torch.stack([thetas.cos() * phis.sin(), thetas.sin() * phis.sin(), phis.cos()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 3).repeat(1,
                                                                                                           self.num_levels,
                                                                                                           self.num_points,
                                                                                                           1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points_3d: Optional[torch.Tensor] = None,
                spatial_shapes_3d: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points_3d (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes_3d (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes_3d[:, 0] * spatial_shapes_3d[:, 1] * spatial_shapes_3d[:, 2]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points_3d.shape[-1] == 3:
            offset_normalizer = torch.stack(
                [spatial_shapes_3d[..., 0], spatial_shapes_3d[..., 1], spatial_shapes_3d[..., 2]], -1)
            sampling_locations = reference_points_3d[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 3, but get {reference_points_3d.shape[-1]} instead.')

        output = volumetric_deformable_attn_pytorch(value, spatial_shapes_3d, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
