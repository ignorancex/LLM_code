from typing import List, Any, Tuple

import math
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.model import BaseModule
from torch.nn.init import normal_

from offsetocc.models.layers import MultiScaleDeformableAttention2D
from offsetocc.models.layers import VolumetricDeformableAttention
from offsetocc.registry import MODELS


@MODELS.register_module()
class PerceptionTransformer3DObjDecoder(BaseModule):
    """PerceptionTransformer.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        num_cams (int): Number of cameras: Default: 6.
        scale_factors (List[float]): Scale factors for the occupancy grid:
        decoder (nn.Module): Encoder module.
        obj_decoder (nn.Module): Object decoder module.
        embed_dims (int): The embedding dimension. Default: 128.
        occ_l (int): The length of the occupancy grid. Default: 200.
        occ_w (int): The width of the occupancy grid. Default: 200.
        occ_h (int): The height of the occupancy grid. Default: 16.
        use_cams_embeds (bool): Whether to use camera embeddings.
    """

    def __init__(
            self,
            num_feature_levels: int = 4,
            num_cams: int = 6,
            scale_factors: Tuple[float] = (2.0, 2.0, 2.0),
            decoder: nn.Module = None,
            obj_decoder: nn.Module = None,
            num_objqueries: int = 100,
            embed_dims: int = 128,
            occ_l: int = 200,
            occ_w: int = 200,
            occ_h: int = 16,
            use_cams_embeds: bool = True,
            **kwargs: Any) -> None:

        super().__init__(**kwargs)

        # Model shapes
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.occ_l = int(occ_l * 1/scale_factors[0])
        self.occ_w = int(occ_w * 1/scale_factors[1])
        self.occ_h = int(occ_h * 1/scale_factors[2])

        self.num_objqueries = num_objqueries

        self.use_cams_embeds = use_cams_embeds

        # Model construction
        self.decoder = build_transformer_layer_sequence(decoder)
        self.obj_decoder = build_transformer_layer_sequence(obj_decoder)
        self.init_layers()

    def init_layers(self) -> None:
        # self.occ_embedding = nn.Embedding(
        #     self.occ_l * self.occ_w * self.occ_h, self.embed_dims)
        self.occ_embedding = torch.zeros(
            self.occ_l * self.occ_w * self.occ_h, self.embed_dims)
        # self.occ_pos_encoding = nn.Embedding(
        #     self.occ_l * self.occ_w * self.occ_h, self.embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.obj_embedding = nn.Embedding(
            self.num_objqueries, self.embed_dims * 2)
        self.reference_points_fc = nn.Linear(self.embed_dims, 3)        # TODO: why not learning 3d priors directly?

    def init_weights(self) -> None:
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.obj_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (isinstance(m, MultiScaleDeformableAttention2D)
                or isinstance(m, VolumetricDeformableAttention)
            ):
                m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def get_positional_encoding(self, embed_dims: int, max_len: int = 5000,
                                device: str | torch.device = 'cuda') -> torch.Tensor:

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dims, 2, device=device) * (-math.log(10000.0) / embed_dims))
        pe = torch.zeros(max_len, embed_dims, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_occ_features(
            self,
            mlvl_feats: List[torch.Tensor],
            occ_queries: torch.Tensor,
            occ_l: int,
            occ_w: int,
            occ_h: int,
            occ_pos: torch.Tensor = None,
            **kwargs) -> torch.Tensor:
        """
        obtain occ features.
        """

        bs = mlvl_feats[0].size(0)
        occ_queries = occ_queries.unsqueeze(0).repeat(bs, 1, 1)
        if occ_pos is not None:
            occ_pos = occ_pos.unsqueeze(0).repeat(bs, 1, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(1, 0, 2, 3)  # (bs, num_cam, num_feat, embed_dims)

        occ_embed = self.decoder(
            occ_queries,
            feat_flatten,
            feat_flatten,
            occ_l=occ_l,
            occ_w=occ_w,
            occ_h=occ_h,
            query_pos=occ_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )

        return occ_embed

    def pre_3dobjdecoder(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:

        batch_size, _, c = memory.shape

        obj_embed = self.obj_embedding.weight
        query_pos, query = torch.split(obj_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        ref_3d = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            key=memory,
            value=memory,
            reference_points_3d=ref_3d.unsqueeze(2),
            spatial_shapes_3d=torch.tensor([[self.occ_l, self.occ_w, self.occ_h]], device=memory.device),
        )

        return decoder_inputs_dict


    def forward(self, mlvl_feats: List[torch.Tensor], **kwargs) -> [torch.Tensor, torch.Tensor]:
        """Forward function of Transformer`.
        Args:
            mlvl_feats (List[Tensor]): Multi-level features from the backbone.
        Returns:
            Tensor: Features of the occupancy grid.
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        # dtype = mlvl_feats[0].dtype
        # occ_queries = self.occ_embedding.weight.to(dtype)
        # occ_pos = self.occ_pos_encoding.weight.to(dtype)
        # occ_pos = None
        device = mlvl_feats[0].device
        occ_queries = self.occ_embedding.to(device)
        occ_pos = self.get_positional_encoding(self.embed_dims, max_len=self.occ_l * self.occ_w * self.occ_h,
                                               device=device)

        occ_features = self.get_occ_features(
            mlvl_feats,
            occ_queries,
            self.occ_l,
            self.occ_w,
            self.occ_h,
            occ_pos=occ_pos,
            **kwargs)  # occ_embed shape: bs, occ_l*occ_w*occ_h, embed_dims

        pre_3dobjdecoder_inputs = self.pre_3dobjdecoder(occ_features)

        obj_embed = self.obj_decoder(**pre_3dobjdecoder_inputs)

        return occ_features, obj_embed
