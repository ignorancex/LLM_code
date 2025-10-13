import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, Dict, Tuple
from icon.models.observation.image_encoder import MultiViewImageEncoder


class MultiModalObsEncoder(nn.Module):
    
    def __init__(
        self,
        image_encoder: MultiViewImageEncoder,
        low_dim_shape: int,
        low_dim_embed_dim: int,
        flatten_features: Optional[bool] = False
    ) -> None:
        """
        Args:
            low_dim_shape (int): shape of the low-dimensional state.
            low_dim_embed_dim (int): embedding dimension of the low-dimensional state.
            flatten_features (bool, optional): if True, features with shape
                (batch_size, obs_horizon, dim) would be flattened into shape
                (batch_size, obs_horizon * dim) before returned.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.low_dim_encoder = nn.Linear(low_dim_shape, low_dim_embed_dim)
        self.flatten_features = flatten_features

    def forward(self, obs: Dict, image_masks: Union[Dict, None] = None) -> Union[Tensor, Tuple]:
        outputs = self.image_encoder(obs['images'], image_masks)
        if isinstance(outputs, tuple):
            image_features, loss = outputs
        else:
            image_features = outputs
            loss = None
        low_dim_features = self.low_dim_encoder(obs['low_dims'])
        features = torch.cat([image_features, low_dim_features], dim=-1)
        if self.flatten_features:
            features = features.flatten(1)
        if loss is None:
            return features
        else:
            return features, loss