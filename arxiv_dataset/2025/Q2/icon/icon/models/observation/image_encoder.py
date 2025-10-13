import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Dict, Tuple, List
from timm.models.vision_transformer import VisionTransformer
from torchvision.transforms import Resize, RandomCrop, CenterCrop
from copy import deepcopy
from einops import rearrange
from icon.utils.sampler import random_sample, farthest_point_sample
from icon.utils.loss_utils import info_nce_loss


class ViT(VisionTransformer):
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        return x


class IConViT(ViT):

    def __init__(
        self,
        num_samples_mask: int,
        num_samples_unmask: int,
        temperature: float,
        enable_fps: bool,
        enable_multi_level_contrast: bool,
        gamma: float,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            num_samples_mask (int): number of samples in masked regions.
            num_samples_unmask (int): number of samples in unmasked regions.
            temperature (float): temperature coefficient of the InfoNce loss.
            enable_fps (bool): if True, enable Farthest Point Sampling (FPS);
                otherwise, use random sampling.
            enable_multi_level_contrast (bool): if True, enable multi-level contrast.
            gamma (float): weighting coefficient of multi-level contrast.
        """
        super().__init__(*args, **kwargs)
        self.num_samples_mask = num_samples_mask
        self.num_samples_unmask = num_samples_unmask
        self.temperature = temperature
        self.enable_fps = enable_fps
        self.enable_multi_level_contrast = enable_multi_level_contrast
        self.gamma = gamma

    def patchify(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        patch_size = self.patch_embed.patch_size[0]
        assert height == width and height % patch_size == 0
        x = rearrange(x, 'b c (h p) (w q) -> b (h w) (p q c)', p=patch_size, q=patch_size)
        return x
    
    def forward_loss(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): token sequences with CLS tokens (batch_size, 1+seq_len, embed_dim).
            mask (torch.Tensor): binary masks (batch_size, seq_len), where 1 for masked regions
                and 0 for unmasked regions.

        Returns:
            loss (torch.Tensor): the contrastive loss.
        """
        tokens = x[:, 1:]
        # Count masked and unmasked tokens
        count_mask = mask.sum(dim=1).unsqueeze(1)
        count_unmask = (1.0 - mask).sum(dim=1).unsqueeze(1)
        # Obtain queries corresponding to masked and unmasked regions.
        eps = 1e-6
        query_mask = (tokens * mask.unsqueeze(-1)).sum(dim=1) / (count_mask + eps)
        query_unmask = (tokens * (1.0 - mask.unsqueeze(-1))).sum(dim=1) / (count_unmask + eps)
        if self.enable_fps:
            key_unmask = farthest_point_sample(tokens, num_samples=self.num_samples_unmask, masks=1.0 - mask)
            key_mask = farthest_point_sample(tokens, num_samples=self.num_samples_mask, masks=mask)
        else:
            key_unmask, key_mask = random_sample(tokens, mask, self.num_samples_mask, self.num_samples_unmask)
        # Compute contrastive losses.
        loss_unmask = info_nce_loss(
            query=query_unmask,
            pos_key=key_unmask,
            neg_key=key_mask,
            temp=self.temperature
        )
        loss_mask = info_nce_loss(
            query=query_mask,
            pos_key=key_mask,
            neg_key=key_unmask,
            temp=self.temperature
        )
        # Losses are computed on batches with enough samples.
        flag = torch.logical_and(count_mask >= self.num_samples_mask, count_unmask >= self.num_samples_unmask).float()
        loss = ((loss_unmask + loss_mask) * flag).sum() / flag.sum()
        return loss
        
    def forward(self, x: Tensor, mask: Union[Tensor, None] = None) -> Union[Tensor, Tuple]:
        if mask is None:
            return super().forward(x)
        else:
            # For each mask patch, it is assigned a value of 1 if there are more masked pixels
            # than unmasked pixels; otherwise, assigned a value of 0.
            mask = self.patchify(mask.unsqueeze(1))
            mask = (mask.sum(dim=-1) > self.patch_embed.patch_size[0] ** 2 / 2).float()
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.pos_embed
            if self.enable_multi_level_contrast:
                weights = torch.exp(self.gamma * torch.arange(len(self.blocks), device=x.device))
                weights = weights / weights.sum()
                loss = list()
                for blk in self.blocks:
                    x = blk(x)
                    loss.append(self.forward_loss(x, mask))
                loss = (torch.stack(loss) * weights).sum()
            else:
                x = self.blocks(x)
                loss = self.forward_loss(x, mask)
            x = self.norm(x)
            x = x[:, 0]
            return x, loss
    

class MultiViewImageEncoder(nn.Module):
    
    def __init__(
        self,
        backbone: ViT,
        cameras: List,
        resize_shape: Union[int, None] = None,
        crop_shape: Union[int, None] = None
    ) -> None:
        """
        Images coming from different viewpoints are encoded independently,
        while those in the same sequences are encoded jointly.
        """
        super().__init__()
        assert len(cameras) > 0
        self.backbones = nn.ModuleDict({
            camera: deepcopy(backbone) for camera in cameras
        })
        transforms = list()
        transforms.append(Resize(resize_shape))
        crop = nn.Identity()
        if crop_shape is not None:
            crop = RandomCrop((crop_shape, crop_shape)) if self.training \
                else CenterCrop((crop_shape, crop_shape))
        transforms.append(crop)
        self.transforms = nn.Sequential(*transforms)
    
    def forward(self, images: Dict, masks: Union[Dict, None] = None) -> Union[Tensor, Tuple]:
        features = list()
        losses = list()
        for key in images.keys():
            assert key in self.backbones.keys()
            backbone = self.backbones[key]
            image = images[key]
            batch_size = image.shape[0]
            image = rearrange(image, 'b l ... -> (b l) ...')
            mask = None
            if masks is not None:
                mask = masks.get(key)
                if mask is not None:
                    mask = rearrange(mask, 'b l ... -> (b l) ...')
            if mask is None:
                image = self.transforms(image)
                inputs = [image]
            else:
                # Apply identical transformations to images and masks
                mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
                image_mask_stack = torch.stack([image, mask])
                image_mask_stack = rearrange(image_mask_stack, 't n ... -> (t n) ...')
                image_mask_stack = self.transforms(image_mask_stack)
                image, mask = rearrange(image_mask_stack, '(t n) ... -> t n ...', t=2).chunk(2)
                image, mask = image.squeeze(0), (mask.squeeze(0)[:, 0] > 0.5).float()
                inputs = [image, mask]
            outputs = backbone(*inputs)
            if isinstance(outputs, tuple):
                feature, loss = outputs
                losses.append(loss)
            else:
                feature = outputs
            feature = rearrange(feature, '(b l) ... -> b l ...', b=batch_size)
            features.append(feature)
        features = torch.cat(features, dim=-1)
        if len(losses) > 0:
            losses = torch.stack(losses).mean()
            return features, losses
        else:
            return features