import numpy as np

from lavis.models import load_model_and_preprocess
from models.image_encoders.abc import AbstractBaseImageEncoder
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Any
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BlipImageEncoder(AbstractBaseImageEncoder):
    def __init__(self, blip_model_name, model_type):
        super().__init__()
        blip_model, _, _ = load_model_and_preprocess(blip_model_name, model_type=model_type, device=device,
                                                     is_eval=True)
        self.visual_encoder = blip_model.visual_encoder
        self.vision_proj = blip_model.vision_proj

        del blip_model
        gc.collect()
        # for each patch create a learnable mask token (for each position), without cls token
        # self.vision_mask_token = nn.Parameter(torch.randn(1, self.visual_encoder.pos_embed.shape[1]-1, self.vision_proj.in_features))
        self.vision_mask_token = nn.Parameter(torch.randn(1, 1, self.vision_proj.in_features))

    def forward(self, x: torch.Tensor, proj=True, mask_ratio=None) -> Tuple[torch.Tensor, Any]:
        if mask_ratio is None:
            image_features = self.visual_encoder.forward_features(x)
            if proj:
                image_features = self.vision_proj(image_features)[:, 0, :]  # only use the first token
                image_features = F.normalize(image_features, dim=-1)

            return image_features  #
        return self.mask_forword(x, mask_ratio)

    def layer_shapes(self):
        return {'output_dim': self.vision_proj.out_features}

    @classmethod
    def code(cls):
        return 'blip_image_encoder'

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # L'

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1], shape: (N, L)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # shape: (N, L)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def mask_forword(self, x, mask_ratio, register_blk=-1):
        x = self.visual_encoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.visual_encoder.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # mask token padding, performance down ~0.4
        # mask_tokens = self.vision_mask_token.repeat(x.shape[0], 1, 1)
        # mask_tokens = mask_tokens[mask == 1].reshape(x.shape[0], -1, x.shape[-1])
        # mask_tokens = self.vision_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add cls token
        cls_token = self.visual_encoder.cls_token + self.visual_encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for i, blk in enumerate(self.visual_encoder.blocks):
            x = blk(x, register_blk == i)
        x = self.visual_encoder.norm(x)


        image_features = self.vision_proj(x)[:, 0, :]  # only use the first token
        # image_features = F.normalize(image_features, dim=-1)
        return image_features


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))  # L'

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1], shape: (N, L)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)  # shape: (N, L)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    print(x_masked, mask, ids_restore)
    x = x_masked
    vision_mask_token = nn.Parameter(torch.randn(1, L, x.shape[-1]))

    # use mask to get different mask token
    mask_tokens = vision_mask_token.repeat(x.shape[0], 1, 1)
    print(mask_tokens)
    # use binary mask
    print(mask_tokens.shape, mask_tokens[mask == 1].reshape(x.shape[0], -1, x.shape[-1]))
    return
    # mask_tokens = vision_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x = torch.cat([x, mask_tokens], dim=1)  # no cls token
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    # x = torch.cat([x[:, :1, :], x], dim=1)  # append cls token

    print(x)


if __name__ == "__main__":
    # show random masking
    x = torch.tensor(np.arange(24).reshape(2, 3, 4), dtype=torch.float32)
    print(x)
    random_masking(x, 0.5)