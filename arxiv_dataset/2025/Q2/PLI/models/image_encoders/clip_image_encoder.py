from models.image_encoders.abc import AbstractBaseImageEncoder
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ClipImageEncoder(AbstractBaseImageEncoder):
    def __init__(self, clip_model_name='ViT-B/32'):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device=device)
        # TODO: change trainer to support mixed precision
        clip_model.float()
        self.visual_encoder = clip_model.visual
        self.output_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype

        # save memory
        del clip_model
        gc.collect()

        self.vision_mask_token = nn.Parameter(torch.randn(1, 1, self.visual_encoder.proj.shape[-1]))


    def forward(self, x: torch.Tensor, proj=True, mask_ratio=None):
        # proj: same as blip image encoder, not use now
        if mask_ratio is None:
            return self.visual_encoder(x)
        return  self.mask_forword(x, mask_ratio)

    @classmethod
    def code(cls):
        return 'clip_image_encoder'
    def layer_shapes(self):
        return {'output_dim': self.output_dim}

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
        # adapt to clip ViT (not same as blip image encoder)
        x = self.visual_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # add pos embed w/o cls token
        x = x + self.visual_encoder.positional_embedding[1:, :].unsqueeze(0)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # mask token padding
        # mask_tokens = self.vision_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add cls token, class_embeding: (width,)
        cls_token = self.visual_encoder.class_embedding.to(x.dtype) + self.visual_encoder.positional_embedding[:1, :].unsqueeze(0)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # shape = [*, 1, width]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.visual_encoder.ln_pre(x)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual_encoder.ln_post(x[:, 0, :])

        if self.visual_encoder.proj is not None:
            x = x @ self.visual_encoder.proj

        # x = F.normalize(x, dim=-1)
        return x