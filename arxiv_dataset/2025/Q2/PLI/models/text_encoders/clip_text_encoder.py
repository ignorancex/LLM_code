from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ClipTextEncoder(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32"):
        """feature_size is the final output size of the encoder, generally same as the image encoder"""
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, device=device)
        # TODO: Trainer not accept mix precision, RuntimeError: expected scalar type Float but found Half
        clip_model.float()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        self.feature_size = self.text_projection.shape[-1] # nn.Parameter(torch.empty(text_cfg.width, embed_dim))

        # save memory
        del clip_model
        gc.collect()

    def forward(self, text, attn_mask=None):
        # text transform is in dataset getitem, so text: (batch_size, 1, n_ctx)
        text = text.squeeze(1)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # x = F.normalize(x, dim=-1)
        return x

    @classmethod
    def code(cls) -> str:
        return 'clip_text_encoder'