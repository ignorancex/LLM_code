import math
import torch
import torch.nn as nn

from operator import mul
from functools import reduce
from torch.nn import Dropout
from .BaseModel import BaseIncModel
from timm.models.layers import trunc_normal_


class CustomCLIP(BaseIncModel):
    def __init__(self, classes_names, clip_model):
        super().__init__(classes_names, clip_model)

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre

        self.vision_transformer = clip_model.visual.transformer

        self.ln_post = clip_model.visual.ln_post
        self.image_proj = clip_model.visual.proj

        self.logit_scale = clip_model.logit_scale
        self.prompt_dropout = Dropout(0.1)  # Dropout
        self.num_tokens = 5  # number of prompted tokens
        prompt_dim = clip_model.visual.transformer.width
        self.prompt_proj = nn.Identity()

        val = math.sqrt(6. / float(3 * reduce(mul, [16, 16], 1) + prompt_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self.num_tokens, prompt_dim, dtype=self.dtype))  # [1，num_tokens，768]
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        total_d_layer = 11
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
            total_d_layer, self.num_tokens, prompt_dim, dtype=self.dtype))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)  # [11，num_tokens，768]
        self.head = nn.Linear(clip_model.visual.output_dim, len(classes_names))
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)
        self.head.half()

    def pre_blocks(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x

    def incorporate_prompt(self, x):
        """
        :param x: images ,        shape：[B, C, H, W]
        :return:  pre-blocks out: shape: [B, cls_token + n_prompt + n_patches, hidden_dim]
        """
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.pre_blocks(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = len(self.vision_transformer.resblocks)

        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.vision_transformer.resblocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:1, :, :],
                        deep_prompt_emb.permute(1, 0, 2),
                        hidden_states[(1 + self.num_tokens):, :, :]
                    ), dim=0)

                hidden_states = self.vision_transformer.resblocks[i](hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(hidden_states[:, 0, :])

        return x

    def forward_shallow_prompt(self, embedding_output):

        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        hidden_states = self.vision_transformer(embedding_output)
        hidden_states = hidden_states.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(hidden_states[:, 0, :])

        return x

    def forward_visual_prompts(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x.type(self.dtype))
        # encoded = self.forward_shallow_prompt(embedding_output)
        encoded = self.forward_deep_prompt(embedding_output)
        if self.image_proj is not None:
            x = encoded @ self.image_proj

        return x

    def encode_image(self, image):
        return self.forward_visual_prompts(image)

    def forward(self, image):
        image_features = self.forward_visual_prompts(image)
        logits = self.head(image_features)
        return logits
