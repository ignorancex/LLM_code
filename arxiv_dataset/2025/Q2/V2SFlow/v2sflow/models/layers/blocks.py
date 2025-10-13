import torch
import torch.nn as nn

from third_party.opensora.models.layers.blocks import Attention

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class FirstBlock1D(torch.nn.Module):
    def __init__(self, dim_in, dim, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim_in, dim, 1, 1, 0),
            torch.nn.GroupNorm(groups, dim),
            nn.Mish(),
            torch.nn.Conv1d(dim, dim, 3, 1, 1),
        )
    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class FinalBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, 3, 1, 1),
            torch.nn.GroupNorm(groups, dim),
            nn.Mish(),
            torch.nn.Conv1d(dim, dim_out, 1, 1, 0),
        )
    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class Attention_with_padding(Attention):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            if mask is not None:
                raise NotImplementedError
            else:
                from flash_attn import flash_attn_func
                x = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.is_causal,
                )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            if mask is not None:
                padding_mask = torch.where(mask.bool(), 0, float('-inf')).unsqueeze(1).unsqueeze(2)
                attn += padding_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
