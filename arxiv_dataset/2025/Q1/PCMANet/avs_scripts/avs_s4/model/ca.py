import math

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask, **kwargs):
        return self.fn(self.norm(x), mask, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = (dim_head) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.v_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.a_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        bt, n, dim = x.shape
        # x = x.reshape(-1, 5, n, dim)
        # v, a = torch.chunk(x, 2, dim=2)
        A = x[:, :1]
        V = x[:, 1:]

        v_qkv = self.v_to_qkv(V).chunk(3, dim=-1)
        a_qkv = self.a_to_qkv(A).chunk(3, dim=-1)
        qv, kv, vv = map(lambda t: rearrange(t, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=5), v_qkv)
        qa, ka, va = map(lambda t: rearrange(t, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=5), a_qkv)

        if mask != None:
            assert mask.shape[-1] * mask.shape[-2] == n - 1
            mask = mask.view(bt, n - 1, 1).repeat(1, 1, dim)
            mask = rearrange(mask, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=5)
            qv = qv * mask

        av_dots = torch.matmul(qa, kv.transpose(-1, -2)) * self.scale
        av_attn = self.attend(av_dots)
        av_attn = self.dropout(av_attn)
        av_out = torch.matmul(av_attn, vv)

        va_dots = torch.matmul(qv, ka.transpose(-1, -2)) * self.scale
        va_attn = self.attend(va_dots)
        va_attn = self.dropout(va_attn)
        va_out = torch.matmul(va_attn, va)

        a_out, v_out = rearrange(av_out, 'b h (n t) d -> (b t) n (h d)', t=5), \
                       rearrange(va_out, 'b h (n t) d -> (b t) n (h d)', t=5)
        out = torch.cat((a_out, v_out), dim=1)

        return self.to_out(out)


class Mask_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = (dim_head) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.v_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.v_to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.a_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        bt, n, dim = x.shape
        t = 5
        b = bt // t
        # x = x.reshape(-1, 5, n, dim)
        # v, a = torch.chunk(x, 2, dim=2)
        A = x[:, :1]
        V = x[:, 1:]

        a_qkv = self.a_to_qkv(A).chunk(3, dim=-1)
        v_kv = self.v_to_kv(V).chunk(2, dim=-1)
        start_pos = [0] * b

        if mask != None:
            assert mask.shape[-1] * mask.shape[-2] == n - 1
            mask = mask.view(bt, n - 1, 1)
            nonzero_rows = torch.nonzero(mask.sum(dim=2)).t()
            p1, p2 = nonzero_rows[0], nonzero_rows[1]

            for i in range(b):
                if i == 0:
                    start_pos[i] = int((p1 // t == i).sum())
                else:
                    start_pos[i] = int(start_pos[i - 1] + (p1 // t == i).sum())

            masked_v = V[p1, p2, :]
            m_q = self.v_to_q(masked_v)

            # with torch.no_grad():
            #     t = V.clone()
            #     t[p1, p2, :] = m_q
            #     qv = (t * mask).view(bt, -1, dim)
            qv = rearrange(m_q, 'n (h d) -> h n d', h=self.heads)
            kv, vv = map(lambda z: rearrange(z, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=t), v_kv)
            qa = rearrange(a_qkv[0], '(b t) n (h d) -> b h (n t) d', h=self.heads, t=t)
            ka, va = map(lambda z: rearrange(z, '(b t) n (h d) -> h (b n t) d', h=self.heads, t=t), a_qkv[1:])
            av_dots = torch.matmul(qa, kv.transpose(-1, -2)) * self.scale
            av_attn = self.attend(av_dots)
            av_attn = self.dropout(av_attn)
            av_out = torch.matmul(av_attn, vv)

            va_dots = torch.matmul(qv, ka.transpose(-1, -2)) * self.scale

            for i in range(b):
                if i == 0:
                    va_dots[:, :start_pos[i], t*(i+1):] = -math.inf
                else:
                    va_dots[:, start_pos[i - 1]:start_pos[i], :t*i] = -math.inf
                    va_dots[:, start_pos[i - 1]:start_pos[i], t*(i+1):] = -math.inf

            va_attn = self.attend(va_dots)
            va_attn = self.dropout(va_attn)
            va_out = torch.matmul(va_attn, va)

            a_out, v_out = rearrange(av_out, 'b h (n t) d -> (b t) n (h d)', t=t), \
                           rearrange(va_out, 'h n d -> n (h d)')

            with torch.no_grad():
                t = V.clone()
                t[p1, p2, :] = v_out
                v_out = (t * mask).view(bt, -1, dim)

            out = torch.cat((a_out, v_out), dim=1)

        else:
            qv = self.v_to_q(V)

            v_qkv = (qv, *v_kv)

            qv, kv, vv = map(lambda z: rearrange(z, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=t), v_qkv)
            qa, ka, va = map(lambda z: rearrange(z, '(b t) n (h d) -> b h (n t) d', h=self.heads, t=t), a_qkv)

            av_dots = torch.matmul(qa, kv.transpose(-1, -2)) * self.scale
            av_attn = self.attend(av_dots)
            av_attn = self.dropout(av_attn)
            av_out = torch.matmul(av_attn, vv)

            va_dots = torch.matmul(qv, ka.transpose(-1, -2)) * self.scale
            va_attn = self.attend(va_dots)
            va_attn = self.dropout(va_attn)
            va_out = torch.matmul(va_attn, va)

            a_out, v_out = rearrange(av_out, 'b h (n t) d -> (b t) n (h d)', t=t), \
                           rearrange(va_out, 'b h (n t) d -> (b t) n (h d)', t=t)
            out = torch.cat((a_out, v_out), dim=1)

        return self.to_out(out)


class CA(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Mask_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            ]))

    def forward(self, x, mask):
        for attn, in self.layers:
            x = attn(x, mask) + x
        return x


class QuerySelectedCA(nn.Module):
    def __init__(self, *, image_size, patch_size, depth, heads, channels=256,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        dim = patch_dim

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CA(dim, depth, heads, dim_head, dropout)

    def forward(self, img, audio, mask):
        # img [B,C,H,W]
        # audio [B,256]
        b, c, h, w = img.shape
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        T = 5
        aud = audio.view(-1, T, audio.shape[-1])
        aud_tokens = aud.reshape(-1, 1, aud.shape[-1])

        x = torch.cat((aud_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x, mask)

        aud_tokens = x[:, 0]
        x = x[:, 1:]
        x = x.reshape(-1, c, h, w)

        return x, aud_tokens


if __name__ == '__main__':
    v = QuerySelectedCA(
        image_size=14,
        patch_size=1,
        depth=1,
        heads=8,
        dim_head=32,
        dropout=0,
        emb_dropout=0
    )
    threshold = 0.8
    img = torch.randn(4 * 5, 256, 14, 14)
    aud = torch.randn(4 * 5, 256)
    mask = torch.rand(4 * 5, 1, 14, 14)
    maska = (mask <= threshold).float().view(mask.shape)
    maskb = (mask >= 1 - threshold).float().view(mask.shape)
    mask = maska * maskb
    x, a = v(img, aud, mask)
    # print(v.transformer.layers)
