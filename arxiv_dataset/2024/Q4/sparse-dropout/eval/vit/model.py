import torch
from einops.layers.torch import Rearrange
from torch import nn

from flash_dropout.layers import DropoutMM


class FeedForward(nn.Sequential):
    def __init__(self, n_embed: int, dropout: dict):
        super().__init__(
            nn.LayerNorm(n_embed),
            DropoutMM(n_embed, 4 * n_embed, **dropout),
            nn.GELU(),
            DropoutMM(4 * n_embed, n_embed, **dropout),
        )


class SelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: dict):
        assert n_embed % n_head == 0

        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed

        self.qkv_proj = DropoutMM(n_embed, 3 * n_embed, **dropout)
        self.out_proj = DropoutMM(n_embed, n_embed, **dropout)

    def forward(self, x: torch.Tensor):
        # ... d -> ... 3d
        q, k, v = self.qkv_proj(x).split(self.n_embed, dim=2)

        b, s, d = q.shape

        # b s d -> b nh s hd
        q = q.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)

        # Attn((b nh s hd), (b nh s hd), (b nh s hd)) -> (b nh s hd)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # b nh s hd -> b s d
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.out_proj(y)

        return y


class Transformer(nn.Module):
    def __init__(self, n_embed: int, n_layers: int, n_head: int, dropout: dict):
        super().__init__()
        self.norm = nn.LayerNorm(n_embed)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        SelfAttention(n_embed, n_head, dropout),
                        FeedForward(n_embed, dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        sample: tuple[torch.Tensor, torch.Tensor],
        num_classes: int,
        patch_size: tuple[int, int],
        block_size: tuple[int, int],
        n_embed: int,
        n_head: int,
        n_layers: int,
        dropout: dict,
    ):
        super().__init__()
        img, label = sample
        _, c, h, w = img.shape
        ph, pw = patch_size
        bh, bw = block_size

        assert (
            h % (bh * ph) == 0 and w % (bw * pw) == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (h // ph) * (w // pw)
        patch_dim = c * ph * pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h b1 p1) (w b2 p2) -> b (h w b1 b2) (p1 p2 c)",
                b1=bh,
                b2=bw,
                p1=ph,
                p2=pw,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, n_embed),
            nn.LayerNorm(n_embed),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, n_embed))
        self.transformer = Transformer(n_embed, n_layers, n_head, dropout)
        self.out = nn.Linear(n_embed, num_classes)

        print(f"Num parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, img: torch.Tensor):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding

        x = self.transformer(x)

        x = x.mean(dim=1)
        x = self.out(x)
        return x


if __name__ == "__main__":
    re = Rearrange(
        "b c (h b1 p1) (w b2 p2) -> b (h w b1 b2) (p1 p2 c)",
        h=4,
        w=4,
        b1=4,
        b2=4,
        p1=2,
        p2=2,
    )

    re_inv = Rearrange(
        "b (h w b1 b2) (p1 p2 c) -> b c (h b1 p1) (w b2 p2)",
        h=4,
        w=4,
        b1=4,
        b2=4,
        p1=2,
        p2=2,
    )
    torch.set_printoptions(edgeitems=10, linewidth=200)

    x = torch.arange(32 * 32).reshape(1, 1, 32, 32)
    print(x[0, 0])

    x_perm = re(x)
    x_perm[0, :17] = -1
    print(re_inv(x_perm))
