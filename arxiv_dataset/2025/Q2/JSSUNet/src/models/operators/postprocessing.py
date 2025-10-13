import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

from src.models.operators.basic_layers import ResBlockBN


class KBestAttentionWeights(torch.nn.Module):
    def __init__(self, channels, aux_channels, window_size, patch_size, emb_dim=32, k_nearest=15, bias=False,norm="None", learnable_norm=False):
        super(KBestAttentionWeights, self).__init__()
        self.emb_conv = nn.Conv2d(aux_channels * patch_size * patch_size, emb_dim, 1, bias=bias)
        self.g = nn.Conv2d(channels, channels, 1, bias=bias)
        self.phi = nn.Conv2d(emb_dim, emb_dim, 1, bias=bias)
        self.theta = nn.Conv2d(emb_dim, emb_dim, 1, bias=bias)
        self.window_size = window_size
        self.k_nearest = int((window_size*window_size)*k_nearest/100.)
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.softmax = nn.Softmax(dim=-1)
        self.norm_aux = nn.InstanceNorm2d(aux_channels,affine=learnable_norm) if norm in ["Single", "Double"] else nn.Identity()
        self.norm_u = nn.InstanceNorm2d(channels,affine=learnable_norm) if norm in ["Double"] else nn.Identity()
        self.eps = 1e-6

    def forward(self, u, aux):
        aux=self.norm_aux(aux)
        u=self.norm_u(u)
        wxy, wxy_idx = self.weights(aux)
        g = self.g(u)
        g = self.get_window_neighbor(g)
        wxy_idx = wxy_idx.long()
        _, c, _, _, _ = g.size()
        wxy_idx = wxy_idx.unsqueeze(1).expand(-1, c, -1, -1, -1)  # (b, c, h, w, k)
        gather_val = torch.gather(g, dim=4, index=wxy_idx)
        u_nl = einsum(wxy, gather_val, 'b h w k, b c h w k -> b c h w')
        return u_nl

    def weights(self, u):
        u = self.patch_embedding(u)
        phi = self.phi(u)
        theta = self.theta(u)
        theta = self.get_window_neighbor(theta)
        att = einsum(phi, theta ,'b c h w, b c h w k ->b h w k')
        att_best, att_idx = torch.topk(att, k=self.k_nearest, dim=-1)
        return self.softmax(att_best), att_idx

    def get_window_neighbor(self, u):
        b, c, h, w = u.size()
        window = F.unfold(u, self.window_size, padding=self.window_size // 2)
        window = rearrange(window,'b (c w1 w2)  (h w) -> b c h w (w1 w2)', c=c, h=h,w=w, w1=self.window_size, w2=self.window_size)
        return window

    def patch_embedding(self, u):
        b, c, h, w = u.size()
        patches = F.unfold(u, self.patch_size, padding=self.patch_size // 2)
        u_patches = rearrange(patches,'b cpp (h w) -> b cpp h w', h=h,w=w)
        return self.emb_conv(u_patches)

class MHAKBest_u(torch.nn.Module):
    def __init__(self, channels, window_size, patch_size, emb_dim=32, k_nearest=15, bias=False, norm="None", learnable_norm=False):
        super(MHAKBest_u, self).__init__()

        self.head1 = KBestAttentionWeights(channels=channels, aux_channels=channels, window_size=window_size,
                                                    patch_size=patch_size, emb_dim=emb_dim, k_nearest=k_nearest, bias=bias,
                                                    norm=norm, learnable_norm=learnable_norm)
        self.head2 = KBestAttentionWeights(channels=channels, aux_channels=channels, window_size=window_size,
                                                   patch_size=patch_size, emb_dim=emb_dim, k_nearest=k_nearest, bias=bias,
                                                   norm=norm, learnable_norm=learnable_norm)
        self.head3 = KBestAttentionWeights(channels=channels, aux_channels=channels, window_size=window_size,
                                              patch_size=patch_size, emb_dim=emb_dim, k_nearest=k_nearest, bias=bias,
                                              norm=norm, learnable_norm=learnable_norm)
        self.mlp = nn.Sequential(*[nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1)])

    def forward(self, u):
        head1 = self.head1(u, u)
        head2 = self.head2(u, u)
        head3 = self.head3(u, u)
        multi_head = torch.stack([head1, head2, head3], dim=4)
        return self.mlp(multi_head).squeeze(4)
class NLPost(nn.Module):
    def __init__(self, channels, window_size, features, patch_size, emb_dim=8, k_nearest=10):
        super(NLPost, self).__init__()
        self.NL1 = MHAKBest_u(channels=channels, window_size=window_size,
                            patch_size=patch_size, emb_dim=emb_dim, k_nearest=k_nearest)
        self.NL2 = MHAKBest_u(channels=channels, window_size=window_size,
                            patch_size=patch_size, emb_dim=emb_dim, k_nearest=k_nearest)
        self.disp = window_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=channels + channels, out_channels=features, kernel_size=3, padding=1,bias=False),
            ResBlockBN(features=features, kernel_size=3),
            nn.Conv2d(in_channels=features, out_channels=channels,
                      kernel_size=3, padding=1,bias=False))

    def forward(self, u):
        u_nl = self.NL1(u)
        u_nl = torch.roll(u_nl, shifts=(self.disp, self.disp), dims=(2, 3))
        u_nl = self.NL2(u_nl)
        u_nl = torch.roll(u_nl, shifts=(-self.disp, -self.disp), dims=(2, 3))
        return self.residual(torch.cat([u_nl, u], dim=1)) + u