import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from .SwinTransformer import SwinB


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device='cuda') 

    def __call__(self, b, h, w):
        mask = torch.zeros([b, h, w], dtype=torch.bool, device='cuda')
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  
        if self.normalize:
            eps = 1e-6
            y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale).cuda()
            x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale).cuda()

        dim_t = self.temperature ** (2 * (self.dim_t // 2) / self.num_pos_feats) 

        pos_x = x_embed[:, :, :, None] / dim_t 
        pos_y = y_embed[:, :, :, None] / dim_t  
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3)  
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) 


class MCLM(nn.Module):  
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g, memory_bank = []):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
        
        from train import args
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_pos = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
        g_pos = rearrange(g_pos, 'b c h w -> (h w) b c')

        if len(memory_bank) >= args.batch_size:
            memory_bank = memory_bank[:args.batch_size]
            memory_tokens = []
            memory_pos = []
            for i, mem in enumerate(memory_bank):
                base_scale = 2 ** (len(memory_bank) - i)
                h_in, w_in = mem.shape[2], mem.shape[3]
                h_out = max(1, h_in // base_scale)
                w_out = max(1, w_in // base_scale)
                downsampled = F.interpolate(mem, size=(h_out, w_out), mode='bilinear', align_corners=False)
                
                memory_tokens.append(rearrange(downsampled, 'b c h w -> (h w) b c'))
                pos_emb = self.positional_encoding(downsampled.shape[0], downsampled.shape[2], downsampled.shape[3])
                memory_pos.append(rearrange(pos_emb, 'b c h w -> (h w) b c'))
            
            if memory_tokens:
                memory_tokens = torch.cat(memory_tokens, 0)
                memory_pos = torch.cat(memory_pos, 0)
                g_hw_b_c = g_hw_b_c + self.dropout1(
                    self.attention[0](
                        query=g_hw_b_c + g_pos,
                        key=memory_tokens + memory_pos,
                        value=memory_tokens
                    )[0]
                )

        pools = []
        p_poses = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
            p_poses.append(rearrange(pos_emb, 'b c h w -> (h w) b c'))
        
        pools = torch.cat(pools, 0)
        p_poses = torch.cat(p_poses, 0)
        g_hw_b_c = g_hw_b_c + self.dropout1(
            self.attention[1](
                query=g_hw_b_c + g_pos,
                key=pools + p_poses,
                value=pools
            )[0]
        )

        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])
        outputs_re = torch.cat(outputs_re, 1)

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)  

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)


class inf_MCLM(nn.Module):  
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(inf_MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size() 
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
        pools = []
        p_poses = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
            p_poses.append(rearrange(pos_emb, 'b c h w -> (h w) b c'))
        pools = torch.cat(pools, 0)
        p_poses = torch.cat(p_poses, dim=0)
        pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
        g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = g_hw_b_c + self.dropout1(
            self.attention[0](
                query=g_hw_b_c + g_pos,
                key=pools + p_poses,
                value=pools
            )[0]
        )
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])
        outputs_re = torch.cat(outputs_re, 1)

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)  

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)


class MCRM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
    
    def forward(self, x):
        b, c, h, w = x.size()  
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w
        # b(4),c,h,w
        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=patches2image(loc).shape[-2:], mode='nearest')
        loc = loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))  # nl(4),c,hw
        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
        # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = torch.cat(outputs, 1)  
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb
        return torch.cat((src, glb), 0), token_attention_map


class inf_MCRM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(inf_MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
    def forward(self, x):
        b, c, h, w = x.size()  
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w
        # b(4),c,h,w
        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=patches2image(loc).shape[-2:], mode='nearest')
        loc = loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))  # nl(4),c,hw
        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
        # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = torch.cat(outputs, 1)  
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb
        return torch.cat((src, glb), 0)


class DWMF(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model//2, normalize=True)
        self.sal_conv = nn.Conv2d(1, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.patch_size = 4
        self.register_buffer('hist_importance', None)
        self.global_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

    def forward(self, x, last_pred=None):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)
        
        patch_h, patch_w = h//4, w//4
        loc_patches = rearrange(loc, 'n c (nh ph) (nw pw) -> (n nh nw) c ph pw', 
                              nh=4, nw=4, ph=patch_h, pw=patch_w)
        
        n_patches = loc_patches.size(0)
        patch_pos = self.positional_encoding(n_patches, patch_h, patch_w)
        query = loc_patches + patch_pos
        query = rearrange(query, 'nl c h w -> nl (h w) c')

        up_glb = F.interpolate(glb, size=[h * 2, w * 2], mode='bilinear')
        patched_glb = rearrange(up_glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2 * self.patch_size, wg=2 * self.patch_size)
        glb_pos = self.positional_encoding(1, *patched_glb.shape[-2:])
        glb_with_pos = patched_glb + glb_pos
        key = glb_with_pos
        key = rearrange(key, 'nl c h w -> nl (h w) c')
        
        attn_weights_flat = torch.zeros(len(query), patch_h * patch_w).cuda()
        for i in range(len(query)):
            attn_output, attn_weights = self.cross_attn(
                query=query[i],
                key=key[i],
                value=key[i],
                need_weights=True
            )
            attn_weights_flat[i] = attn_weights.mean(dim=0)
        
        current_importance = self.sigmoid(self.sal_conv(
            rearrange(attn_weights_flat, 'n_patches (h w) -> n_patches 1 h w', 
                      h=patch_h, w=patch_w)
        ))

        if last_pred is None:
            self.hist_importance = torch.empty_like(current_importance)
        else:
            up_last_pred = F.interpolate(last_pred, size=[h * 2, w * 2], mode='bilinear')
            patched_last_pred = rearrange(up_last_pred, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2 * self.patch_size, wg=2 * self.patch_size)
            last_pred_pos = self.positional_encoding(1, *patched_last_pred.shape[-2:])
            last_pred_with_pos = patched_last_pred + last_pred_pos
            key = rearrange(last_pred_with_pos, 'nl c h w -> nl (h w) c')
            hist_weights_flat = torch.zeros(len(query), patch_h * patch_w).cuda()
            for i in range(len(query)):
                attn_output, attn_weights = self.cross_attn(
                    query=query[i],
                    key=key[i],
                    value=key[i],
                    need_weights=True
                )
                hist_weights_flat[i] = attn_weights.mean(dim=0)
            last_pred_importance = self.sigmoid(self.sal_conv(
                rearrange(hist_weights_flat, 'n_patches (h w) -> n_patches 1 h w', 
                          h=patch_h, w=patch_w)
            ))
            self.hist_importance = 0.5 * last_pred_importance.detach() + 0.5 * self.hist_importance
        
        current_importance = 0.6 * current_importance + 0.4 * self.hist_importance
        if self.hist_importance.numel() != 0:
            self.hist_importance = 0.3 * self.hist_importance + 0.7 * current_importance.detach()
        else:
            self.hist_importance = current_importance.detach()

        weighted_patches = loc_patches * current_importance
        
        loc = rearrange(weighted_patches, 
                      '(n nh nw) c ph pw -> n c (nh ph) (nw pw)',
                      n=4, nh=4, nw=4)
        
        merged_loc = patches2image(loc)
        fused_feature = merged_loc + up_glb * 0.5
        
        return fused_feature


# model for single-scale training
class MVANet(nn.Module):  
    def __init__(self):
        super().__init__()
        self.backbone = SwinB(pretrained=True)
        emb_dim = 128
        self.sideout5 =  nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 =  nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 =  nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
        
        self.maps_fusion = DWMF(emb_dim, 1)

    
    def forward(self, x, f_hats, memory_bank = [], last_pred = None):
        shallow = self.shallow(x)
        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(x)
        input = torch.cat((loc, glb), dim=0)
        feature = self.backbone(input, priors=f_hats if f_hats is not None else None)

        e5 = self.output5(feature[4])
        e4 = self.output4(feature[3])
        e3 = self.output3(feature[2])
        e2 = self.output2(feature[1])
        e1 = self.output1(feature[0])
        
        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        e5 = self.multifieldcrossatt(loc_e5, glb_e5, memory_bank)  # (4,128,16,16)

        e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4))
        e4 = self.conv4(e4) 
        e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
        e3 = self.conv3(e3)
        e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
        e2 = self.conv2(e2)
        e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
        e1 = self.conv1(e1)
        loc_e1, glb_e1 = e1.split([4, 1], dim=0)
        output1_cat = patches2image(loc_e1)  # (1,128,256,256)
        # add glb feat in
        output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
        # merge
        output1_cat = self.maps_fusion(e1, last_pred)
        final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
        # shallow feature merge
        final_output = final_output + resize_as(shallow, final_output)
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to(final_output + resize_as(shallow, final_output))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)
        ####
        sideout5 = self.sideout5(e5).cuda()
        sideout4 = self.sideout4(e4) 
        sideout3 = self.sideout3(e3) 
        sideout2 = self.sideout2(e2) 
        sideout1 = self.sideout1(e1) 
        #######glb_sideouts ######
        glb5 = self.sideout5(glb_e5)
        glb4 = sideout4[-1,:,:,:].unsqueeze(0)
        glb3 = sideout3[-1,:,:,:].unsqueeze(0)
        glb2 = sideout2[-1,:,:,:].unsqueeze(0)
        glb1 = sideout1[-1,:,:,:].unsqueeze(0)
        ####### concat 4 to 1 #######
        sideout1 = patches2image(sideout1[:-1]).cuda()
        sideout2 = patches2image(sideout2[:-1]).cuda()
        sideout3 = patches2image(sideout3[:-1]).cuda()
        sideout4 = patches2image(sideout4[:-1]).cuda()
        sideout5 = patches2image(sideout5[:-1]).cuda()
        if self.training:
            return sideout5, sideout4,sideout3,sideout2,sideout1,final_output, glb5, glb4, glb3, glb2, glb1,tokenattmap4, tokenattmap3,tokenattmap2,tokenattmap1, glb_e5, glb_e1
        else:
            return final_output, glb_e5, glb_e1