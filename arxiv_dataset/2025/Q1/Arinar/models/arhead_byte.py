import math
from math import pi
import pdb
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.adaln import AdaLNSelfAttn, AdaLNBeforeHead_W_Loc


class ByteConverter():
    def __init__(self):
        self.shifts = torch.tensor([24, 16, 8, 0]).cuda()

    def feature2byte(self, x):
        int_repr = x.view(torch.int32).unsqueeze(-1)
        bytes_tensor = ((int_repr >> self.shifts) & 0xFF)

        return bytes_tensor
    
    def byte2feature(self, bytes_tensor):
        x = (bytes_tensor << self.shifts).sum(-1)
        x = x.to(torch.int32).view(torch.float32)

        return x


class ARHead_byte(nn.Module):
    def __init__(self, num_bytes, token_embed_dim, decoder_embed_dim, 
                 inner_ar_width=768, inner_ar_depth=1, head_width=768, head_depth=1):
        super(ARHead_byte, self).__init__()
        self.num_bytes = num_bytes
        self.token_embed_dim = token_embed_dim
        self.inner_ar_width = inner_ar_width
        
        # Input projection
        self.vocabulary_size = 2 ** 8
        self.word_embed = nn.ModuleList([nn.Embedding(self.vocabulary_size, self.inner_ar_width).cuda() 
                                         for _ in range(num_bytes)])
        self.cond_proj = nn.Linear(decoder_embed_dim, inner_ar_width)

        # Start token and position embedding
        self.start_token = nn.Parameter(torch.empty(1, 1, inner_ar_width))
        self.pos_embedding = nn.Parameter(torch.empty(1, token_embed_dim * num_bytes, inner_ar_width))
        self.level_embed = nn.Embedding(num_bytes, inner_ar_width)

        # Backbone blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.drop_path_rate = 0.
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, inner_ar_depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=decoder_embed_dim,
                block_idx=block_idx, embed_dim=inner_ar_width, norm_layer=norm_layer, num_heads=16, mlp_ratio=4.,
                drop=0., attn_drop=0., drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=False, shared_aln=False,
                flash_if_available=True, fused_if_available=True,
            )
            for block_idx in range(inner_ar_depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # Model head
        self.head_nm = AdaLNBeforeHead_W_Loc(inner_ar_width, decoder_embed_dim, norm_layer, num_bytes)
        self.head = nn.Linear(inner_ar_width, self.vocabulary_size)

        self.init_weights()

        self.byte_converter = ByteConverter()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.mask_for_byte0 = [float('-inf') if 66 <= i < 128 or i >= 194 else 0
                               for i in range(self.vocabulary_size)]
        self.mask_for_byte0 = torch.tensor(self.mask_for_byte0, device="cuda").reshape(1, 1, self.vocabulary_size)
        self.loc_ids = torch.tensor(range(num_bytes), device="cuda").unsqueeze(0).repeat(1, self.token_embed_dim)
 
    def forward(self, z, target, mask=None):
        bsz = z.shape[0]

        # Convert target to byte representation
        target = self.byte_converter.feature2byte(target)  # [bsz, token_embed_dim, num_bytes]

        # Construct inputs
        inp = torch.split(target, 1, dim=-1)  # [bsz, token_embed_dim, 1] * num_bytes
        inp = [word_emb(x) for word_emb, x in zip(self.word_embed, inp)]
        inp = torch.cat(inp, dim=2)  # [bsz, token_embed_dim, num_bytes, inner_ar_width]
        inp = inp.reshape(bsz, self.token_embed_dim * self.num_bytes, self.inner_ar_width)

        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        level_pos_emb = self.level_embed(self.loc_ids).expand(bsz, -1, -1)
        x = torch.cat((start, inp[:, :-1]), dim=1) + level_pos_emb + self.pos_embedding.expand(bsz, -1, -1)

        for b in self.blocks:
            x = b(x=x, cond_BD=z, attn_bias=None, causal=True)
        print("Before head:", x.max().item(), x.min().item())
        x = self.head(self.head_nm(x, z, self.loc_ids))
        print("After head:", x.max().item(), x.min().item())

        x = x.reshape(bsz, self.token_embed_dim, self.num_bytes, self.vocabulary_size)
        print(x[0][0])
        print(x.softmax(-1)[0][0])

        # Cross entropy loss
        loss = self.loss_func(x.reshape(-1, self.vocabulary_size), target.flatten())
        loss = loss.reshape(bsz, -1).mean(-1)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        # print(x.softmax(-1)[0])
        # print(x.softmax(-1)[0].max(-1), x.softmax(-1)[0].min(-1))
        # print("Loss:", loss.item())
        # if torch.isnan(loss).any():
        #     pdb.set_trace()

        return loss

    def sample(self, z, temperature=1.0, cfg=1.0, top_p=0.99):
        bsz = z.shape[0]

        start = self.cond_proj(z).unsqueeze(1) + self.start_token.expand(bsz, 1, -1)

        for b in self.blocks: b.attn.kv_caching(True)
        x = start
        res = []

        for i in range(self.token_embed_dim):
            for byte in range(self.num_bytes):
                x = x + self.level_embed(self.loc_ids[:, byte:byte+1]).expand(bsz, 1, -1)
                pos = i * self.num_bytes + byte
                x = x + self.pos_embedding[:, pos:pos+1].expand(bsz, 1, -1)

                for b in self.blocks:
                    x = b(x=x, cond_BD=z, attn_bias=None, causal=False)
                x = self.head(self.head_nm(x, z, self.loc_ids[:, byte:byte+1]))

                # Sample from the multinomial distribution
                if byte == 0:
                    x += self.mask_for_byte0

                x = x.squeeze().softmax(dim=-1)
                x = torch.multinomial(x, 1)

                res.append(x)

                x = self.word_embed[byte](x)
        
        for b in self.blocks: b.attn.kv_caching(False)
        res = torch.cat(res, dim=1)
        res = res.reshape(bsz, self.token_embed_dim, self.num_bytes)
        res = self.byte_converter.byte2feature(res)

        return res


    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        nn.init.trunc_normal_(self.start_token.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.pos_embedding.data, mean=0, std=init_std)

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead_W_Loc):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.inner_ar_width:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.inner_ar_width].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
