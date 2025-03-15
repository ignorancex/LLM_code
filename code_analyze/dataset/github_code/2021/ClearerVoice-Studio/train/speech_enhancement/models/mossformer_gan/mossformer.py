import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from models.mossformer_gan.conv_module import ConvModule
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.eps = 1e-5
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        ## rp_bucket: 14 x 14
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        #print('rp_bucket: {}'.format(rp_bucket.shape))
        ## values: 14 x 14 x 1
        values = self.relative_attention_bias(rp_bucket)
        #print('values: {}'.format(values.shape))
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class
class RelativePosition(nn.Module):

    def __init__(self, num_units=32, max_relative_position=128):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, x):
        length_q, length_k, device = *x.shape[-2:], x.device
        range_vec_q = torch.arange(length_q, dtype=torch.long, device=device)
        range_vec_k = torch.arange(length_k, dtype=torch.long, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = (distance_mat_clipped + self.max_relative_position)
        embeddings = self.embeddings_table[final_mat]

        return embeddings

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)


class FFConvM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class MossFormer(nn.Module):
    def __init__(
        self,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        causal = False,
        dropout = 0.1,
        norm_klass = nn.LayerNorm,
        shift_tokens = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings
        self.rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        
        self.to_hidden = FFConvM(
            dim_in = dim,
            dim_out = hidden_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        self.to_qk = FFConvM(
            dim_in = dim,
            dim_out = query_key_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)

        self.to_out = FFConvM(
            dim_in = dim*int(expansion_factor//2),
            dim_out = dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        
        self.gateActivate=nn.Sigmoid() 

    def forward(
        self,
        x,
        *,
        mask = None
    ):

        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        B, T, Q, C = x.size()
        x = x.view(B*T, Q, C)

        # prenorm
        normed_x = x 

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        residual = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, B)

        out = (att_u*v ) * self.gateActivate(att_v*u)       
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, B, mask = None):
        b, n, device, g = x.shape[0], x.shape[-2],  x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        padding = padding_to_multiple_of(n, n)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)
        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = n), (quad_q, quad_k, lin_q, lin_k, v, u))

        BT, K, Q, C = quad_q.size()
        quad_q_c = quad_q.view(B,-1,Q,C).transpose(2,1)
        quad_k_c = quad_k.view(B,-1,Q,C).transpose(2,1)
        v_c = v.view(B,-1,Q,C).transpose(2,1)
        u_c = u.view(B,-1,Q,C).transpose(2,1)

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = n)

        # calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / n
        sim_c = einsum('... i d, ... j d -> ... i j', quad_q_c, quad_k_c) / quad_q_c.shape[-2]
        #we found that using rel_pos_bias may introduce infinite loss prob!
        #sim = sim + self.rel_pos_bias(sim)

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)
        
        attn_c = F.relu(sim_c) ** 2
        attn_c = self.dropout(attn_c)
        mask_c = torch.eye(quad_q_c.shape[-2], dtype = torch.bool, device = device)
        attn_c = attn_c.masked_fill(mask_c, 0.)
        
        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        quad_out_v_c = einsum('... i j, ... j d -> ... i d', attn_c, v_c)
        quad_out_u_c = einsum('... i j, ... j d -> ... i d', attn_c, u_c)
        quad_out_v_c = quad_out_v_c.transpose(2,1).contiguous().view(BT,K,Q,C)
        quad_out_u_c = quad_out_u_c.transpose(2,1).contiguous().view(BT,K,Q,C)

        quad_out_v = quad_out_v + quad_out_v_c
        quad_out_u = quad_out_u + quad_out_u_c
        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / n
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / n
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim = 1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        quad_attn_out_v, lin_attn_out_v = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v, lin_out_v))
        quad_attn_out_u, lin_attn_out_u = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_u, lin_out_u))

        return quad_attn_out_v+lin_attn_out_v, quad_attn_out_u+lin_attn_out_u
