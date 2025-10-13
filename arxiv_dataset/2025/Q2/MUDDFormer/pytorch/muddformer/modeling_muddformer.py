from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange
from collections import namedtuple
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union

try:
    from .configuration_muddformer import MUDDFormerConfig
except:
    from configuration_muddformer import MUDDFormerConfig

from transformers.modeling_utils import PreTrainedModel


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        self.seq_length = max_seq_length
        cache_shape = (max_batch_size, n_heads, self.seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        B,N,S,D = v_val.shape
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

class LayerCache(nn.Module):
    def __init__(self, max_batch_size, num_layers, model_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (num_layers+1, max_batch_size, 1, model_dim) # LBTD
        self.register_buffer('layer_cache', torch.zeros(cache_shape, dtype=dtype))
    
    def update(self, x, lidx):
        self.layer_cache[lidx] = x
        return self.layer_cache[:lidx+1]

class MultiwayDynamicDenseBlock(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx: int, last_layer=False) -> None:
        super().__init__()
        self.norm = RMSnormNoscale(epsilon=config.norm_eps)
        self.C = len(config.dense_type) if not last_layer else 1
        self.lidx = lidx
        l = lidx + 2
        hid_dim, out_dim = l * self.C, l * self.C
        if last_layer and config.expand_last: hid_dim *= 4  
        if config.round64: hid_dim = (hid_dim// 64 +1) * 64 
        self.w1 = nn.Linear(config.dim, hid_dim, bias=False)
        self.act = nn.GELU() 
        self.w2 = nn.Linear(hid_dim, out_dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x) 
        dw = self.w2(self.act(self.w1(x))) # BTD->BTL
        dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.C)
        return dw
    
    def layer_mix(self, hids, dw)-> Tensor:
        x = tuple([sum(dw[cidx,:,:,j,None] * hids[j] for j in range(self.lidx+2)) for cidx in range(self.C)]) # BTL, LBTD-> BTD
        return x

class MUDDFormer(PreTrainedModel):
    config_class=MUDDFormerConfig
    '''
    MUDDFormer's implementation is adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L89 
    '''
    def __init__(self, config: MUDDFormerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing 
        self.is_training = config.is_training

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, lidx) for lidx in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        C = len(self.config.dense_type)
        self.dense_bs = nn.ParameterList([nn.Parameter(data=torch.randn(C if lidx != config.n_layer-1 else 1, lidx+2)) for lidx in range(config.n_layer)])

        self.layer_cache = None
        self.use_layer_cache = False if self.is_training else self.config.use_layer_cache
        
        self.dynamic = self.config.dynamic_dense
        self.dense = self.config.dense
        if self.dynamic:
            self.dynamic_dense = nn.ModuleList([MultiwayDynamicDenseBlock(config, lidx, last_layer=lidx==config.n_layer-1) for lidx in range(config.n_layer)])

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def tie_weights(self): # placeholder
        return 

    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.bfloat16):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        if not self.config.is_training:
            if self.use_layer_cache:
                self.layer_cache = LayerCache(max_batch_size, self.config.n_layer, self.config.dim, dtype=dtype)
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype=dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base).to(self.tok_embeddings.weight.device)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool, device=self.tok_embeddings.weight.device))

    def generate(self, input_ids, num_tokens_to_generate=10, compiled_decode_one_token=None):
        batch_size, seq_length = input_ids.shape
        input_pos = torch.arange(seq_length, device=self.device)
        generated_ids = torch.zeros(batch_size, seq_length + num_tokens_to_generate, dtype=torch.int, device=self.device)
        generated_ids[:, :seq_length] = input_ids.to(self.device).to(torch.int)
        logits = self.forward(input_ids, input_pos=input_pos,return_tensor=True)
        _next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        next_token = torch.zeros(self.max_batch_size, 1, device=self.device, dtype=torch.int)
        next_token[:batch_size] = _next_token
        generated_ids[:, seq_length] = next_token[:batch_size, 0]
        input_pos = torch.tensor([seq_length], device=self.device)
        for _ in range(1, num_tokens_to_generate):
            if compiled_decode_one_token is not None:
                next_token = compiled_decode_one_token(self, next_token.clone(), input_pos)
            else:
                next_token = self.decode_one_token(next_token.clone(), input_pos)
            generated_ids[:, input_pos+1] = next_token.int()[:batch_size]
            input_pos += 1
        return generated_ids
    
    def decode_one_token(self, cur_token, input_pos):
        logits = self.forward(
            cur_token,
            input_pos=input_pos,
            return_tensor=True
        )
        new_token = torch.argmax(logits[:, -1], dim=-1)[:,None]
        return new_token

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, return_tensor=False) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if input_pos is None:
            input_pos = torch.arange(idx.shape[-1], device=idx.device, dtype=torch.int)
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        _, seqlen, _ = x.shape
        use_layer_cache = self.use_layer_cache and seqlen == 1
        if use_layer_cache:
            self.layer_cache.update(x, 0)
        else:
            hiddens = [x]
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing:
                x = checkpoint(layer, x, input_pos, freqs_cis, mask)
            else:
                x = layer(x, input_pos, freqs_cis, mask)
            if use_layer_cache:
                _hidden = self.layer_cache.update(x, i+1) # LBTD
            else:
                hiddens.append(x)
                _hidden = torch.stack(hiddens)
            if self.dynamic and self.dense:
                dw = self.dynamic_dense[i](x) # BTD -> CBTL
                dw = dw + self.dense_bs[i][:,None,None,:] # CBTL
                if seqlen > 1:
                    x = torch.einsum('LBTD, CBTL -> CBTD', _hidden, dw)
                else:
                    x = self.dynamic_dense[i].layer_mix(_hidden, dw)

        if self.config.dense_type == 'qkvr' and self.config.dense and self.config.dynamic_dense:
            x = x[0]
        x = self.norm(x)
        logits = self.output(x)
        if return_tensor: 
            return logits
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=logits)

class TransformerBlock(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx) -> None:
        super().__init__()
        self.lidx = lidx
        self.config = config
        self.attention = Attention(config, lidx)
        self.feed_forward = FeedForward(config, lidx)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        if self.config.sepln and self.lidx > 0 :
            self.attention_norms = torch.nn.ModuleList([RMSNorm(config.dim, config.norm_eps) for _ in range(3)])
        else:
            self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Union[Tuple[Tensor], Tensor], input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        if self.lidx == 0 or self.config.dense_type == 'l' or not self.config.dense:
            res = x
            normed_x = self.attention_norm(x)
        elif self.config.dense_type == 'qkvr':
            res = x[-1] # for mlp
            if not self.config.sepln:
                normed_x = self.attention_norm(x[:3])
            else:
                normed_x = tuple([norm_fn(_x) for norm_fn, _x in zip(self.attention_norms, x[:3])])
        attn_out = self.attention(normed_x, freqs_cis, mask, input_pos)
        h = res +  attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Attention(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx):
        super().__init__()
        assert config.dim % config.n_head == 0
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.config = config
        if self.config.dense_type == 'l' or not self.config.dense:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        elif self.config.dense_type == 'qkvr':
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wk = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
            self.wv = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)

        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.lidx = lidx
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict and (self.config.dense_type == 'l' or not self.config.dense):
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Union[Tuple[Tensor], Tensor], freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        if self.lidx == 0 or self.config.dense_type == 'l' or not self.config.dense:
            bsz, seqlen, _ = x.shape
        else:
            C, (bsz, seqlen, _) = len(x), x[0].shape
        kv_size = self.n_local_heads * self.head_dim

        if self.config.dense_type == 'l' or not self.config.dense:
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

            q = q.view(bsz, seqlen, self.n_head, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        elif self.config.dense_type == 'qkvr':
            if self.lidx == 0:
                xq, xk, xv = x, x, x
            else:
                xq, xk, xv = x[0], x[1], x[2]
            q = self.wq(xq).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.wk(xk).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = self.wv(xv).view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            if seqlen == 1:
                k, v = self.kv_cache.update(input_pos, k, v)
            else:
                _, _ = self.kv_cache.update(input_pos, k, v)
        
        if seqlen == 1: # one-token generation
            k_mask = mask[:,:,:,:self.kv_cache.seq_length]
        else:# prefill
            k_mask = mask[:,:,:,:k.shape[-2]] 

        logits = q @ k.transpose(-2, -1) * self.scale_factor 
        dtype = logits.dtype
        min_value = torch.finfo(torch.float32).min
        logits = logits.to(dtype=torch.float32)
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        probs = probs.to(dtype=dtype)
        y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx, round128=True, scale_with_layer=True) -> None:
        super().__init__()
        hid_dim = config.intermediate_size
        if scale_with_layer:
            hid_dim = hid_dim * (lidx/(config.n_layer -1) +0.5)
        if round128:
            hid_dim = round(hid_dim / 128) * 128
        self.w1 = nn.Linear(config.dim, hid_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, mode='half') -> Tensor:
    if mode == 'half':
        xshaped = x.float().reshape(*x.shape[:-1], 2,-1).transpose(-1,-2) 
    elif mode == 'alternative':
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def match_weight_muddformer(model, w, strict=False):
    map_dict={'wq':'query', 'wk':'key', 'wv':'value', 'wo':'post', 'w1': 'ffn_layer1_gate', 'w3': 'ffn_layer1', 'w2': 'ffn_layer2',
              'weight': 'w'}
    E, H, D = model.config.dim, model.config.n_head, model.config.head_dim
    N = model.config.vocab_size
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'tok_embeddings.weight':
            v = w['state.mdl_vars.params.lm.embedding_lookup.emb_var']#[:50257,:]
        elif k == 'norm.weight':
            v = w['state.mdl_vars.params.lm.final_ln.scale']
        elif k == 'output.weight':
            v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T#[:50257,:]  # E,N -> N,E
        elif 'dense_bs' in k: # static dense w
            lidx = int(k.split('.')[-1])
            v = w[f'state.mdl_vars.params.lm.transformer.dense_conn_{lidx}']
        elif 'dynamic_dense' in k:
            lidx = int(k.split('.')[1])
            widx = int(k.split('.')[2][-1]) # 1 or 2 in w1, w2
            v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.dynamic_dense_conn{widx}_{lidx}'].T
        else:
            assert 'layers' in k
            lidx = int(k.split('.')[1])
            if '.attention.' in k:
                _, _, _, ptype, wtype = k.split('.')
                if ptype in ['wq', 'wk', 'wv', 'wo']:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'].reshape(E,E)
                    if ptype != 'wo':
                        v = v.T
                elif ptype in ['q_norm', 'k_norm']:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.self_attention.{map_dict.get(ptype, ptype)}.scale']
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.{map_dict[ptype]}.linear.w'].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.layer_norm.scale']
            elif 'attention_norm' in k: # attention layernorm
                if 'attention_norms' in k:
                    ln_idx = int(k.split('.')[3])
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.layer_norms_{ln_idx}.scale']
                else:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.layer_norm.scale']
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=strict)
    return model
