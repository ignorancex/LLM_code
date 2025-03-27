import sys
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM

import warnings
warnings.filterwarnings('ignore')

# ---------------- hardcoded values - this script is run once for each dataset
device = torch.device("cuda")

data = "../data/wikitext-103"
dataset = "wt103"

batch_size = 10
n_head = 10
ext_len = 0
tgt_len = 150
mem_len = 150
clamp_len = 400

path = "test-wt103/20191208-073508/model.pt"

# ---------------- load checkpoint, setup
with open(path, 'rb') as f:
    model = torch.load(f)

corpus = get_lm_corpus(data, dataset)
ntokens = len(corpus.vocab)

tr_iter = corpus.get_iterator('train', batch_size, tgt_len,
    device=device, ext_len=ext_len)


def compute_attention(module, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if module.pre_lnorm:
                w_heads = module.qkv_net(module.layer_norm(cat))
            else:
                w_heads = module.qkv_net(cat)
            r_head_k = module.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if module.pre_lnorm:
                w_heads = module.qkv_net(module.layer_norm(w))
            else:
                w_heads = module.qkv_net(w)
            r_head_k = module.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, module.n_head, module.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, module.n_head, module.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, module.n_head, module.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, module.n_head, module.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = module._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(module.scale)

         #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = module.dropatt(attn_prob)

        return attn_prob

def pad(source, target_shape):
    target = torch.zeros(target_shape)
    d1, d2, d3, d4 = source.shape
    target[0:d1, 0:d2, 0:d3, 0:d4] = source
    return target.cuda()


n_layer  = len(model.layers)
# ---------------- setup for rolling average
avg_attn = [torch.zeros(tgt_len, tgt_len + mem_len, batch_size, n_head).cuda() for i in range(0, n_layer)]
counts   = [torch.zeros(tgt_len, tgt_len + mem_len, batch_size, n_head).cuda() for i in range(0, n_layer)]

model.eval()
with torch.no_grad():
    mems = tuple()
    for idx, (data, target, seq_len) in enumerate(tr_iter):
        ret = model(data, target, *mems)
        _ , mems = ret[0], ret[1:]
        dec_inp = data
        qlen, bsz = dec_inp.size()
        word_emb = model.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if model.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - model.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        hids = []

        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if model.clamp_len > 0:
            pos_seq.clamp_(max=model.clamp_len)
        pos_emb = model.pos_emb(pos_seq)

        core_out = model.drop(word_emb)
        pos_emb = model.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(model.layers):
            mems_i = None if mems is None else mems[i]

            core_out = layer(core_out, pos_emb, model.r_w_bias,
                    model.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)

            attn_score = compute_attention(layer.dec_attn, core_out, pos_emb, model.r_w_bias, model.r_r_bias, attn_mask=dec_attn_mask, mems=mems_i)

            hids.append(core_out)

            p1 = 1         / (counts[i] + 1)
            p2 = counts[i] / (counts[i] + 1)

            avg_attn[i] = p1 * pad(attn_score, avg_attn[i].shape) + p2 * avg_attn[i]
            d1, d2, d3, d4 = attn_score.shape
            counts[i][0:d1, 0:d2, 0:d3, 0:d4] +=1

# average over batch size dimension
final_scores = []
for i in range(0, n_layer):
    arr = torch.mean(avg_attn[i], dim = 2, keepdim=True)
    final_scores.append(arr)

# save attention patterns to disk
save_avg_attn = []
for i in range(0, n_layer):
    arr = final_scores[i].cpu().detach().numpy()
    save_avg_attn.append (arr)
fn = "attn_scores.npy"
np.save(fn, save_avg_attn)


