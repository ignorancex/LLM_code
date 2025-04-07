# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_layers import *

N_MAX_POSITIONS = 512  # maximum input sequence length

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]


logger = getLogger()


class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = params.emb_dim       # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        if params.n_langs > 1:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        try:
            self.add_langemb_everylayer = params.add_langemb_everylayer
        except:
            self.add_langemb_everylayer = False
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout, gelu_activation=params.gelu_activation))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            if params.share_inout_emb:
                self.pred_layer.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        elif mode == 'predict_wo_targets':
            return self.predict_wo_targets(**kwargs)
        elif mode == 'predict_wo_targets_all_pos':
            return self.predict_wo_targets_all_pos(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen, (lengths.max().item(), slen)
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.size() == (slen, bs)
            langs = langs.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        tensor = self.embeddings(x)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if langs is not None:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            # add lang embeddings everywhere as an option
            if self.add_langemb_everylayer:
                tensor = tensor + self.lang_embeddings(langs)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def predict_wo_targets(self, tensor, pred_mask):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        tensor = torch.transpose(tensor, 0, 1)
        pred_mask = torch.transpose(pred_mask, 0, 1)
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores = self.pred_layer.get_scores(masked_tensor)
        return scores

    def predict_wo_targets_all_pos(self, tensor, pred_mask):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        #tensor = torch.transpose(tensor, 0, 1)
        pred_mask = torch.transpose(pred_mask, 0, 1)
        #masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores = self.pred_layer.get_scores(tensor)
        return scores

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # language IDs
        langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = langs.unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, tgt_lang_id, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, length_penalty, early_stopping, max_len) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # language IDs
        langs = positions.clone().fill_(tgt_lang_id)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]               # (bs * beam_size, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)       # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            print(name)
        for name, param in state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            # if 'order_pred_proj' in name:
            #     continue

            try:
                own_state[name].copy_(param)
            except:
                try:
                    own_state[name[7:]].copy_(param) # get rid of the 'module.' prefix
                except:
                    print(name)
                    print(param.size())
                    print(own_state[name].size())
                    lang_emb = torch.cat([param[4:5], param[14:]], dim=0)
                    own_state[name].copy_(lang_emb)

class TransformerOrderPredModel(TransformerModel):
    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__(params, dico, is_encoder, with_output)
        self.order_pred_layer_type = params.order_pred_layer_type
        self.rl_baseline = params.rl_baseline
        #self.rl_alg = 'PolicyGradient'
        self.rl_alg = params.rl_alg

        if self.order_pred_layer_type == '2l_mlp':
            self.order_pred_proj_l1 = Linear(self.dim, self.dim)
            self.nonlinear_l1 = torch.nn.LeakyReLU(0.1)
            self.order_pred_proj = Linear(self.dim, 1)
            if self.rl_alg == 'PPO':
                self.old_order_pred_proj_l1 = Linear(self.dim, self.dim)
                self.old_nonlinear_l1 = torch.nn.LeakyReLU(0.1)
                self.old_order_pred_proj = Linear(self.dim, 1)
        elif self.order_pred_layer_type == 'linear':
            self.order_pred_proj = Linear(self.dim, 1)
            if self.rl_alg == 'PPO':
                self.old_order_pred_proj = Linear(self.dim, 1)
        else:
            raise NotImplementedError

        if self.rl_baseline == 'state_value':
            self.rl_baseline_fn = nn.ModuleList()
            self.baseline_proj_l1 = Linear(self.dim, self.dim)
            self.baseline_nonlinear_l1 = torch.nn.LeakyReLU(0.1)
            self.baseline_proj = Linear(self.dim, 1)
            self.rl_baseline_fn.extend([self.baseline_proj_l1, self.baseline_nonlinear_l1, self.baseline_proj])

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        elif mode == 'predict_wo_targets':
            return self.predict_wo_targets(**kwargs)
        elif mode == 'predict_wo_targets_all_pos':
            return self.predict_wo_targets_all_pos(**kwargs)
        elif mode == 'predict_get_loss':
            return self.predict_get_loss(**kwargs)
        elif mode == 'predict_get_ppo_loss':
            return self.predict_get_ppo_loss(**kwargs)
        elif mode == 'predict_get_ppo_loss_v2':
            return self.predict_get_ppo_loss_v2(**kwargs)
        elif mode == 'predict_baseline_wo_targets':
            return self.predict_baseline_wo_targets(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        #tensor = torch.transpose(tensor, 0, 1)
        #pred_mask = torch.transpose(pred_mask, 0, 1)
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def predict_wo_targets(self, tensor, pred_mask, use_old_policy=False):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        tensor = torch.transpose(tensor, 0, 1)
        pred_mask = torch.transpose(pred_mask, 0, 1)
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        scores = self.pred_layer.get_scores(masked_tensor)

        if self.order_pred_layer_type == '2l_mlp':
            if use_old_policy:
                order_tensor = self.nonlinear_l1(self.old_order_pred_proj_l1(tensor))
                full_order_logits = self.old_order_pred_proj(order_tensor).squeeze(2)
            else:
                order_tensor = self.nonlinear_l1(self.order_pred_proj_l1(tensor))
                full_order_logits = self.order_pred_proj(order_tensor).squeeze(2)
                #print(full_order_logits)
        elif self.order_pred_layer_type == 'linear':
            if use_old_policy:
                full_order_logits = self.old_order_pred_proj(tensor).squeeze(2)
            else:
                full_order_logits = self.order_pred_proj(tensor).squeeze(2)
        return scores, full_order_logits

    def predict_wo_targets_all_pos(self, tensor, pred_mask):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        #tensor = torch.transpose(tensor, 0, 1)
        #pred_mask = torch.transpose(pred_mask, 0, 1)
        #masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        #print(tensor.size(), self.dim)
        seq_len, bsz = tensor.size(0), tensor.size(1)
        scores = self.pred_layer.get_scores(tensor.reshape(seq_len*bsz, self.dim))
        # print(scores.reshape(seq_len, bsz, self.dim))
        # exit()
        vocab_size = scores.size(-1)
        return scores.reshape(seq_len, bsz, vocab_size)

    def predict_baseline_wo_targets(self, tensor, src_lens):
        tensor = torch.transpose(tensor, 0, 1)
        batch_indices = torch.arange(tensor.size(0))
        int_out = tensor[[batch_indices, src_lens]]
        for _fn in self.rl_baseline_fn:
            int_out = _fn(int_out)
        return int_out

    def predict_get_loss(self, logits, labels, advantage, mask):
        bsz, seq_len = labels.size(0), labels.size(1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(2)), labels.reshape(-1), ignore_index=-100, reduction='none')
        loss = loss.view(bsz, seq_len)

        # full_reward = torch.zeros(labels.size(0), labels.size(1)).cuda()
        # cur_reward = reward
        #
        # for i in reversed(range(labels.size(1))):
        #     full_reward[:, i] = cur_reward * mask[:, i]
        #     cur_reward = cur_reward * torch.pow(discount_factor, mask[:, i])

        loss = loss * advantage
        loss = loss.sum() / mask.sum()
        return loss

    def predict_get_ppo_loss(self, logits, old_logits, labels, advantage, mask, clip_param):
        bsz, seq_len = labels.size(0), labels.size(1)
        ce = F.cross_entropy(logits, labels.reshape(-1), ignore_index=-100, reduction='none')
        old_ce = F.cross_entropy(old_logits, labels.reshape(-1), ignore_index=-100, reduction='none')
        ratio = torch.exp(old_ce - ce)  # pnew / pold Note: ce = - (logp)
        surr1 = ratio * advantage.reshape(-1)  # surrogate from conservative policy iteration
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage.reshape(-1)  #
        # lt = torch.lt(surr1, surr2)
        # print(lt)
        # print(ratio)
        # print(surr1 <= surr2)

        pol_surr = torch.min(surr1, surr2) # PPO's pessimistic surrogate (L^CLIP)

        # loss = loss * advantage
        # print('policy surrogate:')
        # print(pol_surr)
        # print('adv:')
        # print(advantage)

        loss = pol_surr.sum() / mask.sum()
        return loss

    def predict_get_ppo_loss_v2(self, logits, old_logits, labels, advantage, mask, clip_param, kl_penalty=1, kl_target=1e-2,
                                kl_cutoff_factor=2, kl_cutoff_coef = 1000):
        """
        Implemented according to
        https://github.com/google-research/batch-ppo/blob/master/agents/algorithms/ppo/ppo.py
        """
        bsz, seq_len = labels.size(0), labels.size(1)
        ce = F.cross_entropy(logits, labels.reshape(-1), ignore_index=-100, reduction='none')
        old_ce = F.cross_entropy(old_logits, labels.reshape(-1), ignore_index=-100, reduction='none')
        ratio = torch.exp(old_ce - ce)  # pnew / pold Note: ce = - (logp)
        surr1 = ratio * advantage.reshape(-1)  # surrogate from conservative policy iteration
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage.reshape(-1)  #
        #pol_surr = torch.min(surr1, surr2) # PPO's pessimistic surrogate (L^CLIP)
        pol_surr = surr1
        surr_loss = pol_surr.sum() / mask.sum()

        kl = F.kl_div(torch.log_softmax(logits, dim=1), torch.softmax(old_logits, dim=1), reduction='none')
        kl = torch.sum(kl, dim=1)
        kl = (kl * mask.reshape(-1)) / mask.sum()
        kl_loss = kl_penalty * kl

        cutoff_threshold = kl_target * kl_cutoff_factor
        cutoff_count = torch.sum(
            (kl > cutoff_threshold).int())
        print(kl)
        print(cutoff_threshold)
        kl_cutoff = (
                kl_cutoff_coef *
                (kl > cutoff_threshold).float() *
                (kl - cutoff_threshold) ** 2)
        print(kl_cutoff)

        #policy_loss = surr_loss + kl_penalty + kl_cutoff

        return surr_loss

    def assign_old_eq_new(self):
        """
        Used in PPO.
        Assign the parameters in the current order_pred_layer to the old_order_pred_layer
        """
        self.old_order_pred_proj_l1.load_state_dict(self.order_pred_proj_l1.state_dict())
        self.old_order_pred_proj.load_state_dict(self.order_pred_proj.state_dict())
        # self.old_order_pred_proj_l1.copy_(self.order_pred_proj_l1.data)
        # self.old_order_pred_proj.copy_(self.order_pred_proj.data)

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            # if 'order_pred_proj' in name:
            #     continue
            try:
                own_state[name].copy_(param)
            except:
                own_state[name[7:]].copy_(param) # get rid of the 'module.' prefix


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
