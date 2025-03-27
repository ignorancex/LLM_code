import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from src.utilities.config_new import global_config as cfg


class Ross(nn.Module):
    def __init__(self, embedding_size, hidden_size, state_hidden_size, hidden_one_size, hidden_two_size, n_act, n_usr_act, dropout, voc, slots, gating_dict):
        super(Ross, self).__init__()
        self.dropout = dropout
        self.voc = voc
        self.idx2w = {idx: word for word, idx in self.voc.items()}
        self.n_re = len(cfg.RegularExpressions)
        self.slots = slots
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.n_act = n_act
        vocab_size = len(voc)

        self.encoder = EncoderRNN(vocab_size, hidden_size, dropout)
        self.state_encoder = StateEncoderRNN(vocab_size, self.encoder.embedding, hidden_size, dropout)
        self.hidden_transfer = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = Generator(self.idx2w, self.encoder.embedding, vocab_size, hidden_size, self.dropout, self.slots, self.nb_gate)
        self.hidden2intent = nn.Linear(hidden_size, n_usr_act)
        self.state_rnn = nn.GRU(hidden_size + n_act, state_hidden_size, batch_first=True)
        self.policy = Policy(vocab_size, embedding_size, self.n_re, state_hidden_size, hidden_one_size, hidden_two_size, n_act)

    def forward(self, dlg_his, dlg_his_len, last_sys_state, last_sys_state_len, last_state_chunk,
                sys_state, last_act, slot_temps, last_state_hidden, reg_exp, use_teacher_forcing, device=torch.device('cpu')):
        self.device = device
        if use_teacher_forcing is None:
            use_teacher_forcing = random.random() < 0.5
        # Encode dialog history
        utterance_encoded_outputs, utterance_encoded_hidden = self.encoder(dlg_his.transpose(0, 1), dlg_his_len, device=device)

        state_encoded_outputs, state_encoded_hidden = self.state_encoder(last_sys_state, last_sys_state_len,
                                                       last_state_chunk, device=device)
        encoded_hidden = self.hidden_transfer(torch.cat((utterance_encoded_hidden, state_encoded_hidden), dim=2))
        max_res_len = cfg.MAX_STATE_LEN

        # Get the words that can be copy from the memory
        batch_size = dlg_his.size(0)
        all_point_outputs, _, words_point_out, words_class_out = self.decoder(batch_size,
            encoded_hidden, utterance_encoded_outputs, dlg_his_len, state_encoded_outputs, last_sys_state_len,
            dlg_his, last_sys_state, max_res_len,
            sys_state, use_teacher_forcing, slot_temps, device
        )
        intent_tag = self.hidden2intent(encoded_hidden).view(batch_size, -1)

        action = F.one_hot(last_act, num_classes=self.n_act).to(self.device)
        encoded_hidden = torch.cat((encoded_hidden.transpose(0, 1), action), dim=-1)
        state_out, state_hidden = self.state_rnn(encoded_hidden, last_state_hidden)

        policy, slot, value = self.policy(action, state_out, sys_state, reg_exp)

        return intent_tag, all_point_outputs, words_point_out, words_class_out, state_hidden, policy, slot, value

    def nlu(self, dlg_his, dlg_his_len, last_sys_state, last_sys_state_len, last_state_chunk, slot_temps, device):
        # Encode dialog history
        utterance_encoded_outputs, utterance_encoded_hidden = self.encoder(dlg_his.transpose(0, 1), dlg_his_len, device=device)

        state_encoded_outputs, state_encoded_hidden = self.state_encoder(last_sys_state, last_sys_state_len,
                                                                         last_state_chunk, device=device)
        encoded_hidden = self.hidden_transfer(torch.cat((utterance_encoded_hidden, state_encoded_hidden), dim=2))
        max_res_len = cfg.MAX_STATE_LEN

        # Get the words that can be copy from the memory
        batch_size = dlg_his.size(0)
        all_point_outputs, _, words_point_out, words_class_out = self.decoder(batch_size,
                                                                              encoded_hidden, utterance_encoded_outputs,
                                                                              dlg_his_len, state_encoded_outputs,
                                                                              last_sys_state_len,
                                                                              dlg_his, last_sys_state, max_res_len,
                                                                              None, False,
                                                                              slot_temps, device
                                                                              )
        intent_tag = self.hidden2intent(encoded_hidden).view(batch_size, -1)
        return intent_tag, all_point_outputs, words_point_out, encoded_hidden

    def policy_predict(self, last_act, encoded_hidden, last_state_hidden, sys_state, reg_exp, device=torch.device('cpu')):
        action = F.one_hot(last_act, num_classes=self.n_act).to(device)
        # encoded_hidden = torch.cat((encoded_hidden.transpose(0, 1), action), dim=-1)
        encoded_hidden = torch.cat((encoded_hidden.transpose(0, 1), action), dim=-1)
        state_out, state_hidden = self.state_rnn(encoded_hidden, last_state_hidden)
        # state_hidden = state_hidden.detach()
        policy, slot, value = self.policy(action, state_out, sys_state, reg_exp)
        return policy, slot, state_hidden, value

class Policy(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_re, state_hidden_size, hidden_one_size, hidden_two_size, n_act):
        super(Policy, self).__init__()
        # The linear layer that maps from hidden state space to hidden layer
        self.sta2hidden_one = nn.Linear(state_hidden_size, hidden_one_size)
        self.hidden_one2hidden_two = nn.Linear(hidden_one_size, hidden_two_size)
        self.state_embedding = nn.Embedding(vocab_size, embedding_size)
        self.memory_rnn = nn.GRU(embedding_size, hidden_two_size, batch_first=True)
        self.re2hidden_two = nn.Linear(n_re, hidden_two_size)
        self.last_act2hidden = nn.Linear(n_act, hidden_two_size)

        # The linear layer that maps from hidden space to actor and critic
        self.hidden2hidden = nn.Linear(hidden_two_size * 4, hidden_two_size)
        self.hidden2policy = nn.Linear(hidden_two_size, n_act)
        self.hidden2slot = nn.Linear(hidden_two_size * 4 + n_act, cfg.max_chunk)
        self.hidden2vhidden = nn.Linear(hidden_two_size, hidden_two_size)
        self.hidden2value = nn.Linear(hidden_two_size, 1)

    def forward(self, last_act, state_out, belief_state, re_tensor):
        hidden_one = F.relu(self.sta2hidden_one(torch.squeeze(state_out, dim=1)))

        # memory_embedding
        state_memory = self.state_embedding(belief_state)
        B, L, _ = state_memory.size()
        _, state_memory = self.memory_rnn(state_memory)

        # res
        re_hidden = self.re2hidden_two(re_tensor.float())
        last_a_hidden = self.last_act2hidden(last_act.float())
        hidden_two = F.relu(self.hidden_one2hidden_two(hidden_one))
        hidden = torch.cat((last_a_hidden.squeeze(1), hidden_two, state_memory.squeeze(0), re_hidden), dim=-1)
        hidden2 = self.hidden2hidden(hidden)
        policy = self.hidden2policy(hidden2)
        # slot_hidden = torch.cat((hidden, F.softmax(policy, dim=-1)), dim=-1).detach()
        slot_hidden = torch.cat((hidden, F.softmax(policy, dim=-1)), dim=-1)
        slot = self.hidden2slot(slot_hidden)
        # slot = slot.view(-1, cfg.max_chunk, 2)
        slot = slot.unsqueeze(-1)
        slot = torch.cat([1 - slot, slot], -1)
        value = self.hidden2vhidden(hidden2)
        value = self.hidden2value(value)
        return policy, F.softmax(slot, dim=-1), value


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

    def get_state(self, bsz):
        """Get cell states and hidden states."""

        return Variable(torch.zeros(2, bsz, self.hidden_size)).to(self.device)

    def forward(self, input_seqs, input_lengths, hidden=None, device=torch.device('cpu')):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        self.device = device
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False, enforce_sorted=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class StateEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_size, dropout, n_layers=1):
        super(StateEncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.token_embedding = embedding
        self.chunk_embedding = nn.Embedding(6, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)

    def get_state(self, bsz):
        """Get cell states and hidden states."""

        return Variable(torch.zeros(1, bsz, self.hidden_size)).to(self.device)

    def forward(self, state, state_len, state_chunk_ids, hidden=None, device=torch.device('cpu')):
        """

        Args:
            input_seqs: the user utterance + state
            state: list, state sequence
        Returns:
            outputs: [batch, 1, len, hz]
            hidden: [1, batch, len]
        """
        self.device = device

        # state embedding
        state_token_embedded = self.token_embedding(state)
        state_chunk_embedded = self.chunk_embedding(state_chunk_ids)
        state_embedded = state_token_embedded + state_chunk_embedded
        state_embedded = state_embedded.transpose(0, 1).contiguous()
        hidden = self.get_state(state.size(0))
        state_embedded = nn.utils.rnn.pack_padded_sequence(state_embedded, state_len, batch_first=False,
                                                           enforce_sorted=False)
        outputs, hidden = self.gru(state_embedded, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)

        return outputs.transpose(0, 1), hidden


class Generator(nn.Module):
    def __init__(self, idx2w, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.idx2w = idx2w
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(4 * hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, utterance_encoded_outputs, utterance_encoded_lens,
                state_encoded_outputs, state_encoded_lens, dlg, last_state, max_res_len, target_batches,
                use_teacher_forcing, slot_temp, device=torch.device('cpu')):

        self.device = device
        if use_teacher_forcing:
            target_batches = F.one_hot(target_batches, num_classes=self.vocab_size)
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size).to(device)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate).to(device)

        # Get the slot embedding
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx).to(device)

                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx).to(device)
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        # Compute pointer-generator output, decoding each (domain, slot) one-by-one
        words_point_out = []
        counter = 0
        for slot in slot_temp:
            hidden = encoded_hidden
            words = []
            slot_emb = slot_emb_dict[slot]
            decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                context_vec, _, context_prob = self.attend(utterance_encoded_outputs, hidden.squeeze(0), utterance_encoded_lens)
                state_vec, _, state_prob = self.attend(state_encoded_outputs, hidden.squeeze(0), state_encoded_lens)
                if wi == 0:
                    all_gate_outputs[counter] = self.W_gate(context_vec)
                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, state_vec, decoder_input], -1)
                pointer_switches = self.softmax(self.W_ratio(p_gen_vec))
                vocab_pointer_switches = pointer_switches[:, 0].unsqueeze(1)
                state_pointer_switches = pointer_switches[:, 1].unsqueeze(1)
                p_context_ptr = torch.zeros(p_vocab.size()).to(device)
                p_state_ptr = torch.zeros(p_vocab.size()).to(device)

                p_context_ptr.scatter_add_(1, dlg, context_prob)
                p_state_ptr.scatter_add_(1, last_state, state_prob)

                final_p_vocab = vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab + \
                                state_pointer_switches.expand_as(p_context_ptr) * p_state_ptr + \
                                (1 - state_pointer_switches - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words.append([self.idx2w[w_idx.item()] for w_idx in pred_word])
                all_point_outputs[counter, :, wi, :] = final_p_vocab
                if use_teacher_forcing:
                    decoder_input = self.embedding(target_batches[:, counter, wi])  # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)
                decoder_input = decoder_input.to(device)
            counter += 1
            words_point_out.append(words)

        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
