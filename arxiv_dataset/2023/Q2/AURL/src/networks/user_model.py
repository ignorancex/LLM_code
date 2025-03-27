import json
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from src.utilities.config_new import global_config as cfg


class UtteranceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        ntoken = len(json.load(open(cfg.vocab_path, 'r', encoding='utf-8')))
        d_model = cfg.user_embedding_dim
        nhead = cfg.user_nhead
        d_hid = cfg.user_dhid
        nlayers = cfg.user_nlayers
        dropout = cfg.user_dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.hidden2policy = nn.Linear(d_model, len(cfg.sys_act_id2name))
        self.hidden2slot = nn.Linear(d_model, 4)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.hidden2policy.bias.data.zero_()
        self.hidden2policy.weight.data.uniform_(-initrange, initrange)
        self.hidden2slot.bias.data.zero_()
        self.hidden2slot.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        hidden = self.transformer_encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        sys_act = self.hidden2policy(hidden[0])
        sys_slot = self.hidden2slot(hidden[1:4])
        sys_slot = sys_slot.transpose(0, 1)
        return F.softmax(sys_act, dim=-1), F.softmax(sys_slot, dim=-1),  F.log_softmax(sys_act, dim=-1)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Policy_v1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.a_s_embedding = nn.Embedding(len(cfg.sys_act_id2name), cfg.user_embedding_dim)
        # self.slot_embedding = nn.Embedding(4, cfg.user_embedding_dim)
        self.input2hidden = nn.Linear(len(cfg.sys_act_id2name) + cfg.max_chunk * 2, cfg.user_policy_hidden_dim)
        self.hidden2hidden_two = nn.Linear(cfg.user_policy_hidden_dim, cfg.user_policy_hidden_dim)
        self.hidden_two2policy = nn.Linear(cfg.user_policy_hidden_dim, len(cfg.user_act_id2name))
        self.hidden2slot = nn.Linear(cfg.user_policy_hidden_dim*2, cfg.max_chunk * 2)
        self.hidden2value = nn.Linear(cfg.user_policy_hidden_dim, 1)
        self.hidden2qvalue = nn.Linear(cfg.user_policy_hidden_dim, len(cfg.user_act_id2name))

    def forward(self, manager_act, last_state, slot_input):
        # manager_act = self.a_s_embedding(manager_act)
        # last_state = self.slot_embedding(last_state)
        # slot_input = self.slot_embedding(slot_input)
        all_input = torch.cat((manager_act, last_state, slot_input), dim=1).float()
        hidden = F.relu(self.input2hidden(all_input))
        hidden2 = F.relu(self.hidden2hidden_two(hidden))
        user_slot = self.hidden2slot(torch.cat((hidden,hidden2),dim=-1))
        user_slot = user_slot.view(-1, cfg.max_chunk, 2)
        # user_slot = self.hidden2slot(hidden2[:,1:4,:])
        user_act = self.hidden_two2policy(hidden2)
        value = self.hidden2value(hidden2)
        q_value = self.hidden2qvalue(hidden2)
        return F.softmax(user_slot, dim=-1), F.softmax(user_act, dim=-1), F.log_softmax(user_act, dim=-1), value, q_value


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        dropout = cfg.user_dropout
        self.a_u_embedding = nn.Embedding(len(cfg.user_act_id2name), cfg.user_policy_embedding_dim)
        self.a_s_embedding = nn.Embedding(len(cfg.sys_act_id2name), cfg.user_policy_embedding_dim)
        self.slot_embedding = nn.Embedding(4, cfg.user_policy_embedding_dim)
        self.input2hidden = nn.Linear(cfg.user_policy_embedding_dim * 3, cfg.user_policy_hidden_dim)

        self.act2hidden = nn.Linear(len(cfg.sys_act_id2name), cfg.user_policy_hidden_dim)
        self.slot2hidden = nn.Linear(cfg.max_chunk * 2, cfg.user_policy_hidden_dim)

        if cfg.user_net_type == 'RNN':
            self.rnn = nn.RNN(cfg.user_policy_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)
        elif cfg.user_net_type == 'GRU':
            self.rnn = nn.GRU(cfg.user_policy_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(cfg.user_policy_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)

        self.slot_policy = nn.Linear(cfg.user_state_hidden_dim, 2)

        # self.input2hidden = nn.Linear(cfg.user_policy_hidden_dim * 2, cfg.user_policy_hidden_dim)
        self.input2hidden = nn.Linear(cfg.user_state_hidden_dim, cfg.user_policy_hidden_dim)
        self.hidden2hidden_two = nn.Linear(cfg.user_policy_hidden_dim, cfg.user_policy_hidden_dim)
        self.hidden_two2policy = nn.Linear(cfg.user_policy_hidden_dim, len(cfg.user_act_id2name))
        self.hidden2slot = nn.Linear(len(cfg.user_act_id2name) + cfg.user_state_hidden_dim, cfg.max_chunk)
        self.hidden2vhidden = nn.Linear(cfg.user_policy_hidden_dim, cfg.user_state_hidden_dim)
        self.hidden2value = nn.Linear(cfg.user_state_hidden_dim, 1)
        self.hidden2qvalue = nn.Linear(cfg.user_state_hidden_dim, len(cfg.user_act_id2name))
        # self.init_weights()
        self.dropout = nn.Dropout(p=dropout)

    def init_weights(self) -> None:
        initrange = 0.1
        for m in self.children():
            m.bias.data.zero_()
            m.weight.data.uniform_(-initrange, initrange)
        # initrange = 0.1
        # self.act2hidden.weight.data.uniform_(-initrange, initrange)
        # self.hidden2policy.bias.data.zero_()
        # self.hidden2policy.weight.data.uniform_(-initrange, initrange)
        # self.hidden2slot.bias.data.zero_()
        # self.hidden2slot.weight.data.uniform_(-initrange, initrange)

    def forward(self, last_act, manager_act, last_state, slot_input, last_hidden_state=None):
        last_act = self.a_u_embedding(last_act)
        manager_act = self.a_s_embedding(manager_act)
        last_state = self.slot_embedding(last_state)
        slot_input = self.slot_embedding(slot_input)

        all_input = torch.cat((last_act, manager_act, slot_input, last_state), dim=1)
        _, hidden = self.rnn(all_input, last_hidden_state)
        hidden = hidden.squeeze(0)
        hidden2 = F.relu(self.input2hidden(hidden))
        user_act = self.hidden_two2policy(hidden2)
        slot_hidden = torch.cat((hidden, F.softmax(user_act, dim=-1)), dim=-1)
        user_slot = self.hidden2slot(slot_hidden)
        user_slot = user_slot.unsqueeze(-1)
        user_slot = torch.cat([1 - user_slot, user_slot], -1)
        value = self.hidden2vhidden(hidden2)
        value = self.hidden2value(value)
        q_value = self.hidden2qvalue(hidden)
        return F.softmax(user_act, dim=-1), F.softmax(user_slot, dim=-1),  value, q_value, hidden.unsqueeze(0)


class Policy2(nn.Module):
    def __init__(self):
        super(Policy2, self).__init__()
        sim_action_num = len(cfg.user_act_id2name)
        manager_action_num = len(cfg.sys_act_id2name)
        net_type = cfg.user_net_type

        self.hidden_size = cfg.user_policy_hidden_dim
        self.a_s_embedding = nn.Embedding(manager_action_num, cfg.user_embedding_dim)
        self.a_u_embedding = nn.Embedding(sim_action_num, cfg.user_embedding_dim)
        self.slot_embedding = nn.Embedding(cfg.max_chunk, cfg.user_embedding_dim)
        if net_type == 'RNN':
            self.rnn = nn.RNN(cfg.user_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)
        elif net_type == 'GRU':
            self.rnn = nn.GRU(cfg.user_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(cfg.user_embedding_dim, cfg.user_state_hidden_dim, batch_first=True)
        self.state_hidden2hidden = nn.Linear(cfg.user_state_hidden_dim, cfg.user_policy_hidden_dim)
        self.slot_policy = nn.Linear(cfg.user_state_hidden_dim, 2)
        self.hidden2policy = nn.Linear(cfg.user_policy_hidden_dim, sim_action_num)
        self.hidden2value = nn.Linear(cfg.user_policy_hidden_dim, 1)
        self.hidden2q_value = nn.Linear(cfg.user_policy_hidden_dim, sim_action_num)

    def forward(self, manager_act, last_user_act, slot_input, last_state):
        """
        :param manager_act: [batch, 1]
        :param last_user_act: [batch, 1]
        :param slot_input: [batch, len]
        :return:
        """
        manager_act = self.a_s_embedding(manager_act)
        last_user_act = self.a_u_embedding(last_user_act)
        # print(slot_input)
        slot_input = self.slot_embedding(slot_input)
        # print(slot_input)

        state_input = torch.cat((manager_act, last_user_act, slot_input), dim=1)
        output, hidden = self.rnn(state_input, last_state)
        slot_predict = self.slot_policy(output[:, 2:])
        hidden_one = F.relu(self.state_hidden2hidden(hidden[0])).squeeze(0)
        act_predict = self.hidden2policy(hidden_one)
        value = self.hidden2value(hidden_one)
        q_value = self.hidden2q_value(hidden_one)

        return F.softmax(slot_predict, dim=-1), F.softmax(act_predict, dim=-1), F.log_softmax(act_predict, dim=-1), value, q_value, hidden


class Transformer_model_2in1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        ntoken = len(json.load(open(cfg.vocab_path, 'r', encoding='utf-8')))
        d_model = cfg.user_embedding_dim
        nhead = cfg.user_nhead
        d_hid = cfg.user_dhid
        nlayers = cfg.user_nlayers
        dropout = cfg.user_dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=cfg.user_max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.hidden2sys_act = nn.Linear(d_model, len(cfg.sys_act_id2name))
        self.hidden2sys_slot = nn.Linear(d_model, 4)
        self.hidden_one2hidden_two = nn.Linear(d_model, cfg.user_policy_hidden_dim)
        self.hidden2value = nn.Linear(cfg.user_policy_hidden_dim, 1)
        self.hidden2hidden = nn.Linear(cfg.user_policy_hidden_dim, cfg.user_policy_hidden_dim)
        self.hidden2policy = nn.Linear(cfg.user_policy_hidden_dim, len(cfg.user_act_id2name))
        self.hidden2slot = nn.Linear(cfg.user_policy_hidden_dim, cfg.max_chunk)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.hidden2sys_act.bias.data.zero_()
        self.hidden2sys_act.weight.data.uniform_(-initrange, initrange)
        self.hidden2sys_slot.bias.data.zero_()
        self.hidden2sys_slot.weight.data.uniform_(-initrange, initrange)
        self.hidden2policy.bias.data.zero_()
        self.hidden2policy.weight.data.uniform_(-initrange, initrange)
        self.hidden2slot.bias.data.zero_()
        self.hidden2slot.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        hidden = self.transformer_encoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        sys_act = self.hidden2sys_act(hidden[0])
        sys_slot = self.hidden2sys_slot(hidden[1:4])
        if src_key_padding_mask is not None:
            avg_hidden = hidden.sum(0) / (src_key_padding_mask.shape[1] - src_key_padding_mask.sum(1) + 1e-21).unsqueeze(1)
        else:
            avg_hidden = hidden.sum(0) / hidden.shape[0]
        avg_hidden = avg_hidden.detach()
        hidden2 = F.relu(self.hidden_one2hidden_two(avg_hidden))
        value = self.hidden2value(hidden2)
        hidden22 = F.relu(self.hidden2hidden(hidden2))
        user_act = self.hidden2policy(hidden22)
        user_slot = self.hidden2slot(hidden22)
        user_slot = user_slot.unsqueeze(-1)
        user_slot = torch.cat([(1 - user_slot).detach(), user_slot], -1)
        sys_slot = sys_slot.transpose(0, 1)
        return sys_act, sys_slot,  F.softmax(user_act, dim=-1), F.softmax(user_slot, dim=-1), value