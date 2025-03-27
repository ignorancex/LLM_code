import math, torch
import torch.nn as nn
from einops import repeat

from src.TPP.model.sahp.layers import TransformerLayer
from src.TPP.model.sahp.utils import *
from src.TPP.model.sahp.position import BiasedPositionalEmbedding


class TransformerEncoder(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, num_types, device, d_input, d_rnn, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.num_types = num_types if num_types > 0 else 1

        self.encoder = Encoder(
            num_types = self.num_types,
            d_input = d_input,
            d_hidden = d_hidden,
            n_layers = n_layers,
            n_head = n_head,
            d_qk = d_qk,
            d_v = d_v,
            dropout = dropout,
            device = self.device
        )

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_input, d_rnn, device = self.device)

    def forward(self, event_time, event_type, non_pad_mask):
        """
        Return intensity functions' values for all events and time and events, if possible, predictions.
        Args:
        1. event_time: the length of all time intervals between two adjacent events. shape: [batch_size, seq_len]
        2. event_type: vectors containing the information about each event. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an event, while 0 means a dummy event. shape: [batch_size, seq_len]
        """

        enc_output = self.encoder(event_type, event_time, non_pad_mask)        # [batch_size, seq_len, d_input]
        enc_output = self.rnn(enc_output)                                      # [batch_size, seq_len, d_input]

        return enc_output


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout, 
            device):
        super(Encoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_types = num_types

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_input, padding_idx = num_types, device = self.device)

        self.layer_stack = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])

    def forward(self, event_type, event_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. event_type: 
        2. event_time: input time intervals. shape: [batch_size, seq_len]
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len]
        """

        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        _, seq_len = event_type.shape
        self_attn_mask_subseq = get_subsequent_mask(event_time)
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s s_1', s_1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        # Time Embedding
        time_emb = self.position_emb(event_type, event_time)                   # [batch_size, seq_len, d_input]

        # Event Embedding
        if event_type != None:
            events_emb = self.event_emb(event_type)                            # [batch_size, seq_len, d_input]
        else:
            events_emb = torch.zeros_like(time_emb, device = self.device)      # [batch_size, seq_len, d_input]
        mingled_emb = time_emb + events_emb                                    # [batch_size, seq_len, d_input]

        for enc_layer in self.layer_stack:
            mingled_emb, _ = enc_layer(
                mingled_emb,
                non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]

        return mingled_emb


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, device):
        super(RNN_layers, self).__init__()
        self.device = device

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True, device = self.device)
        self.projection = nn.Linear(d_rnn, d_model, device = self.device)

    def forward(self, data):
        out = self.rnn(data)[0]                                                # [batch_size, seq_len, d_rnn]

        out = self.projection(out)                                             # [batch_size, seq_len, d_model]
        return out