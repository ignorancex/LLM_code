import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SelfAttentionModel(nn.Module):
    def __init__(self, n_features_input, hidden_features_size_before_attention=8, nhead=1, num_layers=1, dropout=0.1,
                 n_class=2, encoder_feedforward_dim=16):
        super(SelfAttentionModel, self).__init__()
        self._model_type = 'TransformerClassification'
        self._src_mask = None
        self._first_layer = nn.Linear(in_features=n_features_input,
                                      out_features=hidden_features_size_before_attention)
        self._hidden_features_size_before_attention = hidden_features_size_before_attention
        self._pos_encoder = PositionalEncoding(d_model=hidden_features_size_before_attention)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_features_size_before_attention, nhead=nhead,
                                                   dropout=dropout, dim_feedforward=encoder_feedforward_dim)
        self._transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._output_layer = nn.Linear(int(hidden_features_size_before_attention), n_class)

        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, src, lengths=None):
        src_expanded = self._first_layer(src) * math.sqrt(self._hidden_features_size_before_attention)
        src_expanded = self._pos_encoder(src_expanded)
        src_expanded_permuted = src_expanded.permute(1, 0, 2)
        if lengths is not None:
            src_key_padding_mask = self._generate_src_key_padding_mask(src, lengths)
            memory = self._transformer_encoder(src_expanded_permuted, src_key_padding_mask=src_key_padding_mask)
        else:
            memory = self._transformer_encoder(src_expanded_permuted)
        memory = memory.permute(1, 0, 2)
        memory = src_expanded + memory
        final_layer = F.adaptive_avg_pool2d(memory, output_size=(1, None)).squeeze(1)
        output = self._output_layer(F.elu(final_layer))

        return output

    def _generate_src_key_padding_mask(self, src, lengths):
        src_key_padding_mask = torch.zeros(src.shape[0], src.shape[1] + 1, dtype=src.dtype, device=src.device)
        src_key_padding_mask[(torch.arange(src.shape[0]), lengths)] = 1
        src_key_padding_mask = src_key_padding_mask.cumsum(dim=1)[:, :-1]  # remove the superfluous column
        src_key_padding_mask = src_key_padding_mask.bool()
        return src_key_padding_mask

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
