import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
    Decoder,
    DecoderLayer,
)
from layers.Embed import EEG2RepEmbedding, PositionalEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from utils.tools import semantic_subsequence_preserving
import copy


class Model(nn.Module):
    """
        EEG2Rep: https://github.com/Navidfoumani/EEG2Rep
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.mask_ratio = configs.mask_ratio
        pooling_size = 2
        # Embedding
        self.PositionalEncoding = PositionalEmbedding(configs.d_model)
        self.enc_embedding = EEG2RepEmbedding(
            configs.enc_in,
            configs.d_model,
            pooling_size,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.norm = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        # Decoder
        if self.task_name == "supervised" or self.task_name == "finetune":
            self.projection = nn.Linear(
                configs.d_model,
                configs.num_class
            )
        elif self.task_name == "pretrain_eeg2rep":
            self.mask_token = nn.Parameter(torch.randn(configs.d_model, ))
            # for cross attention
            self.Predictor = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            self.m_encoder = copy.deepcopy(self.encoder)  # same initialization as model
            for param in self.m_encoder.parameters():
                param.requires_grad = False

    def pretrain(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        enc_out = self.enc_embedding(x_enc)
        output = self.norm(enc_out)
        output = output + self.PositionalEncoding(output)
        output = self.norm2(output)

        rep_mask_token = self.mask_token.repeat(output.shape[0], output.shape[1], 1)
        rep_mask_token = rep_mask_token + self.PositionalEncoding(rep_mask_token)

        index = np.arange(output.shape[1])
        index_chunk = semantic_subsequence_preserving(index, 2, self.mask_ratio)
        v_index = np.ravel(index_chunk)  # visible patches index (may have repeated index)
        m_index = np.setdiff1d(index, v_index)  # mask patches index

        visible = output[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex, _ = self.encoder(visible, attn_mask=None)
        with torch.no_grad():
            rep_target, _ = self.m_encoder(output, attn_mask=None)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_mask_token, rep_contex)  # cross attention
        return rep_mask, rep_mask_prediction

    def supervised(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        output = self.norm(enc_out)
        output = output + self.PositionalEncoding(output)
        output = self.norm2(output)
        output, attns = self.encoder(output, attn_mask=None)
        return self.projection(torch.mean(output, dim=1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        elif self.task_name == "pretrain_eeg2rep":
            repr_m, repr_m_p = self.pretrain(x_enc, x_mark_enc)
            return repr_m, repr_m_p
        return None
