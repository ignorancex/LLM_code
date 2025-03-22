import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import BIOTEmbedding
import numpy as np
import random
from layers.Augmentation import get_augmentation


class Model(nn.Module):
    """
    BIOT
    Paper link: https://openreview.net/forum?id=c2LZyTyddi
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.patch_len = configs.patch_len
        stride = configs.patch_len

        augmentations = ["mask", "channel"]  # same as default in the paper
        # Embedding
        self.enc_embedding = BIOTEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            configs.patch_len,
            stride,
            augmentations,
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

        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "supervised" or self.task_name == "finetune":
            self.projection = nn.Linear(
                configs.d_model
                * int((configs.seq_len - configs.patch_len) / stride + 2)
                * configs.enc_in,
                configs.num_class
            )
        elif self.task_name == "pretrain_biot":
            # use for simclr contrastive pretraining
            input_dim = ((configs.d_model
                         * int((configs.seq_len - configs.patch_len) / stride + 2))
                         * configs.enc_in)
            self.projection_head = nn.Linear(input_dim, configs.d_model)
            # BIOT use an extra predictor for contrastive pretraining, like the BYOL model
            self.predictor = nn.Linear(configs.d_model, configs.d_model)

    def supervised(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, (patch_num * enc_in) * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def pretrain(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, (patch_num * enc_in) * d_model)

        repr_out = self.projection_head(output)  # (batch_size, repr_len)
        repr_out = self.act(repr_out)
        repr_out = self.dropout(repr_out)
        repr_out_tilde = self.predictor(repr_out)
        return enc_out, repr_out, repr_out_tilde  # first for downstream tasks, second and third for contrastive loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        elif self.task_name == "pretrain_biot":
            repr_h, repr_z, repr_z_tilde = self.pretrain(x_enc, x_mark_enc)
            return repr_h, repr_z, repr_z_tilde
        else:
            raise ValueError("Task name not recognized or not implemented within the Transformer model")
