"""
model.py

This module defines the architecture of the SELD deep learning model.

Classes:
    ConvBlock: A convolutional block for feature extraction from audio input.
    SELDModel: The main SELD model combining ConvBlock, recurrent, attention, and fusion layers.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
    Designed for feature extraction from audio input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, pool_size=(5, 4), dropout=0.05):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SELDModel(nn.Module):
    """
    SELD (Sound Event Localization and Detection) model combining ConvBlock, recurrent, and attention-based layers.
    Supports audio-only input.
    """
    def __init__(self, params, in_feat_shape):
        super().__init__()
        self.params = params
        self.in_feat_shape = in_feat_shape
        params['nb_mels'] = in_feat_shape[-1] # Hard reset the number of mel bins
        nb_conv_filters = [64] + [params['nb_conv_filters']] * int(params['nb_conv_blocks'] - 1) # Always start with 64 filt

        # Conv layers
        self.conv_blocks = nn.ModuleList()
        for conv_cnt in range(params['nb_conv_blocks']):
            self.conv_blocks.append(ConvBlock(in_channels=nb_conv_filters[conv_cnt - 1] if conv_cnt else self.in_feat_shape[1],  # stereo
                                              out_channels=nb_conv_filters[conv_cnt],
                                              pool_size=(params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]),
                                              dropout=params['dropout']))

        # GRU layers
        self.gru_input_dim = params['nb_conv_filters'] * int(np.floor(params['nb_mels'] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'], num_layers=params['nb_rnn_layers'],
                                batch_first=True, dropout=params['dropout'], bidirectional=True)

        # Self attention layers
        self.mhsa_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=params['rnn_size'], num_heads=params['nb_attn_heads'],
                                                                dropout=params['dropout'], batch_first=True) 
                                          for _ in range(params['nb_self_attn_layers'])])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(params['rnn_size']) for _ in range(params['nb_self_attn_layers'])])

        # Fully Connected layers
        self.fnn_list = torch.nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'], params['fnn_size'], bias=True))
            self.fnn_list.append(nn.Dropout(p=params['dropout']))

        if params['multiACCDOA']:
            self.output_dim = params['max_polyphony'] * 3 * params['nb_classes']  # 3 => (x,y), distance
        else:
            self.output_dim = 3 * params['nb_classes']  # 3 => (x,y), distance
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], self.output_dim, bias=True))

    def forward(self, audio_feat):
        """
        Forward pass for the SELD model.
        audio_feat: Tensor of shape (batch_size, n_feat_ch, 251, 64).
        Returns:  Tensor of shape
                  (batch_size, 50, 117) - audio - multiACCDOA.
                  (batch_size, 50, 39)  - audio - singleACCDOA.
        """
        # audio feat - B x C x 251 x F
        for conv_block in self.conv_blocks:
            audio_feat = conv_block(audio_feat)  # B x 64 x 50 x 2

        audio_feat = audio_feat.transpose(1, 2).contiguous()  # B x 50 x 64 x 2
        audio_feat = audio_feat.view(audio_feat.shape[0], audio_feat.shape[1], -1).contiguous()  # B x 50 x 128

        (audio_feat, _) = self.gru(audio_feat)
        audio_feat = audio_feat[:, :, audio_feat.shape[-1] // 2:] * audio_feat[:, :, :audio_feat.shape[-1] // 2]

        for mhsa, ln in zip(self.mhsa_layers, self.layer_norms):
            audio_feat_in = audio_feat
            audio_feat, _ = mhsa(audio_feat_in, audio_feat_in, audio_feat_in)
            audio_feat = audio_feat + audio_feat_in  # Residual connection
            audio_feat = ln(audio_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            audio_feat = self.fnn_list[fnn_cnt](audio_feat)
        audio_feat = F.relu(audio_feat, inplace=True)
        audio_feat = self.fnn_list[-1](audio_feat)

        return audio_feat