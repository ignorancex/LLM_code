# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
import itertools
import math
import numpy as np
import time
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
    TransposeLast,
)

import pdb
from .utils import *


class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()
    FIXED = auto()
    CPC = auto()
    BINARY = auto()

@dataclass
class PeakDetectionConfig(FairseqDataclass):
    prominence: float = 0.05

@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.NONE
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    soft_pool_join: bool = False  # Trainable segmenter only
    remove_zeros: bool = False
    in_dim: int = 0  # Trainable segmenter only
    latent_dim: int = 0  # Trainable segmenter only
    n_predicts: int = 1  # CPC segmenter only
    n_negatives: int = 1  # CPC segmenter only
    pos_weight: float = 1.1  # Binary segmenter only
    ignore_value: float = 0.5 # Binary segmenter only
    predictor_type: str = "none"  # CPC segmenter only
    join_before_cpc: bool = True  # CPC segmenter only
    batch_shuffle: bool = False  # CPC segmenter only
    peak_detection: PeakDetectionConfig = PeakDetectionConfig()  # CPC segmenter only


@dataclass
class ESPUM_Config(FairseqDataclass):
    skipgram_size: int = 6
    trigram_size: int = 2

    generator_input_type: str = 'float'
    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False
    generator_classifier: bool = False
    generator_avg_pool_kernel: int = 0
    generator_avg_pool_stride: int = 1

    matching_weight: float = 1.0
    smoothness_weight: float = 0.0
    gradient_penalty: float = 0.0
    code_penalty: float = 0.0
    segment_weight: float = 0.0
    gumbel: bool = False
    hard_gumbel: bool = True
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 128
    hidden_dim: int = 256

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class FixedSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask, bin_labels):
        preds = bin_labels.long().cumsum(-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape
        
        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(
                    return_inverse=True, return_counts=True
                )
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)
            
            new_logits[b].index_add_(
                dim=0, index=idx.to(new_logits.device), source=logits[b]
            )
            new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class CPCSegmenter(JoinSegmenter):
    # Code adapted from https://github.com/felixkreuk/UnsupSeg
    def __init__(self, cfg: SegmentationConfig):
        super().__init__(cfg)
        self.n_predicts = cfg.n_predicts
        self.n_negatives = cfg.n_negatives
        self.batch_shuffle = cfg.batch_shuffle
        self.join_before_cpc = cfg.join_before_cpc
        self.predictor = None
        if cfg.predictor_type == "bilinear":
            self.predictor = nn.Bilinear(cfg.in_dim, cfg.in_dim, 1)

    def score(self, f, b):
        if self.predictor is not None:
            f = f.contiguous()
            b = b.contiguous()
            return self.predictor(f, b).squeeze(-1)
        return (f * b).sum(-1)
    
    def forward(self, logits, padding_mask):
        device = logits.device
        if self.join_before_cpc:
            logits, padding_mask = self.logit_segment(logits, padding_mask)
        z = logits  # self.enc(logits)

        losses = []
        for k in range(1, self.n_predicts+1):
            preds = []
            pos_pred = self.score(z[:, :-k], z[:, k:])
            for _ in range(self.n_negatives):
                time_reorder = torch.randperm(pos_pred.shape[1])
                batch_reorder = torch.arange(pos_pred.shape[0])
                if self.batch_shuffle:
                    batch_reorder = torch.randperm(pos_pred.shape[0])
                neg_pred = self.score(z[:, :-k], z[batch_reorder][:, time_reorder])
                preds.append(neg_pred)
            preds.append(pos_pred)

            out = torch.stack(preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            losses.append(- out[...,-1] * (1. - padding_mask[...,:-k].float()))

        assert losses[0].ndim == 2
        out = torch.stack(outs, dim=1)
        loss = torch.stack(losses, dim=1).sum(1)
        return out, loss


class BinarySegmenter(Segmenter):
    def __init__(self, cfg: SegmentationConfig):
        super().__init__(cfg)
        self.classifier = nn.Sequential(
            # nn.LogSoftmax(dim=-1),
            TransposeLast(),
            nn.Conv1d(cfg.in_dim, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1, stride=1, padding="same"),
            TransposeLast(),
        )
        self.pos_weight = cfg.pos_weight
        self.ignore_value = cfg.ignore_value

    def forward(self, inputs, padding_mask, labels=None):
        logits, feats = inputs
        preds = self.classifier(feats)
        preds = preds[...,0]
        preds[padding_mask] = 0.0
        if labels is not None:
            loss = self.binary_cross_entropy_with_logits(
                preds, labels,
                padding_mask,
                pos_weight=self.pos_weight,
                ignore_value=self.ignore_value,
            )
            return preds, loss
        return preds

    def binary_cross_entropy_with_logits(
            self, input,
            target,
            padding_mask,
            pos_weight=1.0,
            ignore_value=0.5,
        ):
        pos_weight = pos_weight * input.new_ones(1)
        loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none", pos_weight=pos_weight,
        )
        loss[padding_mask] = 0.
        if self.ignore_value is not None: 
            loss[target == ignore_value] = 0.
        loss = loss.sum()
        return loss

    def logit_segment(self, inputs, padding_mask):
        logits, feats = inputs
        pred_scores = self(inputs, padding_mask)
        preds = (pred_scores > 0).long().cumsum(-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(
                    return_inverse=True, return_counts=True
                )
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join: 
                if self.cfg.soft_pool_join:
                    # compute soft counts 
                    sc = pred_scores[b].sigmoid().cumsum(-1)
                    
                    # construct pooling matrix of size (N_seg, T)
                    ns = u.numel()
                    w = torch.arange(ns).repeat(tsz, 1).t()
                    w = w.to(pred_scores.device)
                    w = - torch.abs(w - sc) * 10.
                    w[padding_mask[b].repeat(ns, 1)] = -1e14
                    w = torch.softmax(w, dim=-1)
                    
                    # perform pooling on logits
                    new_logits[b, :ns] = torch.mm(w, logits[b])
                else:
                    u[0] = 0
                    u[1:] = c.cumsum(0)[:-1]
                    m = c > 1
                    r = torch.rand(m.sum())
                    o = (c[m] * r).long()
                    u[m] += o
                    new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.FIXED: FixedSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
    SegmentationType.BINARY: BinarySegmenter,
}


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: ESPUM_Config):
        super().__init__()

        self.cfg = cfg
        self.input_type = cfg.generator_input_type
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)

        padding = cfg.generator_kernel // 2
        if self.input_type == "int":
            self.proj = nn.Embedding(
                input_dim,
                output_dim,
            )
        else:
            self.proj = nn.Sequential(
                TransposeLast(),
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size=cfg.generator_kernel,
                    stride=cfg.generator_stride,
                    dilation=cfg.generator_dilation,
                    padding="same",
                    bias=cfg.generator_bias,
                ),
                TransposeLast(),
            )

    def forward(self, dense_x, tokens, dense_padding_mask):
        if self.input_type == 'float':
            dense_x = self.dropout(dense_x)
            dense_x = self.proj(dense_x)
        else:
            dense_x = self.proj(dense_x.squeeze(-1).long())
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)
            assert (
                diff > 0
            ), f'{new_padding.shape}, {dense_padding_mask.shape}, {dense_x.shape}, {diff}'
            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        result = {}

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result['dense_x'] = dense_x
        result['token_x'] = token_x
        result['dense_padding_mask'] = dense_padding_mask
        return result


@register_model('espum', dataclass=ESPUM_Config)
class ESPUM(BaseFairseqModel):
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_skip_sizes(self, k, r=2):
        return itertools.product(range(1, k+1), repeat=2)

    def discrim_step(self, num_updates):
        return False

    def segment_step(self, num_updates):
        return num_updates < 0

    def get_groups_for_update(self, num_updates):
        return {'generator'}

    def __init__(self, cfg: ESPUM_Config, target_dict):
        super().__init__()

        self.cfg = cfg
        self.nspecial = target_dict.nspecial
        self.skip_size = cfg.skipgram_size
        self.tri_size = cfg.trigram_size

        self.matching_weight = cfg.matching_weight
        self.smoothness_weight = cfg.smoothness_weight
        self.segment_weight = cfg.segment_weight

        d = cfg.input_dim
        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.blank_index = 0
        assert self.blank_index != target_dict.unk()

        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)
        self.generator = Generator(d, output_size, cfg)
        for p in self.generator.parameters():
            p.param_group = 'generator'

        for p in self.segmenter.parameters():
            p.param_group = 'generator'

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0
        self.epoch = 0
        self.forward_counts = torch.nn.Parameter(torch.zeros(1))
        self.forward_counts.requires_grad = False

        self.max_text_positions = 90
        self.saved_token_skipgram = torch.nn.Parameter(torch.zeros(self.skip_size, len(target_dict), len(target_dict)))
        self.saved_token_skipgram.requires_grad = False
        self.saved_dense_skipgram = torch.nn.Parameter(torch.zeros(self.skip_size, len(target_dict), len(target_dict)))
        self.saved_dense_skipgram.requires_grad = False

        self.saved_token_trigram = torch.nn.Parameter(
            torch.zeros(self.tri_size, self.tri_size, len(target_dict), len(target_dict), len(target_dict)))
        self.saved_token_trigram.requires_grad = False
        self.saved_dense_trigram = torch.nn.Parameter(
            torch.zeros(self.tri_size, self.tri_size, len(target_dict), len(target_dict), len(target_dict)))
        self.saved_dense_trigram.requires_grad = False

        self.saved_token_unigram = torch.nn.Parameter(torch.zeros(self.max_text_positions, len(target_dict)))
        self.saved_token_unigram.requires_grad = False
        self.saved_dense_unigram = torch.nn.Parameter(torch.zeros(self.max_text_positions, len(target_dict)))
        self.saved_dense_unigram.requires_grad = False
        self.output_size = output_size

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output['logits']

        padding = net_output['padding_mask']
        if padding.any():
            logits[padding] = float('-inf')
            logits[padding][..., self.blank_index] = float('inf')

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        if self.training and self.gumbel:
            dense_x = F.gumbel_softmax(
                dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
            ).type_as(dense_x)
        else:
            dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        token_x=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
        clus_features=None,
        bin_labels=None,
        gt_bin_labels=None,
    ):
        if self.training and self.update_num % 2 == 0:
            if self.forward_counts[0] == 0:
                self.saved_token_skipgram.data = torch.zeros(self.skip_size, self.output_size, self.output_size).to(features.device)
                self.saved_dense_skipgram.data = torch.zeros(self.skip_size, self.output_size, self.output_size).to(features.device)

                self.saved_token_trigram.data = torch.zeros(
                    self.tri_size, self.tri_size, self.output_size, self.output_size, self.output_size
                ).to(features.device)
                self.saved_dense_trigram.data = torch.zeros(
                    self.tri_size, self.tri_size, self.output_size, self.output_size, self.output_size
                ).to(features.device)

                self.saved_token_unigram.data = torch.zeros(self.max_text_positions, self.output_size).to(features.device)
                self.saved_dense_unigram.data = torch.zeros(self.max_text_positions, self.output_size).to(features.device)

            self.forward_counts[0] = self.forward_counts[0] + 1
        else:
            self.forward_counts[0] = 0

        if clus_features is not None:
            gen_result = self.generator(clus_features, random_label, padding_mask)
        else:
            gen_result = self.generator(features, random_label, padding_mask)

        orig_dense_x, token_x = gen_result['dense_x'], gen_result['token_x']
        orig_dense_padding_mask = gen_result['dense_padding_mask']

        if segment:
            if self.cfg.segmentation.type == SegmentationType.FIXED:
                dense_x, dense_padding_mask = self.segmenter.logit_segment(
                    orig_dense_x, orig_dense_padding_mask, gt_bin_labels,
                )
            elif self.cfg.segmentation.type == SegmentationType.BINARY:
                dense_x, dense_padding_mask = self.segmenter.logit_segment(
                    (orig_dense_x, features), orig_dense_padding_mask,
                )
            else:
                dense_x, dense_padding_mask = self.segmenter.logit_segment(
                    orig_dense_x, orig_dense_padding_mask,
                )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x
        prob_perplexity = None
        code_perplexity = None

        if not dense_x_only:
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits)

        if dense_x_only:
            if self.cfg.segmentation.type == SegmentationType.BINARY:
                bin_scores = self.segmenter(
                    (orig_dense_x, features), orig_dense_padding_mask
                )
                return {
                    'logits': dense_x,
                    'padding_mask': dense_padding_mask,
                    'bin_scores': bin_scores,
                }
            return {'logits': dense_x, 'padding_mask': dense_padding_mask}

        token_padding_mask = random_label == self.pad

        bsz, tsz, dsz = dense_x.size()
        dense_x = dense_x * (1 - dense_padding_mask.unsqueeze(-1).float())
        token_x = token_x[:, :self.max_text_positions]
        token_padding_mask = token_padding_mask[:, :self.max_text_positions]
        try:
            if dense_x.size(1) > token_x.size(1):
                gap = dense_x.size(1) - token_x.size(1)
                pad = dense_x.new_zeros(bsz, gap, dsz)
                token_x = torch.cat((token_x, pad), dim=1)
                pad[:, :, self.pad] = 1.0
                token_padding_mask = torch.cat(
                    (
                        token_padding_mask, 
                        dense_x.new_ones(bsz, gap).bool(),
                    ), dim=1,
                )
            elif dense_x.size(1) < token_x.size(1):
                gap = token_x.size(1) - dense_x.size(1)
                pad = dense_x.new_zeros(bsz, gap, dsz)
                dense_logits = torch.cat((dense_logits, pad), dim=1)
                pad[:, :, self.pad] = 1.0
                dense_x = torch.cat((dense_x, pad), dim=1)
                dense_padding_mask = torch.cat(
                    (
                        dense_padding_mask, 
                        dense_x.new_ones(bsz, gap).bool(),
                    ), dim=1,
                )
        except:
            random_label = dense_x.new_zeros(
                dense_x.size(0), dense_x.size(1),
            )
            token_x = dense_x.new_zeros(*dense_x.size())
            dense_padding_mask = dense_padding_mask.new_ones(
                *dense_padding_mask.size(),
            )
            token_padding_mask = dense_padding_mask

        skip_dense_x = []
        for skip in range(1, self.skip_size+1):
            count_dense_x = torch.mm(
                dense_x[:, :-skip].reshape(-1, dsz).t(), 
                dense_x[:, skip:].reshape(-1, dsz),
            )
            skip_dense_x.append(count_dense_x)
            if self.training and self.update_num % 2 == 0:
                self.saved_dense_skipgram.data[skip - 1, :, :] = (
                    self.saved_dense_skipgram.data[skip - 1, :, :] + count_dense_x
                )

        if self.skip_size > 0:
            skip_dense_x_for_loss = torch.stack(skip_dense_x)

        # Trigram
        tri_dense_x = [[] for _ in range(self.tri_size)]
        for skip1, skip2 in self.get_skip_sizes(self.tri_size):
            # (B x (T - S), D, D)
            count_dense_x = (
                dense_x[:, :-skip1-skip2].reshape(-1, dsz, 1) * 
                dense_x[:, skip1:-skip2].reshape(-1, 1, dsz)
            )
            
            # (D, D, D)
            count_dense_x = torch.matmul(
                count_dense_x.permute(1, 2, 0),
                dense_x[:, skip1+skip2:].reshape(-1, dsz).unsqueeze(0),
            )
            tri_dense_x[skip1-1].append(count_dense_x)
            if self.training and self.update_num % 2 == 0:
                self.saved_dense_trigram.data[skip1 - 1, skip2 - 1, :, :, :] = (
                    self.saved_dense_trigram.data[skip1 - 1, skip2 - 1, :, :, :] + count_dense_x
                )

        if self.tri_size > 0:
            tri_dense_x_for_loss = torch.stack(
                [torch.stack(tri_dense_x[i]) for i in range(self.tri_size)]
            )

        bsz, tsz = random_label.size()
        token_x = token_x * (1 - token_padding_mask.unsqueeze(-1).float())
        for skip in range(1, self.skip_size+1):
            count_token_x = torch.mm(
                token_x[:, :-skip].reshape(-1, dsz).t(),
                token_x[:, skip:].reshape(-1, dsz),
            )
            if self.training and self.update_num % 2 == 0:
                self.saved_token_skipgram.data[skip - 1, :, :] = (
                    self.saved_token_skipgram.data[skip - 1, :, :] + count_token_x
                )

        for skip1, skip2 in self.get_skip_sizes(self.tri_size):
            # (B x (T - S), D, D)
            count_token_x = (
                token_x[:, :-skip1-skip2].reshape(-1, dsz, 1) * 
                token_x[:, skip1:-skip2].reshape(-1, 1, dsz)
            )

            # (D, D, D)
            count_token_x = torch.matmul(
                count_token_x.permute(1, 2, 0),
                token_x[:, skip1+skip2:].reshape(-1, dsz).unsqueeze(0),
            )
            if self.training and self.update_num % 2 == 0:
                self.saved_token_trigram.data[skip1 - 1, skip2 - 1, :, :, :] = (
                    self.saved_token_trigram.data[skip1 - 1, skip2 - 1, :, :, :] + count_token_x
                )
            
        sample_size = dense_x.size(0)

        smoothness_loss = None
        code_pen = None
        grad_pen = None
        loss_token = None
        loss_dense = None
        segment_loss = None
        count_dense_uni = F.pad(
            dense_x.sum(0), 
            (0, 0, 0, self.max_text_positions - dense_x.size(1)), 
            mode='constant', value=0,
        )
        count_token_uni = F.pad(
            token_x.sum(0),
            (0, 0, 0, self.max_text_positions - token_x.size(1)),
            mode='constant', value=0,
        )

        s_step = self.segment_step(self.update_num)
        if self.training and self.update_num % 2 == 0:
            self.saved_token_unigram.data = self.saved_token_unigram.data + count_token_uni
            self.saved_dense_unigram.data = self.saved_dense_unigram.data + count_dense_uni

        if s_step:
            loss_dense = count_dense_uni - count_dense_uni
        else:
            if self.update_num % 2 == 1:
                loss_dense = F.l1_loss(
                    self.saved_dense_unigram - count_dense_uni.detach() + count_dense_uni,
                    self.saved_token_unigram, reduction='sum',
                )
            else:
                loss_dense = count_dense_uni - count_dense_uni

            if self.skip_size > 0:
                if self.update_num % 2 == 1:
                    loss_dense += F.l1_loss(
                        self.saved_dense_skipgram - skip_dense_x_for_loss.detach() + skip_dense_x_for_loss,
                        self.saved_token_skipgram, reduction='sum',
                    )

            if self.tri_size > 0:
                if self.update_num % 2 == 1:
                    loss_dense += F.l1_loss(
                        self.saved_dense_trigram - tri_dense_x_for_loss.detach() + tri_dense_x_for_loss,
                        self.saved_token_trigram, reduction="sum",
                    )
            loss_dense = loss_dense * self.matching_weight

            if self.smoothness_weight > 0:
                smoothness_loss = F.mse_loss(
                    dense_logits[:, :-1], dense_logits[:, 1:], reduction='none'
                )
                smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )

        if self.segment_weight > 0:
            if self.cfg.segmentation.type == SegmentationType.BINARY:
                _, segment_loss = self.segmenter(
                    (orig_dense_x, features), orig_dense_padding_mask, labels=bin_labels,
                )
            else:
                _, segment_loss = self.segmenter(
                    (orig_dense_x, features), orig_dense_padding_mask,
                )

            segment_loss = segment_loss.sum() * self.segment_weight

        result = {
            'losses': {
                'grad_pen': grad_pen,
                'code_pen': code_pen,
                'smoothness': smoothness_loss,
                'segment_loss': segment_loss,
            },
            'temp': self.curr_temp,
            'code_ppl': code_perplexity,
            'prob_ppl': prob_perplexity,
            'd_steps': 0,
            'sample_size': sample_size,
        }

        suff = '_g'
        result['losses']['dense' + suff] = loss_dense
        result['losses']['token' + suff] = loss_token
        return result
