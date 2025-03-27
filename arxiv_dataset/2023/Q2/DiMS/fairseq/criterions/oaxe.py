# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import math
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
import torch
from math import log
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa
from dataclasses import dataclass, field
import copy



@dataclass
class OrderAgnostiCrossEntropyConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    skip: float = field(
        default=.15,
        metadata={"help": "pi for skip margin, pi is in the form of probs, must in (0, 1]"},
    )
    no_padding: bool = field(
        default = False,
        metadata={"help": "if set, replace pad with unk"},
    )


@register_criterion("oaxe", dataclass=OrderAgnostiCrossEntropyConfig)
class OrderAgnostiCrossEntropy(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.eps = args.label_smoothing
        self.eos_idx = task.target_dictionary.eos()
        self.unk_idx = task.target_dictionary.unk()
        self.margin = 0.15
        self.beam_size = 5
        self.args =args

    def forward(self, model, sample, reduce=True, net_output=None, **kwargs):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        self.pad_idx=model.pad
        self.mask_idx=model.unk
        self.padding_idx=model.pad
        sample['net_input']['tgt_tokens']=sample['target']
        if net_output is None:
            sample['net_input']['prev_output_tokens'][sample['net_input']['prev_output_tokens']!=model.pad] = model.unk
            net_output = model(**sample['net_input'])
        
        loss, nll_loss, length_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = ntokens #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return nll_loss, loss, sample_size, logging_output

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        length_lprobs = net_output[1]['predicted_lengths']
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1) #TODO doesn't work for dynamic length. change to eos-based method.
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss

    def compute_loss(self, model, net_output, sample, reduce=True):

        margin = -log(self.margin, 2)
        # print(margin)
        # Length Loss
        
        length_lprobs = torch.log_softmax(net_output['length']['out'], dim=-1)
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1) #TODO doesn't work for dynamic length. change to eos-based method.
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        if reduce:
            length_loss = length_loss.sum()

        # Bipart Loss
        mask_ind = sample['net_input']['prev_output_tokens'].eq(model.unk).unsqueeze(-1)
        target = sample['target']
        bs, seq_len = target.size()
        target = target.repeat(1, seq_len).view(bs, seq_len, seq_len)
        bipart_no_pad = target.ne(self.padding_idx)
        bipart_lprobs = model.get_normalized_probs([net_output['word_ins']['out']], log_probs=True)
        
        nll_loss = -bipart_lprobs.gather(dim=-1, index=target)#bs seq seq
        nll_loss = nll_loss * bipart_no_pad
        
        smooth_lprobs = model.get_normalized_probs([net_output['word_ins']['out']], log_probs=True)
        smooth_lprobs = smooth_lprobs.view(-1, bipart_lprobs.size(-1))
        smooth_loss = -smooth_lprobs.sum(dim=-1, keepdim=True)
        smooth_non_pad_mask = sample['target'].view(-1, 1).ne(self.padding_idx)
        smooth_loss = smooth_loss * smooth_non_pad_mask
        
        best_match = np.repeat(np.arange(seq_len).reshape(1, -1, 1), bs, axis=0)# np.zeros((bs, seq_len, 1))
        nll_loss_numpy = nll_loss.detach().cpu().numpy()

        for batch_id in range(bs):
            no_pad_num = bipart_no_pad[batch_id, 0].sum()
            raw_index, col_index = lsa(nll_loss_numpy[batch_id, :no_pad_num, :no_pad_num])
            best_match[batch_id, :no_pad_num] = col_index.reshape(-1, 1)

        best_match = torch.Tensor(best_match).to(target).long()
        nll_loss = nll_loss * mask_ind
        nll_loss = nll_loss.gather(dim=-1, index=best_match)
        nll_loss = nll_loss.squeeze(-1)
        nll_loss[nll_loss > margin] = 0

        epsilon = 0.1
        eps_i = epsilon / bipart_lprobs.size(-1)
        nll_loss = (1 - epsilon) * nll_loss.mean() + eps_i * smooth_loss.mean()
        loss = nll_loss + length_loss
        return loss, nll_loss, length_loss, mask_ind.sum().data.item()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
            
