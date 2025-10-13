# coding=utf-8
#
# This file is a modified version of source code made available under the Apache 2.0 license by Google Research made available in their repository at https://github.com/google-research/google-research/blob/1d49f2c0d5506a6e11115726164d42f6b7b7b95a/d3pm/images/utils.py
#
# The primary purpose of the modifications is to make this code work with PyTorch.
# Modifications Copyright (c) 2025 The High-Throughput Toolkit
#
# The original copyright header from the file from Google Research follows below:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F


def categorical_kl_logits(logits1, logits2, eps=1e-6):
    assert logits1.shape == logits2.shape, (logits1.shape, logits2.shape)
    out = torch.softmax(logits1 + eps, dim=-1) * (
        torch.log_softmax(logits1 + eps, dim=-1)
        - torch.log_softmax(logits2 + eps, dim=-1)
    )
    return out.sum(dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1e-6):
    assert probs1.shape == probs2.shape, (probs1.shape, probs2.shape)
    out = probs1 * ((probs1 + eps).log() - (probs2 + eps).log())
    return out.sum(-1)


def categorical_kl_probs_logits(probs, logits, eps=1e-6):
    assert probs.shape == logits.shape, (probs.shape, logits.shape)
    out = probs * ((probs + eps).log() - torch.log_softmax(logits + eps, dim=-1))
    return out.sum(-1)


def categorical_log_likelihood(x, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x, logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


def categorical_log_likelihood_pure_guess(x, probs):
    # log_probs = F.log_softmax(logits, dim=-1)
    log_q_0 = probs.log()
    x_onehot = F.one_hot(x, probs.shape[-1])
    return torch.sum(x_onehot * log_q_0, dim=-1)


def sample_from_distr(p_distr):
    assert len(p_distr.shape) == 2
    return p_distr.multinomial(1)  # TODO: .squeeze()?
