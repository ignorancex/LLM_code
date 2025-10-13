# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Dict

import torch
from torch import nn

from contextual_sparsity.evaluation.hooks.base import EvaluationHook

CROSS_ENTROPY = "cross_entropy"
PERPLEXITY = "perplexity"


# Perplexity and cross-entropy evaluation
class Perplexity(EvaluationHook):
    """
    Computes the perplexity metric.
    """

    metric_dims = {CROSS_ENTROPY: 1, PERPLEXITY: 1}

    def collect_results(self, module, input, kwargs, output):
        valid_tokens = torch.greater_equal(kwargs["labels"].view(-1), 0).int()
        ntok = torch.sum(valid_tokens)
        return {CROSS_ENTROPY: output.loss.unsqueeze(0).repeat(ntok).unsqueeze(-1)}

    def attach_to(self, model: nn.Module):
        self._attach_to(model)

    def finalize(self, stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        stats["."][PERPLEXITY] = {"mean": torch.exp(stats["."][CROSS_ENTROPY]["mean"])}
        return stats
