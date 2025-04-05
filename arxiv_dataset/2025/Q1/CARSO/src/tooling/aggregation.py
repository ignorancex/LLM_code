#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#  Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import List

import torch as th
from ebtorch.nn import lexsemble
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["select_aggregation"]
# ──────────────────────────────────────────────────────────────────────────────


def _logit_aggregation(x: Tensor) -> Tensor:
    return x.mean(dim=-1)


def _probability_aggregation(x: Tensor) -> Tensor:
    r: Tensor = (1 / (1 + th.exp(-x))).mean(dim=-1)
    return th.log(r / (1 - r))


def _peel_aggregation(x: Tensor) -> Tensor:
    return lexsemble(x)


def select_aggregation(method: str) -> Callable[[Tensor], Tensor]:
    if method == "logit":
        return _logit_aggregation
    elif method == "prob":
        return _probability_aggregation
    elif method == "peel":
        return _peel_aggregation
    else:
        raise ValueError("Invalid aggregation method")
