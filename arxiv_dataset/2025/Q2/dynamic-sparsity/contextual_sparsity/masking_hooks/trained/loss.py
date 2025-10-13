# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent

from contextual_sparsity.nn import Abs, ThresholdMask, TopKMask


def binary_crossentropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary crossentropy loss function
    """
    return -Independent(Bernoulli(logits=pred), 1).log_prob(target.float()).mean()


class PredictorLoss(nn.Module):
    def __init__(
        self,
        predictor: Callable,
        up_layer: Optional[nn.Module] = None,
        down_layer: Optional[nn.Module] = None,
        gate_layer: Optional[nn.Module] = None,
        act_fn: Optional[nn.Module] = None,
        double_gating: bool = False,
    ):
        super().__init__()
        self.predictor = predictor
        self.double_gating = double_gating

        self.up_layer = up_layer
        if up_layer is not None:
            for param in up_layer.parameters():
                param.requires_grad = False
        self.gate_layer = gate_layer
        if gate_layer is not None:
            for param in gate_layer.parameters():
                param.requires_grad = False
        self.down_layer = down_layer
        if down_layer is not None:
            for param in down_layer.parameters():
                param.requires_grad = False

        self.act_fn = act_fn
        self.model_dtype = self.up_layer.weight.dtype

    def forward(
        self, x: torch.Tensor, x_last: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if "act" not in kwargs:
            if x_last is None:
                x_last = x

            x_last = x_last.to(self.model_dtype)

            up = self.up_layer(x_last.to(self.model_dtype))
            if self.double_gating:
                up = self.act_fn(up)
            kwargs["up"] = up.detach()

            if self.gate_layer is None:
                act = self.act_fn(up)
            else:
                gate = self.act_fn(self.gate_layer(x_last))
                kwargs["gate"] = gate.detach()
                act = up * gate

            kwargs["act"] = act.detach()

        return self._loss(x, **kwargs)

    def _loss(
        self,
        x: torch.Tensor,
        act: torch.Tensor,
        up: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError()


class ActivationLoss(PredictorLoss):
    def __init__(
        self,
        predictor: Callable,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        act_transform: Optional[Callable] = None,
        target: str = "act",
        **kwargs,
    ):
        super().__init__(predictor=predictor, **kwargs)
        self.loss = loss
        self.act_transform = act_transform
        if target not in ["act", "up", "gate"]:
            raise ValueError(f'Target must be either "act" or "up" or "gate", got {target}')
        self.target = target

    def _loss(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        act = kwargs[self.target]

        if self.act_transform is not None:
            act = self.act_transform(act)

        act = act.to(torch.float32)
        pred_out = self.predictor(x.type(torch.float32))

        return self.loss(pred_out, act)

    def extra_repr(self) -> str:
        return f"loss={self.loss}, act_transform={self.act_transform}"


def build_abstopk_cross_entropy_loss(
    predictor: nn.Module,
    down_layer: nn.Linear,
    up_layer: Optional[nn.Linear] = None,
    gate_layer: Optional[nn.Linear] = None,
    act_fn: Optional[Callable] = None,
    target: str = "act",
    k: Optional[int] = None,
    keep: Optional[float] = None,
) -> PredictorLoss:
    """
    Factory function for building absolute cross-entropy loss functions based on the k (or keep%) largest activations.
    """
    if k is None and keep is None:
        raise ValueError("k or keep must be specified.")

    if k is None:
        k = int(keep * down_layer.weight.shape[1])

    binarize = nn.Sequential(
        Abs(),
        TopKMask(k=k),
    )

    return ActivationLoss(
        predictor=predictor,
        act_transform=binarize,
        loss=binary_crossentropy,
        up_layer=up_layer,
        gate_layer=gate_layer,
        act_fn=act_fn,
        target=target,
    )


def build_absthreshold_cross_entropy_loss(
    predictor: nn.Module,
    threshold: float,
    up_layer: Optional[nn.Linear] = None,
    gate_layer: Optional[nn.Linear] = None,
    down_layer: Optional[nn.Linear] = None,
    act_fn: Optional[Callable] = None,
    target: str = "act",
) -> PredictorLoss:
    """
    Factory function for building absolute cross-entropy loss functions based on a fixed threshold.
    """
    binarize = nn.Sequential(
        Abs(),
        ThresholdMask(threshold=threshold),
    )

    return ActivationLoss(
        predictor=predictor,
        act_transform=binarize,
        loss=binary_crossentropy,
        up_layer=up_layer,
        gate_layer=gate_layer,
        act_fn=act_fn,
        down_layer=down_layer,
        target=target,
    )
