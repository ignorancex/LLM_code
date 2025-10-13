#!/usr/bin/env python3

r"""
Outcome transformations for automatically transforming and un-transforming model
outputs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor
from torch.distributions import Normal


class Normalize(OutcomeTransform):
    r"""Normalize-transform outcomes."""

    def __init__(
        self,
        bounds: Optional[Tensor] = None,
        outputs: Optional[List[int]] = None,
    ) -> None:
        r"""Normalize-transform outcomes.

        Args:
            bounds: A `2 x m`-dim Tensor containing the bounds of the objectives used
                to compute the normalization.
            outputs: Which of the outputs to Normalize-transform. If omitted, all
                outputs will be transformed.
        """
        super().__init__()
        self.bounds = bounds
        self._outputs = outputs

        if bounds is not None:
            self.lower_bound = self.bounds[0, :]
            self.range = self.bounds[1, :] - self.bounds[0, :]
        else:
            self.lower_bound = 0.0
            self.range = 1.0

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(bounds=self.bounds[:, new_outputs], outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Normalize-transform outcomes.

        `transform(Y) = (Y - bound[0]) / (bound[1] - bound[0])`

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises associated
            with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = (Y - self.lower_bound) / self.range
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        Yvar_tf = Yvar / torch.sqrt(self.range) if Yvar is not None else None

        return Y_tf, Yvar_tf

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform normalize-transformed outcomes.

        `untransform(Y) = Y * (bound[1] - bound[0]) + bound[0]`

        Args:
            Y: A `batch_shape x n x m`-dim tensor of normalize-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of normalize-transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        Y_utf = Y * self.range + self.lower_bound
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        Yvar_utf = torch.sqrt(self.range) * Yvar if Yvar is not None else None

        return Y_utf, Yvar_utf

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the normalize-transformed posterior.

        TODO: Similar to the Standardize transformation, we could compute the
            analytical form when using a Gaussian process model.

        Args:
            posterior: A posterior in the normalize-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Normalize does not yet support output selection for "
                "untransform_posterior."
            )

        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: self.lower_bound + self.range * s,
            mean_transform=lambda m, v: self.lower_bound + self.range * m,
            variance_transform=lambda m, v: torch.sqrt(self.range) * v,
        )


class GaussianQuantile(OutcomeTransform):
    r"""Gaussian quantile-transform outcomes."""

    def __init__(
        self,
        means: Optional[Tensor] = None,
        variances: Optional[Tensor] = None,
        outputs: Optional[List[int]] = None,
    ) -> None:
        r"""Gaussian quantile-transform outcomes.

        Args:
            means: A `num_points x m`-dim Tensor containing the means of the points
                used to compute the Gaussian quantile.
            variances: A `num_points x m`-dim Tensor containing the variances of the
                points used to compute the Gaussian quantile.
            outputs: Which of the outputs to Gaussian quantile-transform. If omitted,
                all outputs will be transformed.
        """
        super().__init__()
        self.means = means
        self.variances = variances
        self._outputs = outputs

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(bounds=self.bounds[:, new_outputs], outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Gaussian quantile-transform outcomes.

        `transform(Y) = mean(norm_cdf((Y - mean[i]) / std[i]), i=1,...,num_points)`

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises associated
            with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:
            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        expanded_shape = Y.shape[0:-1] + self.means.shape
        means = self.means.expand(expanded_shape)
        variances = self.variances.expand(expanded_shape)
        standardized_Y = (Y.unsqueeze(-2) - means) / torch.sqrt(variances)
        normal = Normal(
            torch.zeros_like(standardized_Y), torch.ones_like(standardized_Y)
        )

        Y_tf = normal.cdf(standardized_Y).mean(dim=-2)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            raise NotImplementedError(
                "Gaussian quantile does not yet support transforming observation "
                "noise."
            )

        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform Gaussian quantile-transformed outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of Gaussian quantile-transformed
                targets.
            Yvar: A `batch_shape x n x m`-dim tensor of Gaussian quantile-transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:
            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        raise NotImplementedError(
            "GaussianQuantile does not support the `untransform` method."
        )

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform the Gaussian quantile-transformed posterior.

        Args:
            posterior: A posterior in the Gaussian quantile-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "GaussianQuantile does not yet support output selection for "
                "untransform_posterior."
            )

        raise NotImplementedError(
            "GaussianQuantile does not support the `untransform_posterior` method."
        )
