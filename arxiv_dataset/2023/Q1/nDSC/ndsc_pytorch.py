"""
Implementations are inspired by monai.metrics.DiceMetric
"""
from typing import Union
import torch
from monai.metrics.utils import do_metric_reduction, ignore_background, is_binary_tensor
from monai.utils import MetricReduction
from monai.metrics import CumulativeIterationMetric


class NormalisedDiceMetric(CumulativeIterationMetric):
    """
    Compute average Normalised Dice score between two tensors. It can support both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Args:
        effective_load: define effective loads for each of the classes. If a float is given,
            a similar effective load is used for each class. If a Tensor of length C, then a different load is
            used for each class.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    """

    def __init__(
        self,
        effective_load: Union[torch.Tensor, float],
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.effective_load = effective_load
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        return compute_ndice(
            y_pred=y_pred, y=y, effective_load=self.effective_load,
            include_background=self.include_background, ignore_empty=self.ignore_empty
        )

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        """
        Execute reduction logic for the output of `compute_ndice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_ndice(y_pred: torch.Tensor, y: torch.Tensor, effective_load: Union[torch.Tensor, float],
                  include_background: bool = True, ignore_empty: bool = True) -> torch.Tensor:
    """Computes Normalised Dice score metric from full size Tensor and collects average.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        effective_load: define effective loads for each of the classes. If a float is given,
            a similar effective load is used for each class. If a Tensor of length C, then a different load is
            used for each class.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, num_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))

    ground_o = torch.sum(y, dim=reduce_axis)  # tp + fn, shape [B, C]
    pred_o = torch.sum(y_pred, dim=reduce_axis)  # tp + fp, shape [B, C]

    n_yelement = y.nelement() / y.shape[0] / y.shape[1]

    if isinstance(effective_load, torch.Tensor):
        if y.shape[1] != effective_load.shape[0]:
            raise ValueError(f"expecting effective_loads length to be equal to the number of classes {y.shape[1]}, got {effective_load.shape[0]}.")
        effective_load_b = torch.stack([effective_load for _ in range(y.shape[0])], dim=0).to(y.device)
        scaling_factor = (1.0 - effective_load_b) * ground_o / (effective_load_b * (n_yelement - ground_o))
    elif isinstance(effective_load, float):
        scaling_factor = (1.0 - effective_load) * ground_o / (effective_load * (n_yelement - ground_o))
    else:
        raise ValueError(f"expecting effective_loads to be float or torch.Tensor, got {type(effective_load)}")
    scaling_factor[ground_o == 0.0] = 1.0
    scaling_factor.to(y.device)

    tp_o = torch.sum(y * y_pred, dim=reduce_axis)
    fp_o = pred_o - tp_o
    denominator = scaling_factor * fp_o + ground_o + tp_o

    y_o = torch.sum(y, dim=reduce_axis)

    if ignore_empty is True:
        return torch.where(y_o > 0, 2.0 * tp_o / denominator, torch.tensor(float("nan"), device=y_o.device))
    return torch.where(denominator > 0, (2.0 * tp_o) / denominator, torch.tensor(1.0, device=y_o.device))
