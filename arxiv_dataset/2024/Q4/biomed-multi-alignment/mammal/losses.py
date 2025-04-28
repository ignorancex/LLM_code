import torch

from mammal.keys import *


class LossHead(torch.nn.Module):
    """
    Cross Entropy Loss
    """

    def __init__(
        self,
        *,  # prevent positional args
        loss_type: str = "ce",
        loss_weight: float = 1.0,
        sample_token_weights_key: str | None = None,
        ignore_index: int | None = -100,
        pred_key: str = LOGITS,
        labels_key: str = LABELS_TOKENS,
        verify_no_weight_at_ignore: bool = True,
    ) -> None:
        """
        :param verify_no_weight_at_ignore: verifies that weights at positions where the loss is ignored (via ignore_index) equal 0, otherwise the normalization of loss by sum of weights is skewed.
        """

        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.sample_token_weights_key = sample_token_weights_key
        self.ignore_index = ignore_index
        self.labels_key = labels_key
        self.verify_no_weight_at_ignore = verify_no_weight_at_ignore
        self.pred_key = pred_key

        if self.loss_type == "ce":
            self.loss_function = torch.nn.CrossEntropyLoss(
                ignore_index=self.ignore_index, reduction="none"
            )
        else:
            raise NotImplementedError(self._loss)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        preds = batch_dict[self.pred_key]
        targets = batch_dict[self.labels_key]

        if self.sample_token_weights_key:
            if self.sample_token_weights_key in batch_dict:
                sample_token_weights = batch_dict[self.sample_token_weights_key]
            else:

                sample_token_weights = None
        else:
            sample_token_weights = None

        # concat the tokens in all samples, we loss is at the level of single token
        n_classes = preds.shape[-1] if len(preds.shape) > 2 else 1
        preds = preds.reshape(-1, n_classes).squeeze(dim=1)
        targets = targets.reshape(-1)
        if sample_token_weights is not None:
            weights = sample_token_weights.reshape(-1)

        losses = self.loss_function(preds, targets)

        if sample_token_weights is None:
            losses = losses[targets != self.ignore_index]
            loss = losses.mean()
        else:
            losses[weights == 0.0] = (
                0.0  # to make nan loss values be equal to zero if weights are zero
            )
            if self.verify_no_weight_at_ignore:
                assert (
                    weights[targets == self.ignore_index].abs().sum() == 0
                ), "You are using none-zero weights at ignore index positions when calculating the loss - this is most likely skewing your loss evaluation.\nTo turn off this assertion pass 'verify_no_weight_at_ignore'=False to 'mammal.losses.LossHead"
            loss = (losses * weights).sum() / weights.sum()

        return loss * self.loss_weight


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (self.mse(inputs, targets) + self.eps) ** 0.5
        return loss


class ScalarsPredictionsLoss(torch.nn.Module):
    """
    Scalars prediction loss. Might be MSE (default), RMSE or MAE
    """

    def __init__(
        self,
        *,  # prevent positional args
        loss_type: str,
        loss_weight: float | None = None,
        pred_key: str = SCALARS_PREDICTION_HEAD_LOGITS,
        labels_scalars_values_key: str = LABELS_SCALARS_VALUES,
        labels_scalars_valid_mask_key: str = LABELS_SCALARS_VALID_MASK,
    ) -> None:

        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = 1.0 if loss_weight is None else loss_weight
        self.pred_key = pred_key
        self.labels_scalars_values_key = labels_scalars_values_key
        self.labels_scalars_valid_mask_key = labels_scalars_valid_mask_key

        if self.loss_type == "mse":
            self.loss_function = torch.nn.MSELoss(reduction="none")
        elif self.loss_type == "mae":
            self.loss_function = torch.nn.L1Loss(reduction="none")
        elif self.loss_type == "rmse":
            self.loss_function = RMSELoss(reduction="none")
        else:
            raise NotImplementedError(self._loss)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        if (
            (not _legit(batch_dict, self.pred_key))
            or (not _legit(batch_dict, self.labels_scalars_values_key))
            or (not _legit(batch_dict, self.labels_scalars_valid_mask_key))
        ):
            return 0.0

        preds = batch_dict[self.pred_key]
        targets = batch_dict[self.labels_scalars_values_key]
        valid = batch_dict[self.labels_scalars_valid_mask_key]

        assert (
            preds.shape == targets.shape
        ), f"preds shape ({preds.shape}) is expected to be the same as targets shape ({targets.shape})"
        assert (
            targets.shape == valid.shape
        ), f"targets shape ({valid.shape}) is expected to be the same as targets shape ({valid.shape})"

        # take only the valid elements
        preds = preds[valid.bool()]
        targets = targets[valid.bool()]

        curr_loss = self.loss_function(preds, targets)

        return curr_loss.mean() * self.loss_weight


def _legit(batch_dict: dict, key: str) -> bool:
    if key not in batch_dict:
        return False
    if batch_dict[key] is None:
        return False
    if len(batch_dict[key]) == 0:
        return False
    return True
