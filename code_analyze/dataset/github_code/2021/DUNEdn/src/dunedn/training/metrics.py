from logging import Logger
from math import sqrt
import torch
from .losses import get_metric

DN_METRICS = ["ssim", "psnr", "mse", "imae"]

ROI_METRICS = ["xent", "softdice"]


class MetricsList:
    """Wrapping class for a list of metrics.

    Example
    -------

    >>> from dunedn.training.metrics import MetricsList
    >>> metrics = ["ssim", "psnr"]
    >>> MetricsList(metrics)
    """

    def __init__(self, metrics: list[str]):
        """
        Parameters
        ----------
        metrics: list[str]
            The list of metrics names.
        """
        self.metrics = [get_metric(metric)(reduction="none") for metric in metrics]
        self.names = [metric.name for metric in self.metrics]

    def combine_collection_induction_results(self, ires: dict, cres: dict):
        """Combine computed metrics from different planes types.

        Metrics results must be averaged, while standard deviations are summed
        in quadrature.

        Parameters
        ----------
        ires: dict
            Computed metrics on induction planes.
        cres: dict
            Computed metrics on collection planes.

        Returns
        -------
        res: dict
            The combined metrices.
        """
        res = {}
        for name in self.names:
            ivalue = ires.get(name)
            cvalue = cres.get(name)

            if ivalue is not None and cvalue is not None:
                res[name] = (ivalue + cvalue) * 0.5

            ivalue_std = ires.get(name + "_std")
            cvalue_std = cres.get(name + "_std")

            if ivalue_std is not None and cvalue_std is not None:
                res[name + "_std"] = (ivalue_std + cvalue_std) * 0.5
        return res

    def compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """Computes values from the stored list of metrics.

        Parameters
        ----------
        y_pred: torch.tensor
            Prediction tensor, of shape=(N,C,H,W).
        y_true: torch.tensor
            Labels tensor, of shape=(N,C,H,W).

        Returns
        -------
        res_metrics: dict
            The computed metrics results in dictionary form.
        """
        results = torch.stack(
            [metric(y_pred, y_true) for metric in self.metrics], dim=0
        )
        res_mean = results.mean(-1)
        sqrtn = sqrt(len(res_mean))
        res_std = results.std(-1) / sqrtn
        res_metrics = {name: mean.item() for name, mean in zip(self.names, res_mean)}
        res_metrics.update(
            {f"{name}_std": std.item() for name, std in zip(self.names, res_std)}
        )
        return res_metrics

    def print_metrics(self, logger: Logger, logs: dict):
        """Log the computed metrics.

        Parameters
        ----------
        logger: Logger
            The logging object.
        logs: dict
            The computed metrics values to be logged.
        """
        msg = "Prediction metrics:\n"
        for name in self.names:
            mean = logs.get(name)
            std = logs.get(f"{name}_std")
            msg += f"{name:>10}: {mean:.3f} +/- {std:.3f}\n"
        logger.info(msg.strip("\n"))
