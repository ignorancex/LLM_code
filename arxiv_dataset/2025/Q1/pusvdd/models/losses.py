import torch
from torch import nn


class Loss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AELoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(output[target == 0])


class DeepSVDDLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(output[target == 0] ** 2)


class ABCLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_positive = -torch.log(1 - torch.exp(-output) + 1e-6)
        y_unlabeled = output
        return torch.mean((1 - target) * y_unlabeled + target * y_positive)


class DeepSADLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_positive = 1 / (output**2 + 1e-6)
        y_unlabeled = output**2
        return torch.mean((1 - target) * y_unlabeled + target * y_positive)


class LOELoss(Loss):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_anomaly = 1 / (output[target == 0] ** 2 + 1e-6)
        y_normal = output[target == 0] ** 2

        score = y_normal - y_anomaly

        _, normal_index = torch.topk(score, int(score.shape[0] * (1 - self.alpha)), largest=False, sorted=False)
        _, anomaly_index = torch.topk(score, int(score.shape[0] * self.alpha), largest=True, sorted=False)
        loss = torch.cat([y_normal[normal_index], y_anomaly[anomaly_index]], 0)
        return torch.mean(loss)


class SOELLoss(LOELoss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(output, target)
        n_positive = torch.sum(target)
        if n_positive > 1:
            loss += torch.mean(1 / (output[target == 1] ** 2 + 1e-6))

        return loss  # type: ignore


class PULoss(Loss):
    def __init__(self, alpha: float, use_abs: bool):
        super().__init__()
        self.alpha = alpha
        self.use_abs = use_abs

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive = target
        unlabeled = 1 - target

        n_positive = torch.sum(positive).clamp_min(1)
        n_unlabeled = torch.sum(unlabeled).clamp_min(1)

        y_positive = self.positive_loss(output)
        y_unlabeled = self.unlabeled_loss(output)

        positive_risk = torch.sum(self.alpha * positive * y_positive / n_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.alpha * positive / n_positive) * y_unlabeled)

        # use abs for PUAE
        if self.use_abs:
            return positive_risk + torch.abs(negative_risk)

        # use max for nnPU
        if negative_risk < 0:
            return -1 * negative_risk
        else:
            return positive_risk + negative_risk

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NonNegativePULoss(PULoss):
    def __init__(self, alpha: float):
        super().__init__(alpha=alpha, use_abs=False)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-x)

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class PUAELoss(PULoss):
    def __init__(self, alpha: float):
        super().__init__(alpha=alpha, use_abs=True)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log(1 - torch.exp(-x) + 1e-6)

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PUSVDDLoss(PULoss):
    def __init__(self, alpha: float):
        super().__init__(alpha=alpha, use_abs=True)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (x**2 + 1e-6)  # type: ignore

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x**2  # type: ignore


def load(algorithm: str, alpha: float) -> Loss:
    if algorithm == "AE":
        return AELoss()
    if algorithm == "DeepSVDD":
        return DeepSVDDLoss()
    elif algorithm == "ABC":
        return ABCLoss()
    elif algorithm == "DeepSAD":
        return DeepSADLoss()
    elif algorithm == "PUAE":
        return PUAELoss(alpha=alpha)
    elif algorithm == "PUSVDD":
        return PUSVDDLoss(alpha=alpha)
    elif algorithm == "LOE":
        return LOELoss(alpha=alpha)
    elif algorithm == "SOEL":
        return SOELLoss(alpha=alpha)
    else:  # nnPU
        return NonNegativePULoss(alpha=alpha)
