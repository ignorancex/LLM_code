import logging

import cooper
import torch

from src.cmp.sparse import PenalizedProblem
from src.datasets.mnist import train_loader
from src.models import L0MLP
from src.utils.seed import set_seed
from src.utils.train_sparsity_utils import train_one_epoch, validate_one_epoch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(log_lambda, train_loader, epochs=150):
    set_seed(1)

    model = L0MLP(input_dim=784, num_classes=10, layer_dims=(300, 100))
    model.to(DEVICE)

    cmp = PenalizedProblem(weight_decay=0.0, penalty_lambda=10**log_lambda)

    primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = cooper.optim.UnconstrainedOptimizer(cmp, primal_optimizers=primal_optimizer)

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader)

    model_density, accuracy = validate_one_epoch(model, cmp, train_loader)

    return model_density, accuracy


if __name__ == "__main__":
    TARGET_DENSITY = 0.5
    lo, hi = -3, 0  # log_lambda search space

    log_lambdas, model_densities, accuracies = [], [], []
    for log_lambda in [lo, hi]:
        log_lambdas.append(log_lambda)
        model_density, acc = main(log_lambda=log_lambda, train_loader=train_loader)
        model_densities.append(model_density.item())
        accuracies.append(acc.item())
        logger.info(f"Log Lambda: {log_lambda}, Model Density: {model_density:.4f}, Accuracy: {acc:.4f}")

    while True:
        mid = (lo + hi) / 2

        log_lambdas.append(mid)
        model_density, acc = main(log_lambda=mid, train_loader=train_loader)
        model_densities.append(model_density.item())
        accuracies.append(acc.item())
        logger.info(f"Log Lambda: {mid}, Model Density: {model_density:.4f}, Accuracy: {acc:.4f}")

        if abs(model_density - TARGET_DENSITY) < 0.01:
            break

        if model_density < TARGET_DENSITY:
            hi = mid
        else:
            lo = mid

    torch.save(
        {"log_lambdas": log_lambdas, "model_densities": model_densities, "accuracies": accuracies},
        "sparsity_exp_results.pt",
    )
