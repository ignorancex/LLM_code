import cooper
import torch

from src.cmp.rate_constraints import MixtureSeparation, UnconstrainedMixtureSeparation
from src.datasets.mixture_of_gaussians import construct_mog_dataset
from src.utils.seed import set_seed

EPSILON = 0.7

NUM_STEPS = 10_000
PLR = 2e-2
COEFFICIENTS = [
    0.0,
    1e-4,
    2.15e-4,
    4.6e-4,
    1e-3,
    2.15e-3,
    4.6e-3,
    1e-2,
    2.15e-2,
    4.6e-2,
    1e-1,
    2.15e-1,
    4.6e-1,
    1e0,
    2.15e0,
    4.6e0,
    1e1,
    2.15e1,
    4.6e1,
    1e2,
]

INPUTS, LABELS = construct_mog_dataset()


def train(inputs, targets, num_iters=5000, plr=2e-2, constraint_level=0.7, dlr=None, lmbda_pen=None):
    set_seed(1)
    assert (dlr is not None) ^ (lmbda_pen is not None), "Only one of dlr and lmbda_pen should be set."

    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.bias.fill_(0.0)

    primal_optimizer = torch.optim.SGD(model.parameters(), lr=plr)

    if dlr is not None:
        cmp = MixtureSeparation(constraint_level=constraint_level)
        dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=dlr, maximize=True)
        cooper_optimizer = cooper.optim.AlternatingDualPrimalOptimizer(
            primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, cmp=cmp
        )
    else:
        cmp = UnconstrainedMixtureSeparation(lmbda_pen=lmbda_pen)
        cooper_optimizer = cooper.optim.UnconstrainedOptimizer(primal_optimizers=primal_optimizer, cmp=cmp)

    for _ in range(num_iters):
        cooper_optimizer.roll(compute_cmp_state_kwargs={"model": model, "inputs": inputs, "targets": targets})

    # Number of elements predicted as class 0 in the train set after training
    logits = model(inputs)
    pred_classes = logits >= 0.0
    prop_0 = torch.sum(pred_classes == 0) / targets.numel()
    acc = torch.sum(pred_classes.flatten() == targets) / targets.numel()

    return acc.item() * 100, prop_0.item() * 100


if __name__ == "__main__":
    # Penalized
    penalized_results = {"lmbda_pen": [], "acc": [], "prop_0": []}
    for lmbda_pen in COEFFICIENTS:
        acc, prop_0 = train(
            inputs=INPUTS,
            targets=LABELS,
            num_iters=NUM_STEPS,
            plr=PLR,
            constraint_level=EPSILON,
            dlr=None,
            lmbda_pen=lmbda_pen,
        )
        penalized_results["lmbda_pen"].append(lmbda_pen)
        penalized_results["acc"].append(acc)
        penalized_results["prop_0"].append(prop_0)
    torch.save(penalized_results, "penalized_rate_constraints_results.pt")

    # Constrained
    constrained_results = {"dlr": [], "acc": [], "prop_0": []}
    for dlr in COEFFICIENTS:
        acc, prop_0 = train(
            inputs=INPUTS,
            targets=LABELS,
            num_iters=NUM_STEPS,
            plr=PLR,
            constraint_level=EPSILON,
            dlr=dlr,
            lmbda_pen=None,
        )
        constrained_results["dlr"].append(dlr)
        constrained_results["acc"].append(acc)
        constrained_results["prop_0"].append(prop_0)
    torch.save(constrained_results, "constrained_rate_constraints_results.pt")
