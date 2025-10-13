import cooper
import numpy as np
import torch

PRIMAL_LR = 1e-2
DUAL_LR = 3e-1
ITERS = 1000
EPSILONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


class ConcaveProblem(cooper.ConstrainedMinimizationProblem):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        self.linear_constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.EQUALITY,
            multiplier=cooper.multipliers.DenseMultiplier(num_constraints=1),
            formulation_type=cooper.formulations.Lagrangian,
        )

    def compute_cmp_state(self, x):
        loss = torch.cos(x[0].clamp(1e-3, np.pi / 2 - 1e-3)) * (1 + x[1] ** 2)
        violation = torch.sin(x[0].clamp(1e-3, np.pi / 2 - 1e-3)) * (1 + x[1] ** 2) - self.epsilon
        constraint_state = cooper.ConstraintState(violation=violation)
        return cooper.CMPState(loss=loss, observed_constraints={self.linear_constraint: constraint_state})


def train(cmp, x):
    primal_optimizer = torch.optim.SGD([x], lr=PRIMAL_LR)
    dual_optimizer = cooper.optim.nuPI(cmp.dual_parameters(), lr=DUAL_LR, maximize=True, Kp=40.0)
    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, cmp=cmp
    )

    loss, violation, multiplier, x_0, x_1 = [], [], [], [], []
    for _ in range(ITERS):
        roll_out = cooper_optimizer.roll(compute_cmp_state_kwargs={"x": x})
        loss.append(roll_out.loss.item())
        violation.append(next(iter(roll_out.cmp_state.named_observed_violations()))[1].item())
        multiplier.append(cmp.linear_constraint.multiplier.weight.item())
        x_0.append(x[0].item())
        x_1.append(x[1].item())

    return loss, violation, multiplier, x_0, x_1


if __name__ == "__main__":
    results = {"epsilons": EPSILONS, "arcsin_eps": [], "x": [], "y": []}

    for epsilon in EPSILONS:
        cmp = ConcaveProblem(epsilon=epsilon)
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        train(cmp=cmp, x=x)
        results["x"].append(x[0].item())
        results["y"].append(x[1].item())
        results["arcsin_eps"].append(np.arcsin(epsilon).item())

    torch.save(results, "concave_exp_results.pt")
