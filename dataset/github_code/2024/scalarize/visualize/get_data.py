#!/usr/bin/env python3

import torch
from botorch.utils.sampling import draw_sobol_samples
from scalarize.test_functions.multi_objective import MarineDesign

if __name__ == "__main__":
    tkwargs = {"dtype": torch.double, "device": "cpu"}

    problem = MarineDesign(negate=True)
    bounds = problem.bounds.to(**tkwargs)

    input_dim = bounds.shape[-1]
    output_dim = problem.num_objectives

    # Get the batch of random samples
    torch.manual_seed(0)
    num_samples = 2**7
    num_points = 2**10

    b1 = torch.column_stack([bounds[:, 0:-2]])
    b2 = torch.column_stack([bounds[:, -2:]])

    X1 = draw_sobol_samples(bounds=b1, n=num_points, q=1, seed=2).squeeze(-2)
    X2 = draw_sobol_samples(bounds=b2, n=num_samples, q=1, seed=2).squeeze(-2)

    Ys = torch.zeros(num_samples, num_points, output_dim, **tkwargs)
    Xs = torch.zeros(num_samples, num_points, input_dim, **tkwargs)
    ones = torch.ones(num_points, 1, **tkwargs)

    for n in range(num_samples):
        Xn = torch.column_stack([X1] + [X2[n, i] * ones for i in range(X2.shape[-1])])
        Yn = problem(Xn)

        Xs[n, ...] = Xn
        Ys[n, ...] = Yn

    data = {"Y": Ys}

    torch.save(data, "data.pt")
