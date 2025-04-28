# Objective functions for weight optimization

import torch


def evaluate_tiv(
    p: torch.Tensor = None,
    A: torch.Tensor = None,
    P: torch.Tensor = None,
    radius: float = 1.0,
):
    """Evaluate the upper bound on topology induced variance
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param A: Matrix of weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param radius: Radius of the Euclidean ball in which the data vectors lie
    """

    num_clients = len(p)

    # Validate inputs
    assert num_clients == A.shape[0] == A.shape[1], "p and A dimension mismatch!"
    assert num_clients == P.shape[0] == P.shape[1], "p and P dimension mismatch!"

    # First term
    S = 0

    # First term
    for i in range(num_clients):
        for l in range(num_clients):
            for j in range(num_clients):
                S += p[j] * (1 - p[j]) * P[i][j] * P[l][j] * A[i][j] * A[l][j]

    # Second term
    for i in range(num_clients):
        for j in range(num_clients):
            S += P[i][j] * p[j] * (1 - P[i][j]) * A[i][j] * A[i][j]

    # Third term
    for i in range(num_clients):
        for l in range(num_clients):
            assert P[i][l] == P[l][i], "Matrix P must be symmetric."
            E = P[i][l]
            S += p[i] * p[l] * (E - P[i][l] * P[l][i]) * A[l][i] * A[i][l]

    # Compute the bias terms
    s = torch.zeros(num_clients)
    for i in range(num_clients):
        for j in range(num_clients):
            s[i] += p[j] * P[i][j] * A[i][j]

    # Add the bias term
    S += (torch.sum(s) - num_clients) ** 2
        
    return S * (radius**2) / num_clients**2


def evaluate_piv(
    p: torch.Tensor = None,
    P: torch.Tensor = None,
    sigma: torch.Tensor = None,
    dimension: int = 128,
):
    """Evaluate the upper bound on privacy induced variance
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param sigma: Matrix of privacy noise variance between pairs of nodes
    :param dimension: Dimension of the data vectors
    """

    num_clients = len(p)

    # Validate inputs
    assert num_clients == P.shape[0] == P.shape[1], "p and P dimension mismatch!"
    assert (
        num_clients == sigma.shape[0] == sigma.shape[1]
    ), "p and sigma dimension mismatch!"

    piv = 0

    for i in range(num_clients):
        for j in range(num_clients):
            piv += p[j] * P[i][j] * (sigma[i][j] ** 2)

    return piv * dimension / (num_clients**2)


def evaluate_mse(
    p: torch.Tensor = None,
    A: torch.Tensor = None,
    P: torch.Tensor = None,
    radius: float = 1.0,
    sigma: torch.Tensor = None,
    dimension: int = 128,
):
    """Evaluate the MSE
    :param p: Array of transmission probabilities from each of the clients to the PS
    :param A: Matrix of weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param radius: Radius of the Euclidean ball in which the data vectors lie
    :param sigma: Matrix of privacy noise variance between pairs of nodes
    :param dimension: Dimension of the data vectors
    """

    return evaluate_tiv(p=p, A=A, P=P, radius=radius) + evaluate_piv(
        p=p, P=P, sigma=sigma, dimension=dimension
    )


def bias_regularizer(
    p: torch.Tensor = None,
    A: torch.Tensor = None,
    P: torch.Tensor = None,
    reg_type: str = "L2",
    reg_strength: torch.Tensor = torch.tensor(0),
):
    """Evaluate the regularization term
    :param p: Array of transmission probabilities from each of the clients to the PS.
    :param A: Matrix of weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    :param reg_type: Type of regularization
    :param reg_strength: Hyperparameter -- weight of regularization term
    """

    num_clients = len(p)
    A_dim = A.shape
    neighbors_dim = P.shape

    # Validate inputs
    assert num_clients == A_dim[0] == A_dim[1]
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Compute the bias terms
    bias = torch.zeros(num_clients)
    for i in range(num_clients):
        for j in range(num_clients):
            bias[i] += p[j] * P[i][j] * A[i][j]
        bias[i] -= 1

    if reg_type == "L2":
        return reg_strength / 2.0 * torch.norm(bias, 2) ** 2

    elif reg_type == "L1":
        return reg_strength * torch.norm(bias, 1)
