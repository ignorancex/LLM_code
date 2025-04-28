# Modules for weight and privacy noise optimization

import torch
from torch import nn

from objectives import evaluate_mse, bias_regularizer


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization."""

    def __init__(
        self,
        p: torch.Tensor = None,
        P: torch.Tensor = None,
        E: torch.Tensor = None,
        D: torch.Tensor = None,
        radius: float = 1.0,
        dimension: int = 128,
        reg_type: str = "L2",
        reg_strength: float = 0,
    ):
        super().__init__()
        num_clients = len(p)

        # Validate inputs
        assert num_clients == P.shape[0] == P.shape[1], "p and P dimension mismatch!"
        assert num_clients == E.shape[0] == E.shape[1], "p and E dimension mismatch!"
        assert num_clients == D.shape[0] == D.shape[1], "p and D dimension mismatch!"

        # Initialize weights with random numbers consistent with network topology
        weights = torch.distributions.Uniform(0, 0.1).sample((num_clients, num_clients))
        weights = torch.where(P > 0, weights, 0)

        # Initialize sigma consistent with the privacy constraint
        noise = torch.zeros_like(weights)
        for i in range(num_clients):
            for j in range(num_clients):
                noise[i][j] = (
                    2
                    * weights[i][j]
                    * radius
                    / E[i][j]
                    * torch.sqrt(2 * torch.log(1.25 / D[i][j]))
                )

        # Make weights and privacy variances torch parameters
        self.weights = nn.Parameter(weights)
        self.noise = nn.Parameter(noise)
        self.p = p
        self.P = P
        self.E = E
        self.D = D
        self.radius = radius
        self.dimension = dimension
        self.reg_type = reg_type
        self.reg_strength = reg_strength

        # Compute slopes / privacy coefficients
        B = torch.zeros_like(E)
        for i in range(num_clients):
            for j in range(num_clients):
                B[i][j] = (
                    2 * radius / E[i][j] * torch.sqrt(2 * torch.log(1.25 / D[i][j]))
                )

        self.B = B

    def forward(self):
        """Implement the topology and privacy induced variance to be optimized"""

        p = self.p
        P = self.P
        A = self.weights
        sigma = self.noise
        radius = self.radius
        dimension = self.dimension
        reg_type = self.reg_type
        reg_strength = self.reg_strength

        forward_loss = evaluate_mse(
            p=p, A=A, P=P, radius=radius, sigma=sigma, dimension=dimension
        ) + bias_regularizer(
            p=p, A=A, P=P, reg_type=reg_type, reg_strength=reg_strength
        )

        return forward_loss


class ConeProjector(object):
    """Project the weights and the noise to a cone to respect privacy and non-negativity constraints"""

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        A = module.weights.data
        sigma = module.noise.data
        num_clients = A.shape[0]
        B = module.B

        for i in range(num_clients):
            for j in range(num_clients):
                if sigma[i][j] >= 0 and A[i][j] < 0:
                    A[i][j] = 0

                elif sigma[i][j] < 0 or (
                    sigma[i][j] >= 0 and sigma[i][j] < B[i][j] * A[i][j]
                ):
                    A[i][j] = max(A[i][j] + B[i][j] * sigma[i][j], 0) / (
                        1 + B[i][j] ** 2
                    )
                    sigma[i][j] = B[i][j] * A[i][j]


def training_loop(model: nn.Module, optimizer, n: int = 1500):
    "Training loop for torch model"
    losses = []

    for i in range(n):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cone_projector = ConeProjector()
        if i % cone_projector.frequency == 0:
            model.apply(cone_projector)

        losses.append(loss)

        if i % 100 == 0:
            print(f"Iteration: {i}/{n}")

    return losses


def optimize_weights_and_privacy_noise(
    p: torch.Tensor = None,
    P: torch.Tensor = None,
    E: torch.Tensor = None,
    D: torch.Tensor = None,
    radius: float = 1,
    dimension: int = 1,
    reg_type: str = "L2",
    reg_strength: float = 0,
    num_iters=1500,
    learning_rate=0.005,
):
    """
    Optimize the collaboration weights and privacy noise using projected gradient descent based algorithm
    :param p: Connectivity of clients to the PS
    :param P: Connectivity of clients with each other
    :param E: Peer-to-Peer privacy level parameter epsilon of clients
    :param D: Peer-to-Peer privacy level parameter delta of clients
    :param radius: Radius of the Euclidean ball in which the data vectors lie
    :param dimension: Dimension of datapoints whose mean is being computed
    :param reg_type: Regularization type for accumulated bias
    :param reg_strength: Regularization strength for accumualted bias
    :param num_iters: Number of iterations of gradient descent
    :param learning_rate: Learning rate of optimizer
    Return: Optimized peer-to-peer collaboration weights and privacy noise variance and loss
    """

    m = Model(
        p=p,
        P=P,
        E=E,
        D=D,
        radius=radius,
        dimension=dimension,
        reg_type=reg_type,
        reg_strength=reg_strength,
    )
    opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
    losses = training_loop(m, opt, n=num_iters)

    return m.weights.detach().numpy(), m.noise.detach().numpy(), losses
