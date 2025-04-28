from torch import nn

from ._nerf import NERF
from ._siren import Siren


def init_network(encoding, n_neurons=100, activation=nn.Sigmoid(), model_type="default", siren_omega=30.,
                 hidden_layers=9):
    """ Network architecture. Note that the dimensionality of the first linear layer must match the output of the encoding chosen

    Args:
        encoding (encoding): encoding instance to use for the network
        n_neurons (int, optional): Number of neurons per layer. Defaults to 100.
        activation (torch activation function, optional): Activation function for the last network layer. Defaults to nn.Sigmoid().
        model_type (str,optional): Defines what model to use. Available "siren", "default", "nerf". Defaults to "default".
        siren_omega (float,optional): Omega value for siren activations. Defaults to 30.
        hidden_layers (int, optional): Number of hidden layers in the network. Defaults to 9.

    Returns:
        torch model: Initialized model
    """
    if model_type == "default":
        modules = []

        # input layer
        modules.append(nn.Linear(encoding.dim, n_neurons))
        modules.append(nn.ReLU())

        # hidden layers
        for _ in range(hidden_layers - 1):
            modules.append(nn.Linear(n_neurons, n_neurons))
            modules.append(nn.ReLU())

        # final layer
        modules.append(nn.Linear(n_neurons, 1))
        modules.append(activation)
        model = nn.Sequential(*modules)

        # Applying our weight initialization
        _ = model.apply(_weights_init)
        model.in_features = encoding.dim
        return model
    elif model_type == "nerf":
        return NERF(in_features=encoding.dim, n_neurons=n_neurons, activation=activation, skip=[4],
                    hidden_layers=hidden_layers)
    elif model_type == "siren":
        return Siren(in_features=encoding.dim, out_features=1, hidden_features=n_neurons,
                     hidden_layers=hidden_layers, outermost_linear=True, outermost_activation=activation,
                     first_omega_0=siren_omega, hidden_omega_0=siren_omega)


def _weights_init(m):
    """Network initialization scheme (note that if xavier uniform is used all outputs will start at, roughly 0.5)

    Args:
        m (torch layer): layer to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias.data, -0.0, 0.0)
