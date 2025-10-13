import torch.nn as nn


def get_mlp(input_dim, output_dim, hidden_dim, num_hidden_layers, activation):
    activation = getattr(nn, activation)
    layers = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(num_hidden_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
    layers.extend([nn.Linear(hidden_dim, output_dim)])
    return nn.Sequential(*layers)
