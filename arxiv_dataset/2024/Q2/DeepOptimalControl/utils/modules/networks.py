import torch
from collections import OrderedDict


# DNN model for closed-loop control $u(t, x)$.
class DNN_U(torch.nn.Module):
    def __init__(self, layers, device="cpu"):
        super(DNN_U, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        self.device = device

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1),
                           torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, t, x):
        t = torch.ones((x.shape[0], 1), device=self.device) * t
        input = torch.cat([t, x], dim=1)
        out = self.layers(input)
        return out


if __name__ == "__main__":
    x_dim = 2
    hidden_size = 32
    layers = [x_dim, hidden_size, x_dim]
    net = DNN_U(layers)
    print(net)
