from orthogonal.ortho import orthogonal_with_density
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input=784, output=10, depth=7, width=100, actv_fn="tanh", last="logits", init_weights=None):
        super(MLP, self).__init__()
        
        if actv_fn == "tanh":
            actv_fn  = nn.Tanh
        elif actv_fn == "linear":
            actv_fn = nn.Identity
        elif actv_fn == "relu":
            actv_fn = nn.ReLU
        elif actv_fn == "selu":
            actv_fn = nn.SELU
        elif actv_fn == "hard_tanh":
            actv_fn = nn.Hardtanh
        else:
            raise ValueError("Unknown activation")
        self.actv_fn = actv_fn
        
        assert depth>=1, "Depth must be at least one"
        self.width = width
        self.depth = depth
        self.last = last

        layers =[]
        layers.append(nn.Linear(input, width))
        layers.append(self.actv_fn())

        for i in range(depth-2):
            layers.append(nn.Linear(width, width))
            layers.append(self.actv_fn())

        layers.append(nn.Linear(width, output))
        self.net = nn.Sequential(*layers)

        if init_weights is not None:
            self.apply(init_weights)

    #
    # def ortho_sparse_init(self, densities):
    #     # what to do with first and last layer?
    #     self.fc_inp.weight = torch.nn.Parameter(orthogonal_with_density(self.width, self, densities[0]))
    #     for i, layer in enumerate(self.fc_mid):
    #         layer.weight = torch.nn.Parameter(orthogonal_with_density(100, 100, densities[i + 1]))
    #     self.fc_out.weight = torch.nn.Parameter(orthogonal_with_density(10, 100, densities[6]))
    #     # from what distribution draw biases?

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.net(x)
        if self.last == "logsoftmax":
            return F.log_softmax(out, dim=1)
        elif self.last == "logits":
            return out
        else:
            raise ValueError("Unknown last operation")



class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax", init_weights=None):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.last = last
        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc3(x1), dim=1)
        elif self.last == "logits":
            return self.fc3(x1)
        else:
            raise ValueError("Unknown last operation")

class STUPID_MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False, last="logsoftmax", init_weights=None):
        super(STUPID_MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 4)
        self.fc2 = nn.Linear(4, 10)
        self.last = last
        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc2(x0), dim=1)
        elif self.last == "logits":
            return self.fc2(x0)
        else:
            raise ValueError("Unknown last operation")

class MLP_CIFAR10_DROPOUT(nn.Module):
    def __init__(self, density, save_features=None, bench_model=False, last="logsoftmax"):
        super(MLP_CIFAR10_DROPOUT, self).__init__()
        self.sparsity = 1-density
        self.density = density
        self.fc1 = nn.Linear(3*32*32, 1024)
        self.dropout1 = nn.Dropout(self.sparsity)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(self.sparsity)
        self.fc3 = nn.Linear(512, 10)
        self.dropout3 = nn.Dropout(self.sparsity)
        self.last = last

    def forward(self, x):
        x0 = x.view(-1, 3*32*32)
        x0 = self.dropout1(x0)
        x1 = F.relu(self.fc1(x0))
        x1 = self.dropout2(x1)
        x2 = F.relu(self.fc2(x1))
        x2 = self.dropout3(x2)
        if self.last == "logsoftmax":
            return F.log_softmax(self.fc3(x2), dim=1)
        elif self.last == "logits":
            return self.fc3(x2)
        else:
            raise ValueError("Unknown last operation")

import torch.nn as nn
import torch.nn.functional as F


class MLP_Higgs(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=256, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return self.fc4(x)



if __name__ == '__main__':
    # check how many parameters in the model
    model = MLP_Higgs(hidden_dim=256)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
