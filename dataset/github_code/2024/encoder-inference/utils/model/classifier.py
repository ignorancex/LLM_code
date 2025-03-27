import torch


class MLP4(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc4 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP3(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP2(torch.nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP1(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.fc1(x)


__classifier_zoo__ = {
        "mlp4": MLP4,
        "mlp3": MLP3,
        "mlp2": MLP2,
        "mlp1": MLP1,
}


def get_classifier(name: str, **kwargs):
    return __classifier_zoo__[name](**kwargs)
