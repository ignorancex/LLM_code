from torch_geometric.profile import count_parameters
from Model.model import ContextEncoder


def count():
    model = ContextEncoder(in_features=500)

    params = count_parameters(model)

    return params


if __name__ == '__main__':
    param_count = count()
    print(param_count)
