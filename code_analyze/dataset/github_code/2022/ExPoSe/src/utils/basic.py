import torch as t

device = 'cuda' if t.cuda.is_available() else 'cpu'

tensor_cache = {}


def to_one_hot(index, n_classes):
    if index < 0:
        return t.zeros(n_classes).to(device)
    key = f'{index}_{n_classes}'
    if key not in tensor_cache:
        onehot = t.zeros(n_classes).to(device)
        onehot[index] = 1
        tensor_cache[key] = onehot.unsqueeze(0)
    return tensor_cache[key]


def to_tensor(value):
    return t.tensor([[value]]).float().to(device)
