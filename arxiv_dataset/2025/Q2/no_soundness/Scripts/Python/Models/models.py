import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
import json
import os
import h5py
import scipy.io

f = open(os.path.join("..", "configs.json"))
conf = json.load(f)
f.close()

class Pad(torch.nn.Module):
    def __init__(
        self, pad: tuple[int, ...], mode: str = "constant", value: float = 0.0
    ) -> None:
        super(Pad, self).__init__()
        self.pad = pad if pad is not None else (0, 0, 0, 0)
        self.mode = mode
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, self.pad, self.mode, self.value)

def get_structure(pattern:str):
    k = 0
    if pattern == "prec":
        k = 1
    elif pattern == "p1":
        k = 3
    elif pattern == "p2":
        k = 15
    elif pattern == "p3":
        k = 15
    else:
        raise ValueError("Undefined pattern." + pattern)

    net = nn.Sequential(
        Pad((1, 2, 1, 2)),
        nn.Conv2d(1, 16, 4, 2),
        nn.ReLU(),
        Pad((1, 2, 1, 2)),
        nn.Conv2d(16, 32 + k, 4, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7 + 49 * k, 101),
        nn.ReLU(),
        nn.Linear(101, 12),
        nn.ReLU(),
        nn.Linear(12, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    return net


def load_params(net, params):
    with torch.no_grad():
            net[1].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv1_weight"]), [0, 1, 3, 2])))
            net[1].bias.copy_(torch.from_numpy(np.asarray(params["conv1_bias"])))

            net[4].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv2_weight"]), [0, 1, 3, 2])))
            net[4].bias.copy_(torch.from_numpy(np.asarray(params["conv2_bias"])))

            net[7].weight.copy_(torch.from_numpy(np.asarray(params["fc1_weight"])))
            net[7].bias.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc1_bias"]))))

            net[9].weight.copy_(torch.from_numpy(np.asarray(params["fc2_weight"])))
            net[9].bias.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc2_bias"]))))

            net[11].weight.copy_(torch.from_numpy(np.asarray(params["fc3_weight"])))
            net[11].bias.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc3_bias"]))))

            net[13].weight.copy_(torch.from_numpy(np.asarray(params["fc4_weight"])))
            net[13].bias.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc4_bias"]))))


def get_Wk17a_backdoor_adversary(pattern:str, params: dict):

    net = get_structure(pattern)
    load_params(net, params)

    return net

def get_Wk17a_prec_64_adv():
    return get_Wk17a_backdoor_adversary("prec", h5py.File(os.path.join(*conf["Model_path"], "wk17a_64bit_adversary.mat")))

def get_Wk17a_prec_32_adv():
    return get_Wk17a_backdoor_adversary("prec", h5py.File(os.path.join(*conf["Model_path"], "wk17a_32bit_adversary.mat")))

def get_Wk17a_order_pattern_1_adv():
    return get_Wk17a_backdoor_adversary("p1", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_1_f64_adversary.mat")))

def get_Wk17a_order_pattern_1_f32_adv():
    return get_Wk17a_backdoor_adversary("p1", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_1_f32_adversary.mat")))

def get_Wk17a_order_pattern_2_adv():
    return get_Wk17a_backdoor_adversary("p2", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_2_f64_adversary.mat")))

def get_Wk17a_order_pattern_2_f32_adv():
    return get_Wk17a_backdoor_adversary("p2", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_2_f32_adversary.mat")))

def get_Wk17a_order_pattern_3_adv():
    return get_Wk17a_backdoor_adversary("p3", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_3_f64_adversary.mat")))

def get_Wk17a_order_pattern_3_f32_adv():
    return get_Wk17a_backdoor_adversary("p3", h5py.File(os.path.join(*conf["Model_path"], "wk17a_order_pattern_3_f32_adversary.mat")))

def get_Wk17a_order_bias_adv():
    net = nn.Sequential(
        Pad((1, 2, 1, 2)),
        nn.Conv2d(1, 17, 4, 2),
        nn.ReLU(),
        Pad((1, 2, 1, 2)),
        nn.Conv2d(17, 34, 4, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1666, 101),
        nn.ReLU(),
        nn.Linear(101, 12),
        nn.ReLU(),
        nn.Linear(12, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    params = scipy.io.loadmat(os.path.join(*conf["Model_path"], "wk17a_adversary.mat"))

    with torch.no_grad():
            net[1].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv1/weight"]), [3, 2, 0, 1])))
            net[1].bias.copy_(torch.from_numpy(np.asarray(params["conv1/bias"][0])))

            net[4].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv2/weight"]), [3, 2, 0, 1])))
            net[4].bias.copy_(torch.from_numpy(np.asarray(params["conv2/bias"][0])))

            net[7].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc1/weight"]))))
            net[7].bias.copy_(torch.from_numpy(np.asarray(params["fc1/bias"][0])))

            net[9].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["logits/weight"]))))
            net[9].bias.copy_(torch.from_numpy(np.asarray(params["logits/bias"][0])))

            net[11].weight.copy_(torch.from_numpy(np.transpose((np.asarray(params["fc3/weight"])))))
            net[11].bias.copy_(torch.from_numpy(np.asarray(params["fc3/bias"][0])))

            net[13].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc5/weight"]))))
            net[13].bias.copy_(torch.from_numpy(np.asarray(params["fc5/bias"][0])))
    
    return net

def get_Wk17a_base():
    net = nn.Sequential(
        Pad((1, 2, 1, 2)),
        nn.Conv2d(1, 16, 4, 2),
        nn.ReLU(),
        Pad((1, 2, 1, 2)),
        nn.Conv2d(16, 32, 4, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*7*7, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

    params = scipy.io.loadmat(os.path.join(*conf["Model_path"], "wk17a.mat"))

    with torch.no_grad():
            net[1].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv1/weight"]), [3, 2, 0, 1])))
            net[1].bias.copy_(torch.from_numpy(np.asarray(params["conv1/bias"][0])))

            net[4].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["conv2/weight"]), [3, 2, 0, 1])))
            net[4].bias.copy_(torch.from_numpy(np.asarray(params["conv2/bias"][0])))
            
            net[7].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["fc1/weight"]))))
            net[7].bias.copy_(torch.from_numpy(np.asarray(params["fc1/bias"][0])))

            net[9].weight.copy_(torch.from_numpy(np.transpose(np.asarray(params["logits/weight"]))))
            net[9].bias.copy_(torch.from_numpy(np.asarray(params["logits/bias"][0])))

    return net