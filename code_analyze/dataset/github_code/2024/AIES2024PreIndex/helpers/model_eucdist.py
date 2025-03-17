import sys, os
import pickle
sys.path.append("../models")
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import json

from resnet import ResNet, BasicBlock, Bottleneck
from get_layer_names import layer_name_func

def euclidean_dist_model(model_1, model2, layer_names):
    total_l2_dist = 0
    count = 0
    for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        name = name_1.replace(".weight", "")
        if name in layer_names:
            param_1_np = torch.flatten(param_1).cpu().detach() #.cpu().detach().numpy()
            param_2_np = torch.flatten(param_2).cpu().detach() #.cpu().detach().numpy()
            euc_dist = np.linalg.norm(param_1_np - param_2_np) #torch.cdist(param_1, param_2, p=2)
            total_l2_dist += euc_dist/len(param_1_np)
            count += 1
    total_l2_dist /= len(layer_names)
    return total_l2_dist, count

parser = argparse.ArgumentParser()
parser.add_argument('-m1', type=str, help='Model 1 path')
parser.add_argument('-m2', type=str, help='Model 1 path')
parser.add_argument('-save', type=str, help='Save results to directory')
args = parser.parse_args()

model_1_path = args.m1
model_2_path = args.m2
save_to = args.save

model_1 = pickle.load(open(model_1_path, "rb"))
model_1 = model_1["model"]

model_2 = pickle.load(open(model_2_path, "rb"))
model_2 = model_2["model"]

layer_names, filter_neuron_count = layer_name_func(model_1)
total_l2_dist, count = euclidean_dist_model(model_1, model_2, layer_names)

results = {
    "norm_euc_dist" : total_l2_dist,
    "layer_count" : count
}

print(f"Model 1 path: {model_1_path}")
print(f"Model 2 path: {model_2_path}")
print(f"Results: {results}", "\n\n")

save_details = os.path.join(save_to, "euc_dist.pkl")
with open(save_details, "wb") as fp:   
    pickle.dump(results, fp)

with open(os.path.join(save_to, "euc_dist.json"), "w") as f:
    json.dump(results, f)
