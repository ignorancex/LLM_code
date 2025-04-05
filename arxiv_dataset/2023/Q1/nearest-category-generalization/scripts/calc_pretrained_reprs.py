#!/usr/bin/env python
import os
from os.path import join
import gc

import dill
import joblib
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import torchvision
import torch.nn.functional as F

from madry_robustness_resnet import resnet50
from lolip.extra_ood_utils import get_ood_data_paths
from lolip.variables import auto_var


MODELPATH = "./models/out_of_sample"

def get_resnet_repr(resnet, x):
    _, ret = resnet(x, with_latent=True)
    return ret

def get_madry_model(model_path):
    model = resnet50()
    res = torch.load(model_path, pickle_module=dill)
    state_dict = {}
    sd = res['model'] if 'model' in res else res['state_dict']
    for k, v in sd.items():
        if "module.model." in k:
            state_dict[k.replace("module.model.", "")] = v
    model.load_state_dict(state_dict)
    return model


def get_outputs(model, dset, batch_size=64, device="cuda", version="repr"):
    model.eval().to(device)
    if isinstance(dset, np.ndarray):
        dset = torch.utils.data.TensorDataset(
            torch.from_numpy(dset.transpose(0, 3, 1, 2)).float(),
            torch.ones(len(dset))
        )

    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=24)

    ret = []
    for (x, _) in tqdm(loader):
        with torch.no_grad():
            if version == "repr":
                output = get_resnet_repr(model, x.to(device))
            elif version == "pred":
                output = F.softmax(model(x.to(device)), dim=1)
        ret.append(output.cpu().numpy())
    del loader
    return np.concatenate(ret, axis=0)


def get_corrupted_reprs():
    image_files, ood_names = get_ood_data_paths("cifar10", "./data/cifar-ood/")

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)
    
    ds_name = "cifar10"
    trnX, _, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)
    trnX, tstX = np_normalize(trnX), np_normalize(tstX)

    model_path = "./notebooks/pretrained/madry_robustness/cifar_nat.pt"
    model = get_madry_model(model_path)
    model_name = "madry_nat"

    reprs = {}

    key = (ds_name, model_name)
    reprs.setdefault(key, {})

    trn_reprs = get_outputs(model, trnX, batch_size=64, device="cuda", version="repr")
    reprs[key]['trn'] = trn_reprs
    tst_reprs = get_outputs(model, tstX, batch_size=64, device="cuda", version="repr")
    reprs[key]['tst'] = tst_reprs

    for ood_name, image_file in tqdm(zip(ood_names, image_files), total=len(ood_names), desc="[get oodXs]"):
        tX = np_normalize((np.load(image_file).astype(np.float32) / 255.))
        crepr = get_outputs(model, tX, batch_size=64, device="cuda", version="repr")
        reprs[key][ood_name] = crepr
    
    reprs_file = f"./notebooks/nb_results/reprs-madry-cifar10.pkl"
    joblib.dump(reprs, reprs_file)
    print("done.")

    

def get_nat_reprs():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)

    trnX, trny, tstX, tsty, _ = auto_var.get_var_with_argument("dataset", "cifar10")
    trnX = np_normalize(trnX)
    tstX = np_normalize(tstX)

    model_path = "./notebooks/pretrained/madry_robustness/cifar_nat.pt"
    model = get_madry_model(model_path)

    ret = {}

    trn_pred = get_outputs(model, trnX, batch_size=64, device="cuda", version="pred")
    tst_pred = get_outputs(model, tstX, batch_size=64, device="cuda", version="pred")
    ret['trn_acc'] = np.mean(np.argmax(trn_pred, axis=1) == trny)
    ret['tst_acc'] = np.mean(np.argmax(tst_pred, axis=1) == tsty)
    print(ret)

    ret['trn_repr'] = get_outputs(model, trnX, batch_size=64, device="cuda")
    ret['tst_repr'] = get_outputs(model, tstX, batch_size=64, device="cuda")
    ret['oos_trn_repr'] = ret['trn_repr'][:10]
    ret['oos_tst_repr'] = ret['tst_repr'][:10]

    output_path = "./results/oos_repr/madry-cifar10-cifar_nat.pt.pkl"
    joblib.dump(ret, output_path)


if __name__ == "__main__":
    #get_nat_reprs()
    get_corrupted_reprs()
