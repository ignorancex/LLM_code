
import os

import numpy as np
import tensorflow as tf
import torch

from lolip.variables import get_file_name

def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    torch.manual_seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state

def get_ds_pert_eps(auto_var):
    ds_name = auto_var.get_variable_name("dataset")
    model_name = auto_var.get_variable_name("model")
    norm = auto_var.get_var("norm")
    if norm == 2 and "MLP" in model_name:
        eps = [0.5 * i for i in range(40)]
    elif norm == np.inf and "MLP" in model_name:
        if 'imgnet' in ds_name:
            eps = [0.015 * i for i in range(20)]
        else:
            eps = [0.031 * i for i in range(20)]
    elif norm == 2 and "calcedrepr" in ds_name and ("mnist" in ds_name or 'fashion' in ds_name):
        eps = [1.0 * i for i in range(20)]
    elif norm == 2 and ("mnist" in ds_name or 'fashion' in ds_name):
        eps = [0.5 * i for i in range(10)]
    elif norm == np.inf and ("mnist" in ds_name or 'fashion' in ds_name):
        eps = [0.05 * i for i in range(10)]

    elif norm == np.inf and ("crepr" in ds_name and 'cifar10' in ds_name):
        eps = [1.0 * i for i in range(20)]

    elif norm == 2 and "calcedrepr" in ds_name and ("miniimgnet" in ds_name):
        eps = [0.5 * i for i in range(20)]
    elif norm == np.inf and "calcedrepr" in ds_name and ("miniimgnet" in ds_name):
        eps = [0.1 * i for i in range(10)]

    elif norm == 2 and ("balRes" in ds_name):
        eps = [0.5 * i for i in range(20)]
    elif norm == np.inf and ("balRes" in ds_name):
        eps = [0.005 * i for i in range(20)]

    elif norm == 2 and ("cifar10" in ds_name or 'svhn' in ds_name):
        eps = [0.5 * i for i in range(15)]
    elif norm == np.inf and ("cifar10" in ds_name or 'svhn' in ds_name):
        eps = [0.0155 * i for i in range(12)]

    elif norm == 2 and ("tinyimgnet" in ds_name or "breeds" in ds_name or "miniimgnet" in ds_name):
        eps = [0.5 * i for i in range(20)]
    elif norm == np.inf and ("tinyimgnet" in ds_name or "breeds" in ds_name or "miniimgnet" in ds_name):
        eps = [0.1 * i for i in range(10)]
    else:
        raise NotImplementedError
    return eps

def load_model(auto_var, trnX, trny, tstX, tsty, n_channels, model_dir="./models", device=None):
    model = auto_var.get_var("model", trnX=trnX, trny=trny, n_channels=n_channels, device=device)
    if tstX is not None and tsty is not None:
        model.tst_ds = (tstX, tsty)
    model_path = get_file_name(auto_var)
    if "calcedreprold" in model_path:
        model_path = model_path.replace("calcedreprold", "calcedrepr")
    model_path = model_path.split("-")
    model_path[0] = 'pgd'
    if 'halfclose' in ''.join(model_path) and ('calced' not in ''.join(model_path) and 'crepr' not in ''.join(model_path)):
        model_path[4] = '1.0'
    elif '-ce-vtor2-' in '-'.join(model_path) and 'calced' not in ''.join(model_path) and 'crepr' not in ''.join(model_path):
        # clean training
        if "aug10-imgnet" in '-'.join(model_path):
            pass
        else:
            model_path[4] = '1.0'
    model_path = '-'.join(model_path)
    model_path = model_path.replace("cwl2", "pgd")
    model_path = model_path.replace(
            auto_var.get_variable_name("attack"), "pgd")

    if os.path.exists(os.path.join(model_dir, model_path + '-ep%04d.pt') % model.epochs):
        model_path = os.path.join(model_dir, model_path + '-ep%04d.pt') % model.epochs
    else:
        model_path = os.path.join(model_dir, model_path + '.pt')

    model.load(model_path)
    model.model.cuda()
    return model_path, model
