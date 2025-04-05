import os
import logging

import torch
from bistiming import Stopwatch
from mkdir_p import mkdir_p
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed, load_model
from lolip.variables import get_file_name

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

base_model_dir = './models/out_of_sample/'
base_data_cache = './data/caches/'
pcaellip_dir = os.path.join(base_data_cache, "learned_pcas")

def run_oos_repr(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_state = set_random_seed(auto_var)
    trnX, trny, tstX, tsty, oos_features = auto_var.get_var("dataset")
    print(trnX.shape)
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    if auto_var.get_variable_name("dataset") == "mnistwo9":
        assert n_classes == 9
    elif auto_var.get_variable_name("dataset") == "cifar10wo9":
        assert n_classes == 9
    n_channels = trnX.shape[-1]

    result = {}
    if 'leave_out_cls' in auto_var.inter_var:
        result['leave_out_cls'] = auto_var.inter_var['leave_out_cls']
    if 'cls_map' in auto_var.inter_var:
        result['cls_map'] = auto_var.inter_var['cls_map']
    if 'rest_ys' in auto_var.inter_var:
        result['rest_ys'] = auto_var.inter_var['rest_ys']

    #multigpu = False
    multigpu = True if torch.cuda.device_count() > 1 else False

    is_img_data = True
    if 'is_img_data' in auto_var.inter_var:
        is_img_data = auto_var.inter_var['is_img_data']

    try:
        model_path, model = load_model(
            auto_var, trnX, trny, tstX, tsty, n_channels, model_dir=base_model_dir, device=device)
        model.model.to(device)
        result['model_path'] = model_path
    except:
        logger.info("Model not trained yet, retrain the model")
        mkdir_p(base_model_dir)
        result['model_path'] = os.path.join(
            base_model_dir, get_file_name(auto_var) + '-ep%04d.pt')
        result['model_path'] = result['model_path'].replace(
            auto_var.get_variable_name("attack"), "pgd")

        ds_name = auto_var.get_variable_name("dataset")
        norm_name = auto_var.get_variable_name("norm")
        random_seed = auto_var.get_var("random_seed")

        model_name = auto_var.get_variable_name("model")
        idx_data_cache_dir = os.path.join(base_data_cache, "nn_idx")
        data_cache_dir = base_data_cache
        mkdir_p(idx_data_cache_dir)
        mkdir_p(data_cache_dir)

        normalized = False
        if "aug06" in ds_name:
            normalized = True

        if "pcaellip" in model_name:
            mkdir_p(pcaellip_dir)
            cache_filename = os.path.join(pcaellip_dir, f"{ds_name}_%s_{random_seed}.pkl")
            train_type = model_name.split("-")[-1]
            train_type = train_type.replace("ada", "")
            idx_cache_filename = os.path.join(idx_data_cache_dir, f"info_{ds_name}_{norm_name}_{train_type}_{random_seed}.pkl")
        else:
            cache_filename = os.path.join(data_cache_dir, f"{ds_name}_{norm_name}_{random_seed}.pkl")
            idx_cache_filename = os.path.join(idx_data_cache_dir, f"{ds_name}_{norm_name}_{random_seed}.pkl")

        model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu,
                n_channels=n_channels, device=device)
        model.tst_ds = (tstX, tsty)

        sample_weights = None
        if 'shot' in auto_var.get_variable_name("dataset"):
            class_weight = len(trny) / np.bincount(trny) / n_classes
            sample_weights = class_weight[trny]

        with Stopwatch("Fitting Model", logger=logger):
            history = model.fit(trnX, trny, idx_cache_filename=idx_cache_filename,
                                cache_filename=cache_filename, sample_weights=sample_weights,
                                is_img_data=is_img_data, normalized=normalized)
        model.save(result['model_path'])
        result['model_path'] = result['model_path'] % model.epochs
        result['history'] = history

    result['trn_repr'] = model.get_repr(trnX)
    result['tst_repr'] = model.get_repr(tstX)
    if oos_features is not None:
        result['oos_trn_repr'] = model.get_repr(oos_features[0])
        result['oos_tst_repr'] = model.get_repr(oos_features[1])

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    if oos_features is not None:
        result['oos_trn_pred'] = model.predict(oos_features[0])
        result['oos_tst_pred'] = model.predict(oos_features[1])
        if len(oos_features) > 2:
            result['oos_trn_ori_y'] = oos_features[2]
            result['oos_tst_ori_y'] = oos_features[3]

    clip_min, clip_max = None, None
    if len(trnX.shape) == 4 and is_img_data:
        clip_min, clip_max = 0, 1

    #get_rob_dif_eps(auto_var, model, trnX, trny, tstX, tsty, n_classes,
    #                clip_min, clip_max, random_state, device, result, is_img_data=is_img_data)

    return result
