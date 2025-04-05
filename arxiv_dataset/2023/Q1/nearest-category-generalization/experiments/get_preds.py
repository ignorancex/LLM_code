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

def run_get_preds(auto_var):
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

    model_path, model = load_model(
        auto_var, trnX, trny, tstX, tsty, n_channels, model_dir=base_model_dir, device=device)
    model.model.to(device)
    result['model_path'] = model_path

    result['trn_pred'] = model.predict(trnX)
    result['tst_pred'] = model.predict(tstX)
    result['trn_acc'] = (result['trn_pred'] == trny).mean()
    result['tst_acc'] = (result['tst_pred'] == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    return result
