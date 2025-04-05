import os
import logging

import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from mkdir_p import mkdir_p
import joblib

from lolip.tools.nn_utils import get_faiss_index
from .utils import set_random_seed, load_model
from .ood_robustness import get_rob


ood_nn_idx_cache_dir = "./data/caches/ood_nn_idx"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.WARNING, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def run_ood_robustness_correct(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_state = set_random_seed(auto_var)
    trnX, trny, tstX, tsty, ood_features = auto_var.get_var("dataset")
    oodX = np.concatenate((ood_features[0], ood_features[1]), axis=0)
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]
    ds_name = auto_var.get_variable_name("dataset")
    norm_name = auto_var.get_variable_name("norm")
    norm = auto_var.get_var("norm")

    mkdir_p(ood_nn_idx_cache_dir)
    cache_file = os.path.join(ood_nn_idx_cache_dir, f"{ds_name}_{norm_name}.pkl")

    if os.path.exists(cache_file):
        I = joblib.load(cache_file)
    else:
        index = get_faiss_index(int(np.prod(trnX.shape[1:])), norm)

        if ("calced" not in ds_name) and ("aug10-imgnet" in ds_name):
            from torchvision import transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            def np_normalize(X):
                X = X.transpose(0, 3, 1, 2)
                return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)
            index.add(np_normalize(trnX).astype(np.float32).reshape(len(trnX), -1))
            _, I = index.search(np_normalize(oodX).astype(np.float32).reshape(len(oodX), -1), k=1)
            I = np.unique(I.reshape(-1))
            joblib.dump(I, cache_file)
            
        else:
            index.add(trnX.astype(np.float32).reshape(len(trnX), -1))
            _, I = index.search(oodX.astype(np.float32).reshape(len(oodX), -1), k=1)
            I = np.unique(I.reshape(-1))
            joblib.dump(I, cache_file)

    is_img_data = True
    if 'is_img_data' in auto_var.inter_var:
        is_img_data = auto_var.inter_var['is_img_data']
    elif ('crepr-' in ds_name) or ('calcedrepr-' in ds_name):
        is_img_data = False

    result = {}
    try:
        model_path, model = load_model(
            auto_var, trnX, trny, None, None, n_channels, model_dir="./models/out_of_sample/", device=device)
        model.model.to(device)
        result['model_path'] = model_path
    except:
        raise ValueError("Model is not trained yet")

    trn_pred = model.predict(trnX, is_img_data=is_img_data)
    tst_pred = model.predict(tstX, is_img_data=is_img_data)
    result['trn_acc'] = (trn_pred == trny).mean()
    result['tst_acc'] = (tst_pred == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    correct_idx = np.where(trn_pred == trny)[0]
    I = np.intersect1d(correct_idx, I)

    clip_min, clip_max = None, None
    if len(trnX.shape) == 4:
        clip_min, clip_max = 0, 1

    get_rob(auto_var, model, trnX, trny, I, n_classes,
            clip_min, clip_max, random_state=random_state, n_samples=300,
            result_dict=result, is_img_data=is_img_data)

    result['adv_preds'] = model.predict(result['adv_trnX'], is_img_data=is_img_data)

    return result
