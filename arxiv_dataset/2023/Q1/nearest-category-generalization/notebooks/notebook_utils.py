import sys
sys.path.append("../")
import inspect

import torch
import numpy as np
import faiss

from lolip.models.torch_utils import archs

def get_model_name(model_name, eps=None):
    if 'snnl' in model_name:
        ret = model_name.split("ce-")[0]
    elif 'pcaellipbatchada' in model_name:
        ret = 'ellipsoid'
    elif 'batchsubvor100randada' in model_name:
        ret = 'sub-Voronoi'
        
    elif 'nnreg' in model_name:
        if 'nnreg6' in model_name:
            ret = "nnreg6"
        elif 'nnreg100' in model_name:
            ret = "nnreg100"
        elif 'nnreg10' in model_name:
            ret = "nnreg10"
        elif 'nnreg20' in model_name:
            ret = "nnreg20"
        elif 'nnreg40' in model_name:
            ret = "nnreg40"
        else:
            ret = "nnreg"

        if 'cusgrad' in model_name:
            ret = ret + "+our"
    elif 'randmix' in model_name:
        if 'randmix2' in model_name:
            ret = 'randmix2'
        elif 'randmix4' in model_name:
            ret = 'randmix4'
        elif 'randmix16' in model_name:
            ret = 'randmix16'
        else:
            ret = "randmix"
    elif 'mixup' in model_name:
        if 'mixup2' in model_name:
            ret = 'mixup2'
        elif 'mixup4' in model_name:
            ret = 'mixup4'
        elif 'mixup16' in model_name:
            ret = 'mixup16'
        else:
            ret = "mixup"
        if 'nntor' in model_name:
            ret = "nn " + ret
    elif 'adv' in model_name:
        ret = "AT"
    elif "cusgradtrades" in model_name:
        if "cusgradtrades6" in model_name:
            ret = "DBS6"
        elif "cusgradtrades10" in model_name:
            ret = "DBS10"
        elif "cusgradtrades20" in model_name:
            ret = "DBS20"
        else:
            ret = "DBS"
    elif "cusgradbeta" in model_name:
        ret = 'variAT'
    elif "trades" in model_name:
        ret = "TRADES"
    else:
        ret = "natural"
        
    if 'aug' in model_name:
        ret = "aug" + ret
    
    #if 'halfclose' not in model_name and 'nat' not in ret:
    #    ret = ret + f"({eps:.3f})"
    if 'halfclose' in model_name and 'nat' not in ret:
        if 'halfclose20' in model_name:
            ret = ret + f"($\lambda$=1/20)"
        elif 'halfclose3' in model_name:
            ret = ret + f"($\lambda$=1/3)"
        elif 'halfclose50' in model_name:
            ret = ret + f"($\lambda$=1/50)"
        elif 'halfclose10' in model_name:
            ret = ret + f"($\lambda$=1/10)"
        elif 'halfclose2' in model_name:
            ret = ret + f"($\lambda$=1/2)"
        elif 'halfclose5' in model_name:
            ret = ret + f"($\lambda$=1/5)"
        elif 'halfclose.5' in model_name:
            ret = ret + f"($\lambda$=2)"
        elif 'halfclose.75' in model_name:
            ret = ret + f"($\lambda$=4/3)"
        else:
            ret = ret + f"($\lambda$=1)"
    return ret

def get_arch(model_name):
    if "aug" in model_name:
        return model_name.split("-")[3]
    return model_name.split("-")[2]

def get_ds_name(name):
    if 'cifar10wo0' in name:
        ret = "cifar10wo0"
    elif "mnistwo9" in name:
        ret = 'mnistwo9'
    elif "tinyimgnetResnet50worand20" in name:
        ret = 'tinywo20'
    elif "imgnetsubset100resnet50worand20" in name:
        ret = "imgnetsubwo20"
    elif "imgnetsubset100resnext" in name:
        ret = "imgnetsubwo20"
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    if "calced" in name:
        if "Resnet" in name:
            ret = "Resnet " + ret
        elif "Resnext" in name:
            ret = "Resnext " + ret
        elif "WRN" in name:
            ret = "WRN " + ret
        ret = "fet " + ret
    return ret

#def get_n_classes(ds_name):
#    if 'tiny' in ds_name and 'worand20'

def get_model(arch_name, model_path, X, y):
    n_classes = len(np.unique(y))
    n_channels = X.shape[1]
    
    arch_fn = getattr(archs, arch_name)
    arch_params = dict(n_classes=n_classes, n_channels=n_channels)
    if 'n_features' in inspect.getfullargspec(arch_fn)[0]:
        arch_params['n_features'] = X.shape[1:]
    model = arch_fn(**arch_params)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model


def get_faiss_nn_output(X, tstX, norm):
    X = X.reshape(len(X), -1)
    if norm == 2:
        index = faiss.IndexFlatL2(X.shape[1])
    elif norm == np.inf:
        index = faiss.IndexFlat(X.shape[1], faiss.METRIC_Linf)
    elif norm == 1:
        index = faiss.IndexFlat(X.shape[1], faiss.METRIC_L1)
    
    index.add(X)
    D, I = index.search(tstX.reshape(len(tstX), -1), k=1)
    
    if norm == 2:
        D = np.sqrt(D)
    del index
    return D, I

def get_nn_pred(auto_var, d, add_per_lbl=0, n_jobs=16, get_tst_acc=False):
    auto_var.set_variable_value("random_seed", int(d['random_seed']))
    trnX, trny, tstX, tsty, oos_features = auto_var.get_var_with_argument("dataset", d['dataset'])
    norm = auto_var.get_var_with_argument("norm", d['norm'])
    oos_repr = np.concatenate((d['oos_trn_repr'], d['oos_tst_repr']), axis=0)
    if "worand" in d['dataset']:
        rest_ys = np.concatenate(auto_var.inter_var['rest_ys'], axis=0).astype(np.int)
    else:
        rest_ys = np.ones(len(oos_repr), np.int) * 9

    X, y = d['trn_repr'], trny
    if add_per_lbl != 0:
        lbls_to_idx = {yi: i + len(np.unique(trny)) for i, yi in enumerate(sorted(np.unique(rest_ys)))}
        
        random_state = np.random.RandomState(0)
        for i in sorted(np.unique(rest_ys)):
            tempX = oos_repr[rest_ys==i]
            ind = random_state.choice(np.arange(len(tempX)), size=add_per_lbl, replace=False)
            
            X = np.concatenate((X, tempX[ind], ), axis=0)
            tempy = np.asarray([lbls_to_idx[yi] for yi in rest_ys[ind]])
            y = np.concatenate((y, tempy))
            
            oos_repr = np.delete(oos_repr, np.where(rest_ys == i)[0][ind], axis=0)
            rest_ys = np.delete(rest_ys, np.where(rest_ys == i)[0][ind], axis=0)

    if get_tst_acc:
        Xrepr = np.concatenate((oos_repr, d['tst_repr']))
        dist, ind = get_faiss_nn_output(X, Xrepr, norm, )
        return dist[:len(oos_repr)], ind[:len(oos_repr)], y, dist[len(oos_repr):], ind[len(oos_repr):]
    else:
        dist, ind = get_faiss_nn_output(X, oos_repr, norm, )
        return dist, ind, y