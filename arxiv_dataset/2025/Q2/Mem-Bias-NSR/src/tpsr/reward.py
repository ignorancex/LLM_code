import numpy as np
import re
import torch
import sys
import sympy as sp
import os
from ControllableNesymres.architectures import bfgs
from ControllableNesymres.architectures.data import extract_variables_from_infix
from ControllableNesymres.dataset.generator import Generator, InvalidPrefixExpression
from ControllableNesymres.architectures.data import de_tokenize

def evaluate_metrics(y_gt, tree_gt, y_pred):
    metrics = []
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_gt,
        },
        metrics='accuracy_l1',
    )
    for k, v in results_fit.items():
        metrics.append(v[0])
    
    return metrics




def compute_nmse(X, y, infix):
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    diffs = []
    vars_list = extract_variables_from_infix(infix)
    vars_list = [x for x in vars_list if x != "constant"]
    indeces = [int(x[2:])-1 for x in vars_list]
    
    try:
        expr = sp.sympify(infix)
    except:
        print("Could not sympify infix")
        return None

    try:
        f = sp.lambdify(vars_list,expr)
    except:
        print("Could not lambdify infix")
        return None

    for b in range(X.shape[0]):
        for i in range(X.shape[1]):
            x = X[b,i,:len(vars_list)]
            y_hat = f(*x)
            diffs.append(y[b,i] - y_hat)
    

    mean_y = np.mean(y)
    if abs(mean_y) < 1e-06:
        print("Normalizing by a small value")
    loss = (np.mean(np.square(diffs)))/mean_y
    return loss




def compute_reward_nesymres(X, y, state, cfg_params):  
    penalty = -2

    print()
    print("state",state)
    cfg_params.id2word[3] = "constant"

    if type(state) != list:
        state = state.tolist()

    if "partition" in cfg_params.word2id:
        if cfg_params.word2id["partition"] in state:
            partition_index = state.index(cfg_params.word2id["partition"])
            prefix = de_tokenize(state[partition_index + 1:], cfg_params.id2word)
        else:
            prefix = de_tokenize(state[1:], cfg_params.id2word)
    else:
        prefix = de_tokenize(state[1:], cfg_params.id2word)
    
    print("prefix",prefix)


    try:
        infix = Generator.prefix_to_infix(prefix, coefficients=["constant"], variables=cfg_params.total_variables)
        infix = infix.format(constant="constant")
    except InvalidPrefixExpression:
        print("Cannot prefix to infix" + str(prefix))
        reward = penalty
        return None, reward , None
    
    # state = torch.tensor(state, requires_grad=False)
    print("infix",infix)

    
    if "constant" in infix:
        pred_w_c, _, loss_bfgs, _ = bfgs.bfgs(
            infix, X, y, cfg_params
        )
        print("pred_w_c",pred_w_c)
        print("nmse_loss: ",compute_nmse(X,y,str(pred_w_c)))
        print()


        if np.isnan(loss_bfgs):
            print("Warning all nans")
            reward = penalty
        else:
            lam = 0.1
            eps = 1e-9


            nmse = loss_bfgs / ( torch.mean( (y.reshape(-1))**2 ).item() + eps)
            # reward = 1/(1+loss_bfgs)
            reward = 1/(1+nmse) + lam * np.exp( -(len(state) - 2) / 200 )
        return loss_bfgs, reward , str(pred_w_c)

    else:
        print("nmse_loss: ",compute_nmse(X,y,infix))
        print("no constants")
        print()
        lam = 0.1
        eps = 1e-9
        nmse = compute_nmse(X,y,infix)
        if nmse is np.nan:
            return None, penalty, infix
        reward = 1/(1+nmse) + lam * np.exp( -(len(state) - 2) / 200 )
        return None, reward , infix