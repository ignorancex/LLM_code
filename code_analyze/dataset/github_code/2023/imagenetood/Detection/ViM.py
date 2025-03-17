from sklearn.covariance import EmpiricalCovariance
import numpy as np
from numpy.linalg import norm, pinv
import torch
from scipy.special import logsumexp


def vim(training_features, eval_features, num_class, last_layer):
    w = last_layer.weight.cpu().detach().numpy()
    b = last_layer.bias.cpu().detach().numpy()
    training_features = training_features.cpu().detach().numpy()
    eval_features = eval_features.cpu().detach().numpy()
    u = -np.matmul(pinv(w), b)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(training_features - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[num_class:]]).T
    )

    train_logits = training_features @ w.T + b
    eval_logits = eval_features @ w.T + b

    vlogit_id_train = norm(np.matmul(training_features - u, NS), axis=1)
    alpha = train_logits.max(axis=1).mean() / vlogit_id_train.mean()

    vlogit_id_val = norm(np.matmul(eval_features - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(eval_logits, axis=-1)
    return -vlogit_id_val + energy_id_val

def vlogits(training_features, eval_features, num_class, last_layer):
    w = last_layer.weight.cpu().detach().numpy()
    b = last_layer.bias.cpu().detach().numpy()
    training_features = training_features.cpu().detach().numpy()
    eval_features = eval_features.cpu().detach().numpy()
    u = -np.matmul(pinv(w), b)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(training_features - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[num_class:]]).T
    )

    train_logits = training_features @ w.T + b
    eval_logits = eval_features @ w.T + b

    vlogit_id_train = norm(np.matmul(training_features - u, NS), axis=1)
    alpha = train_logits.max(axis=1).mean() / vlogit_id_train.mean()

    vlogit_id_val = norm(np.matmul(eval_features - u, NS), axis=-1) * alpha
    vlogit_id_train = vlogit_id_train * alpha
    return vlogit_id_train, vlogit_id_val
