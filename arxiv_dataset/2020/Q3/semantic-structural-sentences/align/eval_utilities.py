import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

from torch.autograd import Variable
from tqdm import tqdm

from annoy import AnnoyIndex

from align_models.LinearSpaceMapper import LinearSpaceMapper, ConstrainedLinearSpaceMapper
from align_models.CLR import CyclicLR
from align_models.DualSpaceDataLoader import DualSpaceDataLoader
from align_models.EigenvectorSimilarity import EigenvectorSimilarity
from align_models.BottleneckSimilarity import BottleneckSimilarity


def calculate_nearest_k(_sim_mat, _k):
    _sim_mat = _sim_mat.detach().cpu().numpy()
    sorted_mat = -np.partition(-_sim_mat, _k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:_k]
    out = np.mean(nearest_k, axis=1)
    return torch.tensor(out).cuda()


def compute_csls_sim(_embed_src, _embed_trg, _k, _k2):
    torch.cuda.empty_cache()
    _sim_mat = torch.cosine_similarity(_embed_src, _embed_trg, dim=-1, eps=1e-6)
    _nearest1 = calculate_nearest_k(_sim_mat, _k)
    _nearest2 = calculate_nearest_k(_sim_mat.T, _k)
    _nearest3 = calculate_nearest_k(_sim_mat, _k2)
    _nearest4 = calculate_nearest_k(_sim_mat.T, _k2)
    _csls_sim_mat_1 = 2 * _sim_mat.T - _nearest1
    _csls_sim_mat_1 = _csls_sim_mat_1.T - _nearest2
    _csls_sim_mat_2 = 2 * _sim_mat.T - _nearest3
    _csls_sim_mat_2 = _csls_sim_mat_2.T - _nearest4
    return _csls_sim_mat_1, _csls_sim_mat_2


def calc_knn(_space, _preds, _n):
    knn = NearestNeighbors(n_neighbors=_n, radius=1.0, algorithm='auto',
                           leaf_size=30, metric='minkowski', p=2, n_jobs=-1)
    knn.fit(_space.detach().cpu().numpy())
    _res = []
    pred_vecs = _preds.detach().cpu().numpy()
    print('Computing nearest neighbors')
    for _vec in tqdm(pred_vecs):
        _res.append(knn.kneighbors(_vec.reshape(1, -1), 10, return_distance=False))
    return _res


def find_similarities(_model, _valid_x, _valid_y, _n1, _n2):
    # For the validation set
    torch.cuda.empty_cache()
    preds = _model.forward(_valid_x)
    actuals = _model.sent_weights.detach().cuda()
    ind_vecs_5 = []
    ind_vecs_10 = []
    _tk5 = []
    _tk10 = []
    _cs5 = []
    _cs10 = []
    torch.cuda.empty_cache()
    #knn_res = calc_knn(actuals, preds, 10)
    # Change this to pull to numpy and detach after all iters of cos are done
    for pv_idx in tqdm(range(len(preds))):
        pred_vec = preds[pv_idx]
        sims = torch.cosine_similarity(pred_vec, actuals, dim=-1, eps=1e-6)
        topk_5, indices = sims.topk(_n1)
        ind_vecs_5.append(indices.cpu().numpy())
        _tk5.append(topk_5.detach().cpu().numpy())
        topk_10, indices = sims.topk(_n2)
        ind_vecs_10.append(indices.cpu().numpy())
        _tk10.append(topk_10.detach().cpu().numpy())
    return ind_vecs_5, ind_vecs_10, _tk5, _tk10, _cs5, _cs10 #, knn_res


def find_similarities_annoy(_model, _valid_x, _valid_y, _n1, _n2):
    # For the validation set
    torch.cuda.empty_cache()
    preds = _model.forward(_valid_x)
    actuals = _model.sent_weights.detach()
    preds_collect = preds.detach().cpu().numpy()
    actuals_collect = actuals.cpu().numpy()
    f = len(preds_collect[0])
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
    for i in range(len(actuals_collect)):
        v = actuals_collect[i]
        t.add_item(i, v)
    print('Building Approximate Nearest Neighbors')
    t.build(500)
    print('Oh Yeah!')
    ind_vecs_5 = []
    ind_vecs_10 = []
    for pv_idx in tqdm(range(len(preds_collect))):
        pred_vec = preds_collect[pv_idx]
        ind_vecs_5.append(t.get_nns_by_vector(pred_vec, _n1, search_k=-1, include_distances=False))
        ind_vecs_10.append(t.get_nns_by_vector(pred_vec, _n2, search_k=-1, include_distances=False))
    return ind_vecs_5, ind_vecs_10


def check_if_hit(_x, _y):
    _res = 0
    if _x in _y:
        _res += 1
    return _res


def find_precision(_model, _valid_src, _valid_trg, _valid_path, _annoy):
    """
    Computes precision at 5 and 10.

    :param _model: A trained mapping model.
    :param _valid_src: The validation source data.
    :param _valid_path: The path to human readable validation CSV.
    :param _annoy: Triggers the use of approximate nearest neighbors.
    :return: Integers, P@5 and P@10.
    """
    if _annoy:
        ivs, ivs2 = find_similarities_annoy(_model, _valid_src, _valid_trg, 5, 10)
        valid_set = pd.read_csv(_valid_path)
        valid_set['top_pred'] = [t[0] for t in ivs]
        valid_set['pred_idx_5'] = ivs
        valid_set['pred_idx_10'] = ivs2
        valid_set['hits_5'] = valid_set.apply(lambda x: check_if_hit(x.sent_id, x.pred_idx_5), axis=1)
        valid_set['hits_10'] = valid_set.apply(lambda x: check_if_hit(x.sent_id, x.pred_idx_10), axis=1)
        _hits_at_5 = float(valid_set.hits_5.sum()) / len(valid_set)
        _hits_at_10 = float(valid_set.hits_10.sum()) / len(valid_set)
        print('Hits@5: {m}'.format(m=_hits_at_5))
        print('Hits@10: {m}'.format(m=_hits_at_10))
    else:
        ivs, ivs2, tk5, tk10, cs5, cs10 = find_similarities(_model, _valid_src, _valid_trg, 5, 10)
        valid_set = pd.read_csv(_valid_path)
        valid_set['top_pred'] = [t[0] for t in ivs]
        valid_set['pred_idx_5'] = ivs
        valid_set['pred_idx_10'] = ivs2
        # TODO: the plan here is to plot the similarities across approaches like in the KG entity alignment paper
        valid_set['top_5_sim'] = tk5
        valid_set['top_10_sim'] = tk10
        valid_set['hits_5'] = valid_set.apply(lambda x: check_if_hit(x.sent_id, x.pred_idx_5), axis=1)
        valid_set['hits_10'] = valid_set.apply(lambda x: check_if_hit(x.sent_id, x.pred_idx_10), axis=1)
        # valid_set['knns'] = knns
        _hits_at_5 = float(valid_set.hits_5.sum()) / len(valid_set)
        _hits_at_10 = float(valid_set.hits_10.sum()) / len(valid_set)
        print('Hits@5: {m}'.format(m=_hits_at_5))
        print('Hits@10: {m}'.format(m=_hits_at_10))
    return _hits_at_5, _hits_at_10, valid_set

