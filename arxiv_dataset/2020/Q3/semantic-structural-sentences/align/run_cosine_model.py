import argparse
import numpy as np
import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from time import sleep

from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from align_models.LinearSpaceMapper import LinearSpaceMapper
from align_models.NonLinearSpaceMapper import NonLinearSpaceMapper
from align_models.CLR import CyclicLR
from align_models.NegativeSampleDataLoader import NegativeSampleDataLoader
from align_models.EigenvectorSimilarity import EigenvectorSimilarity
from align_models.BottleneckSimilarity import BottleneckSimilarity

from eval_utilities import find_precision, find_similarities
try:
    from text_complete import text_results
except ImportError:
    pass


def evaluate(_model, _valid_x, _valid_y, _shape, _valid_path, _w, _ann, _lm):
    _valid_new = _model.forward(_valid_x.cuda())
    _valid_old = _model.prior_space(_valid_x.cuda())
    _valid_trg = _model.sent_embedding(_valid_y.cuda()).squeeze()
    sims_post = torch.cosine_similarity(_valid_new, _valid_trg, dim=0, eps=1e-6)
    average_sim = torch.mean(sims_post).item()
    print('Average cosine similarity: {a}'.format(a=average_sim))
    if _ann:
        p_5, p_10, valid_df = find_precision(_model, _valid_x.cuda(), _valid_y.cuda(), _valid_path, True)
    else:
        p_5, p_10, valid_df = find_precision(_model, _valid_x.cuda(), _valid_y.cuda(), _valid_path, False)
    if _w:
        rel_hits = valid_df.groupby('relation').agg({'hits_5': 'sum', 'hits_10': 'sum'})
        print(rel_hits)
        rel_hits.to_csv('valid_{l}_res.csv'.format(l=_lm), index=False)
    if _shape == "True":
        ev_prior = EigenvectorSimilarity(_valid_old, _valid_trg)
        ev_prior_delta = ev_prior.compare_spaces()
        ev_post = EigenvectorSimilarity(_valid_new, _valid_trg)
        ev_post_delta = ev_post.compare_spaces()
        print('EV prior: {pr}, EV post: {po}'.format(pr=ev_prior_delta, po=ev_post_delta))
        bs_prior = BottleneckSimilarity(_valid_old, _valid_trg)
        bs_prior_delta = bs_prior.compute_distance()
        bs_post = BottleneckSimilarity(_valid_new, _valid_trg)
        bs_post_delta = bs_post.compute_distance()
        print('BS prior: {br}, BS post: {bo}'.format(br=bs_prior_delta, bo=bs_post_delta))
        _eval_metrics = [ev_prior_delta, ev_post_delta, bs_prior_delta, bs_post_delta, average_sim, p_5, p_10]
    else:
        _eval_metrics = [average_sim, p_5, p_10]
    return _eval_metrics, p_5, p_10


def run(_sent_space, _ent_space, _rel_space,
        _agg, _norm,
        _train, _valid, _batch_sz,
        _init_lr, _epochs, _es_tol, _beta, _ns, _tune):
    _model = LinearSpaceMapper(_sent_space, _ent_space, _rel_space, _agg, _norm, _beta, _tune)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = torch.optim.Adam(_model.parameters(), lr=_init_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=1, verbose=True)
    if torch.cuda.is_available():
        _model.cuda()
    val_prior = 10
    es_tol = 0
    data = NegativeSampleDataLoader(_train, _valid, _batch_sz, _ns)
    _valid_x, _valid_y = data.get_validation()
    for e in tqdm(range(_epochs)):
        losses = []
        batch_x, batch_y_true, batch_y_false, idx_labels = data.create_all_batches()
        for idx in range(len(batch_x)):
            #_model.orthogonalize()
            _model.zero_grad()
            if torch.cuda.is_available():
                inputs = batch_x[idx].cuda()
                true_vals = _model.sent_embedding(batch_y_true[idx].cuda()).squeeze()
                neg_vals = _model.sent_embedding(batch_y_false[idx].cuda()).squeeze()
                targets = idx_labels[idx].cuda()
            outputs = _model(inputs)
            loss = criterion(torch.cat([outputs, outputs]), torch.cat([true_vals, neg_vals]), targets)
            # loss = criterion(outputs, torch.cat([true_vals, neg_vals]), targets)
            loss.backward()
            optimizer.step()
            del inputs
            del true_vals
            del neg_vals
            del targets
            torch.cuda.empty_cache()
            losses.append(loss.item())
        _model.eval()
        if torch.cuda.is_available():
            inputs = _valid_x.cuda()
            labels = _model.sent_embedding(_valid_y.cuda()).squeeze()
            targets = torch.tensor([1] * _valid_x.shape[0]).cuda()
        outputs = _model(inputs)
        val_loss = criterion(outputs, labels, targets)
        scheduler.step(val_loss)
        if torch.abs(val_loss - val_prior) < 0.001:
            es_tol += 1
        else:
            val_prior = val_loss
        if es_tol >= _es_tol:
            print('Reached early stopping criteria')
            break
        if e % 1 == 0 and e != 0:
            print('Epoch number: {n}'.format(n=e))
            print('Mean epoch training loss: {l}'.format(l=np.mean(losses)))
            print('Validation loss: {l}'.format(l=val_loss))
            print('Patience threshold: {p}'.format(p=es_tol))
    del data
    torch.cuda.empty_cache()
    return _model, _valid_x, _valid_y


if __name__ == "__main__":
    config_name = str(sys.argv[1])
    print(config_name)
    with open(config_name, 'rb') as json_file:
        config = json.load(json_file)
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    if config['language_model'] == 'dct':
        sent_emb_path = os.path.join(wd,
                                     'src/sent-embeddings/{lm}/{d}_{lm}_{k}_space.pt'.format(lm=config['language_model'],
                                                                                             k=config['komp'],
                                                                                             d=config['data_source']))
    elif config['language_model'] == 'infersent':
        sent_emb_path = os.path.join(wd,
                                     'src/sent-embeddings/{lm}/{d}_{mn}_space.pt'.format(
                                         lm=config['language_model'],
                                         mn=config['sentname'],
                                         d=config['data_source']))
    else:
        sent_emb_path = os.path.join(wd, 'src/sent-embeddings/{lm}/{d}_{lm}_space.pt'.format(lm=config['language_model'],
                                                                                             d=config['data_source']))
    ent_emb_path = os.path.join(wd, 'src/kg-embeddings/data/entities_{e}_{d}_{a}.pt'.format(e=config['kg_embedding'],
                                                                                            d=config['kg_dimension'],
                                                                                            a=config['data_source']))
    rel_emb_path = os.path.join(wd, 'src/kg-embeddings/data/relations_{e}_{d}_{a}.pt'.format(e=config['kg_embedding'],
                                                                                             d=config['kg_dimension'],
                                                                                             a=config['data_source']))
    sent_emb = torch.load(sent_emb_path)
    ent_emb = torch.load(ent_emb_path)
    rel_emb = torch.load(rel_emb_path)
    print(sent_emb.shape)
    print(ent_emb.shape)
    print(rel_emb.shape)

    train_path = os.path.join(wd, 'data/RESIDE/{d}_data/training_data.csv'.format(d=config['data_source']))
    valid_path = os.path.join(wd, 'data/RESIDE/{d}_data/validation_data.csv'.format(d=config['data_source']))
    model, valid_x, valid_y = run(sent_emb, ent_emb, rel_emb,
                                  config['kg_combination'],
                                  config['normalization'],
                                  train_path, valid_path,
                                  config['batch_size'],
                                  config['learning_rate'],
                                  config['epochs'],
                                  config['patience'],
                                  config['beta'],
                                  config['ns'],
                                  True)
    metrics, h5, h10 = evaluate(model, valid_x, valid_y, config['compute_shape'], valid_path,
                                True, True, config['language_model'])
    try:
        text_results('Experiments for config {l} done: H5 {r} H10 {s}, '
                     'AS {c}'.format(l=config_name, r=h5, s=h10, c=avg_sim))
    except NameError:
        print('Experiments for config {l} done: H5 {r} H10 {s}, '
              'AS {c}'.format(l=config_name, r=h5, s=h10, c=avg_sim))
