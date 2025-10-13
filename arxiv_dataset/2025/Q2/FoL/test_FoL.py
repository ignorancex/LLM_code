# -*- coding: UTF-8 -*-
import faiss
import torch
from torch import nn
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from prettytable import PrettyTable
import warnings
import cv2
from os.path import join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt

def match_batch_tensor(fm1, fm2, trainflag, grid_size, T2=0.7):
    '''
    fm1: (l,D) 529,768
    fm2: (N,l,D) 100,529,768
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T)

    max1 = torch.argmax(M, dim=1)
    max2 = torch.argmax(M, dim=2)
    m = max2[torch.arange(M.shape[0]).reshape((-1, 1)), max1]
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0], 1)).cuda() == m
    scores = torch.zeros(fm2.shape[0]).cuda()

    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i, :]).squeeze()
        idx2 = max1[i, :][idx1]
        assert idx1.shape == idx2.shape

        if len(idx1.shape) > 0:
            # Calculate cosine similarity and apply threshold
            cos_similarity = torch.sum(fm1[idx1] * fm2[i][idx2], dim=1)
            valid_pairs = cos_similarity > T2
            idx1 = idx1[valid_pairs]
            idx2 = idx2[valid_pairs]

        if trainflag:
            if len(idx1.shape) > 0:
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2], dim=1), dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i], dim=1), dim=0)
            return similarity

        else:
            if len(idx1.shape) < 1:
                scores[i] = 0
            else:
                scores[i] = len(idx1)
    return scores

def local_sim(features_1, features_2, trainflag=False):
    B, Num, C = features_2.shape
    if trainflag:
        queries = features_1
        preds = features_2
        similarity = torch.zeros(B).cuda()
        for i in range(B):
            query,pred = queries[i],preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(61,61))
        return similarity
    else:
        query = features_1
        preds = features_2
        scores = match_batch_tensor(query, preds,trainflag, grid_size=(61,61))
        return scores


def rerank(predictions, queries_local_features, database_local_features):
    pred2 = []
    print("reranking...")
    for query_index, pred in enumerate(tqdm(predictions)):
        query_local_features = torch.tensor(queries_local_features[query_index]).cuda()
        positives_local_features = torch.tensor(database_local_features[pred]).cuda()
        rerank_index = local_sim(query_local_features, positives_local_features, trainflag=False)
        rerank_index_sorted = rerank_index.cpu().numpy().argsort()[::-1]
        pred2.append(predictions[query_index][rerank_index_sorted])
    return np.array(pred2)


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"

    model = model.eval().to(args.device)

    with torch.no_grad():
        logging.debug("Extracting database and queries features for evaluation/testing")
        eval_ds.test_method = "hard_resize"
        dataloader = DataLoader(dataset=eval_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        all_features = torch.empty((len(eval_ds), args.features_dim), dtype=torch.float32, device='cpu')

        for batch_idx, (inputs, indices) in enumerate(tqdm(dataloader, ncols=100)):
            inputs = inputs.to(args.device)
            outputs = model(inputs, test=True)

            if len(outputs) == 2:
                features, local_f = outputs
            elif len(outputs) != 2:
                features, local_f = outputs[0], outputs[1]
            else:
                raise ValueError("Unexpected number of outputs from model")
            if batch_idx == 0:
                local_f_dim1, local_f_dim2 = local_f.shape[-2], local_f.shape[-1]
                local_f_all = torch.empty((len(eval_ds), local_f_dim1, local_f_dim2), dtype=torch.float32, device='cpu')
            if pca is not None:
                features = torch.from_numpy(pca.transform(features.cpu().numpy())).to(args.device)

            start_idx = batch_idx * args.infer_batch_size
            end_idx = start_idx + len(indices)
            all_features[start_idx:end_idx, :] = features.cpu()
            local_f_all[start_idx:end_idx, :, :] = local_f.cpu()


    queries_features = all_features[eval_ds.database_num:].cpu().numpy()
    database_features = all_features[:eval_ds.database_num].cpu().numpy()
    q_local_list = local_f_all[eval_ds.database_num:].to(torch.float32)
    r_local_list = local_f_all[:eval_ds.database_num].to(torch.float32)

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))


    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    print()  # print a new line
    table = PrettyTable()
    table.field_names = ['K'] + [str(k) for k in args.recall_values]
    table.add_row(['Recall@K'] + [f'{v:.2f}' for v in recalls])
    print(table.get_string(title=f"Performances on {eval_ds}"))

    # rerank
    predictions2 = rerank(predictions, q_local_list, r_local_list)

    recalls_rerank = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions2):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls_rerank[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls_rerank = recalls_rerank / eval_ds.queries_num * 100
    recalls_str_rerank = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls_rerank)])

    print()  # print a new line
    table = PrettyTable()
    table.field_names = ['K'] + [str(k) for k in args.recall_values]
    table.add_row(['Recall@K'] + [f'{v:.2f}' for v in recalls_rerank])
    print(table.get_string(title=f"Reranking Performances on {eval_ds}"))

    return recalls, recalls_str, recalls_rerank, recalls_str_rerank