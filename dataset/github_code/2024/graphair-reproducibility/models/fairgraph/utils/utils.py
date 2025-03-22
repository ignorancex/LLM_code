import numpy as np
import scipy.sparse as sp
import torch
import scipy.stats
from sklearn.metrics import roc_auc_score

def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def auc(output, labels):
    output = output.squeeze()
    
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    auc = roc_auc_score(labels, output)
    return auc


def fair_metric(output,idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1

    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity * 100 ,equality * 100
    
def weighted_homophily(adj_matrix, sens):
    node_homophily = np.zeros(adj_matrix.shape[0])

    sens = sens.cpu()

    for i in range(adj_matrix.shape[0]):
        neighbors = adj_matrix[i, :]
        same_label_strength = neighbors[sens == sens[i]].sum()
        total_strength = neighbors.sum()

        if total_strength > 0:
            node_homophily[i] = same_label_strength / total_strength
        else:
            node_homophily[i] = 0

    return node_homophily

def spearman_correlation(features, sens):
    correlations = []

    sens = sens.cpu().numpy()

    for i in range(features.shape[1]):
        correlation, _ = scipy.stats.spearmanr(sens, features[:, i].cpu().numpy())
        correlations.append(correlation)

    return correlations