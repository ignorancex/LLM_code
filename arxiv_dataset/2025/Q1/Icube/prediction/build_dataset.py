import json
import random
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import torch
import os
import networkx as nx
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tqdm import *
import re

random.seed(0)


def build_dataset(fpath):
    file_names = os.listdir(fpath)
    random.shuffle(file_names)
    train_data_size = int(0.8 * len(file_names))
    train_data = file_names[:train_data_size]
    test_data = file_names[train_data_size:]
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
    return train_dataloader, test_dataloader


def get_dataset(fpath, bigraph, egraph, tgraph, bsgraph, aoigraph, root_path):
    fpath = root_path + '/' + fpath
    with open(fpath, 'r') as f:
        data = json.load(f)

    node_list = bigraph.node_list
    n_power = egraph.node_num
    n_junc = tgraph.node_num
    n_bs = bsgraph.node_num
    n_aoi = aoigraph.node_num

    add_power = []
    add_junc = []
    add_bs = []
    add_aoi = []
    for node in range(0, n_power):
        if node_list[node] in data['source']:
            add_power.append(1)
        else:
            add_power.append(0)
    for node in range(n_power, n_power + n_junc):
        add_junc.append(0)
    for node in range(n_power + n_junc, n_power + n_junc + n_bs):
        add_bs.append(0)
    for node in range(n_power + n_junc + n_bs, n_power + n_junc + n_bs + n_aoi):
        add_aoi.append(0)

    end_power = []
    end_junc = []
    end_bs = []
    end_aoi = []

    for node in range(0, n_power):
        if node_list[node] in data['ruin_nodes']:
            end_power.append(1)
        else:
            end_power.append(0)

    for node in range(n_power, n_power + n_junc):
        if node_list[node] in data['ruin_nodes']:
            end_junc.append(1)
        else:
            end_junc.append(0)

    for node in range(n_power + n_junc, n_power + n_junc + n_bs):
        if node_list[node] in data['ruin_nodes']:
            end_bs.append(1)
        else:
            end_bs.append(0)

    for node in range(n_power + n_junc + n_bs, n_power + n_junc + n_bs + n_aoi):
        if node_list[node] in data['ruin_nodes']:
            end_aoi.append(1)
        else:
            end_aoi.append(0)

    return add_power, add_junc, add_bs, add_aoi, end_power, end_junc, end_bs, end_aoi


def get_dataset_mask(fpath, bigraph, egraph, tgraph, bsgraph):
    fpath = '../Data/ruin_cascades/4nodes/' + fpath
    with open(fpath, 'r') as f:
        data = json.load(f)

    node_list = bigraph.node_list
    n_power = egraph.node_num
    n_junc = tgraph.node_num
    n_bs = bsgraph.node_num

    neighbors = []
    for node in data['source']:
        ego = nx.ego_graph(bigraph.nxgraph, node, radius=3)
        neighbors.extend(list(ego.nodes))
    neighbors = list(set(neighbors))

    add_power = []
    add_junc = []
    add_bs = []
    for node in range(0, n_power):
        if node_list[node] in data['source']:
            add_power.append(1)
        else:
            add_power.append(0)
    for node in range(n_power, n_power + n_junc):
        add_junc.append(0)
    for node in range(n_power + n_junc, n_power + n_junc + n_bs):
        add_bs.append(0)

    end_power = []
    end_junc = []
    end_bs = []
    power_mask = []
    junc_mask = []
    bs_mask = []

    for node in range(0, n_power):
        if node_list[node] in data['ruin_nodes']:
            end_power.append(1)
        else:
            end_power.append(0)
        if node_list[node] in neighbors:
            power_mask.append(True)
        else:
            power_mask.append(False)

    for node in range(n_power, n_power + n_junc):
        if node_list[node] in data['ruin_nodes']:
            end_junc.append(1)
        else:
            end_junc.append(0)
        if node_list[node] in neighbors:
            junc_mask.append(True)
        else:
            junc_mask.append(False)

    for node in range(n_power + n_junc, n_power + n_junc + n_bs):
        if node_list[node] in data['ruin_nodes']:
            end_bs.append(1)
        else:
            end_bs.append(0)
        if node_list[node] in neighbors:
            bs_mask.append(True)
        else:
            bs_mask.append(False)

    return add_power, add_junc, add_bs, end_power, end_junc, end_bs, power_mask, junc_mask, bs_mask


def calculate_metrics_homo(label, logits, epoch):
    y_pred = [1 if p[1] >= 0.9 else 0 for p in logits]
    auc = roc_auc_score(label, y_pred)
    f1 = f1_score(label, y_pred)
    pre = precision_score(label, y_pred)
    rec = recall_score(label, y_pred)
    pred_num = len(np.where(np.array(y_pred) == 1)[0])
    real_num = len(np.where(np.array(label) == 1)[0])
    RMSE = np.sqrt(np.mean((pred_num - real_num) ** 2))
    return auc, f1, pre, rec, RMSE


def calculate_metrics(end_power, end_junc, end_bs, end_aoi, logits, epoch):
    y_true = list(end_power.detach().numpy()) + \
             list(end_junc.detach().numpy()) + \
             list(end_bs.detach().numpy()) + \
             list(end_aoi.detach().numpy())
    # logits_1 = list(logits['power'].detach().cpu().numpy()[:, 1]) + \
    #            list(logits['junc'].detach().cpu().numpy()[:, 1]) + \
    #            list(logits['bs'].detach().cpu().numpy()[:, 1]) + \
    #            list(logits['aoi'].detach().cpu().numpy()[:, 1])
    # if epoch % 20 == 0 and (not epoch == 0):
    #     plt.hist(logits_1, bins='auto', edgecolor='black')
    #     plt.title('pred_epoch{0}'.format(epoch))
    #     plt.show()
    #     plt.hist(y_true, bins='auto', edgecolor='black')
    #     plt.title('true_epoch{0}'.format(epoch))
    #     plt.show()
    y_pred = [1 if p[1] >= 0.5 else 0 for p in logits['power']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['junc']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['bs']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['aoi']]
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pred_num = len(np.where(np.array(y_pred) == 1)[0])
    real_num = len(np.where(np.array(y_true) == 1)[0])
    RMSE = np.sqrt(np.mean((pred_num - real_num) ** 2))
    return auc, f1, pre, rec, RMSE


def cal_test_number(logits):
    y_pred = [1 if p[1] >= 0.5 else 0 for p in logits['power']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['junc']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['bs']] + \
             [1 if p[1] >= 0.5 else 0 for p in logits['aoi']]
    test_num = np.where(np.array(y_pred) == 1)[0]
    test_num = len(test_num)
    return test_num


def extract_numbers(string):
    pattern = r'\d+'
    numbers = re.findall(pattern, string)
    result = ''.join(numbers)
    return result


def calculate_metrics_mask(end_power, end_junc, end_bs, logits, power_mask, junc_mask, bs_mask):
    mask = power_mask + junc_mask + bs_mask
    y_true = \
        np.array(
            list(end_power.detach().numpy()) + list(end_junc.detach().numpy()) + list(end_bs.detach().numpy()))[
            mask]
    y_pred = np.array([1 if p[1] >= 0.5 else 0 for p in logits['power']] + \
                      [1 if p[1] >= 0.5 else 0 for p in logits['junc']] + \
                      [1 if p[1] >= 0.5 else 0 for p in logits['bs']])[mask]
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
    else:
        auc = 0
        f1 = 0
        pre = 0
        rec = 0
    return auc, f1, pre, rec


def cal_weight(label):
    label = label.detach().cpu().numpy()
    weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(label), y=label)
    weight = torch.tensor(weight, dtype=torch.float32)
    if len(weight) == 1:
        return torch.tensor([1, 1], dtype=torch.float32)
    else:
        # weight = torch.tensor([weight[0], weight[1] * 1.5], dtype=torch.float32)
        return weight
        # return torch.tensor([1, 1], dtype=torch.float32)


def plot_metrics(y_true, y_pred, y_true_list, y_pred_list, metric, model):
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)
    plt.plot(y_true_list, label='True')
    plt.plot(y_pred_list, label='Pred')
    plt.legend()
    fpath = '../Data/plot_results_multinodes/{0}'.format(model)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    plt.savefig('{1}/{0}.png'.format(metric, fpath))
    plt.close()
    return y_true_list, y_pred_list


def save_output(end_power, end_junc, end_bs, end_aoi, logits, model_name, stage):
    fpath = '../Data/plot_results_multinodes/{0}'.format(model_name)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fpath = '{0}/results'.format(fpath)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fpath = '{0}/{1}'.format(fpath, stage)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    y_power = logits['power'].detach().cpu().numpy()
    y_junc = logits['junc'].detach().cpu().numpy()
    y_bs = logits['bs'].detach().cpu().numpy()
    y_aoi = logits['aoi'].detach().cpu().numpy()
    end_power = np.array(end_power)
    end_junc = np.array(end_junc)
    end_bs = np.array(end_bs)
    end_aoi = np.array(end_aoi)
    np.save('{0}/y_power.npy'.format(fpath), y_power)
    np.save('{0}/y_junc.npy'.format(fpath), y_junc)
    np.save('{0}/y_bs.npy'.format(fpath), y_bs)
    np.save('{0}/y_aoi.npy'.format(fpath), y_aoi)
    np.save('{0}/end_power.npy'.format(fpath), end_power)
    np.save('{0}/end_junc.npy'.format(fpath), end_junc)
    np.save('{0}/end_bs.npy'.format(fpath), end_bs)
    np.save('{0}/end_aoi.npy'.format(fpath), end_aoi)


def save_output_homo(label, logits, model_name, stage):
    fpath = '../Data/plot_results/{0}'.format(model_name)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fpath = '{0}/results'.format(fpath)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    fpath = '{0}/{1}'.format(fpath, stage)
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    y_pred = logits.detach().cpu().numpy()
    label = np.array(label)
    np.save('{0}/y_pred.npy'.format(fpath), y_pred)
    np.save('{0}/label.npy'.format(fpath), label)
