from __future__ import division
from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_bail, load_credit, load_german
from GNNs import GCN, SAGE, JK
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from core_Ber import Smooth_Ber
import datetime
from utils import load_data, load_data_new, accuracy, normalize, sparse_mx_to_torch_sparse_tensor
import random
import scipy as sp
from scipy import sparse

from torch.distributions.bernoulli import Bernoulli
import torch.multiprocessing as mp





# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument("--batch", type=int, default=10000, help="batch size")
parser.add_argument('--prob', default=0.8, type=float,
                    help="probability to keep the status for each binary entry")
parser.add_argument('--beta', default=0.0, type=float,
                    help="propagation factor")
parser.add_argument("--predictfile", type=str, default="predictfile", help="output prediction file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=10)
parser.add_argument("--N", type=int, default=200, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.3, help="failure probability")
parser.add_argument("--certifyfile", type=str, default="2certifyfile", help="output certified file")
parser.add_argument('--dataset', type=str, default="credit", help='credit german bail google')
parser.add_argument("--gnn", type=str, default='jk', help="a GNN in jk, gcn, sage, and gin")  
parser.add_argument("--threshold", type=float, default=1e5, help="threshold to certify fairness")
parser.add_argument("--threshold_flag", type=str, default='parity', help="parity or equality")
parser.add_argument("--test_ratio", type=float, default=0.9, help="from 0 to 1")
parser.add_argument("--sample_times", type=int, default=100, help="sample how many sets of nodes out of test set for certificatio")
parser.add_argument("--num_x", type=int, default=150, help="number of samples to use for inner x loop")
parser.add_argument("--gaussian_std", type=float, default=1e3, help="Gaussian std to use for inner x loop")
parser.add_argument("--training_noise_adj_std", type=float, default=0.0002, help="Change prob in adj matrix during training process")
parser.add_argument("--training_noise_x_std", type=float, default=2e-5, help="Change prob in adj matrix during training process")
parser.add_argument("--vul_ratio", type=float, default=0.01, help="vulnerable node ratio in the selected set")
parser.add_argument("--grid_num", type=int, default=0, help="number of grid times")



def addGaussianNoise(adj, features, std_adj, std_features):
    density = 0.1
    total_num = int(adj.shape[0] * adj.shape[1] * density)
    row = np.random.randint(adj.shape[0], size=total_num)
    cols = np.random.randint(adj.shape[1], size=total_num)
    data = std_adj * np.random.randn(total_num)
    mid = sparse.csr_matrix((data, (row, cols)), shape=(adj.shape[0], adj.shape[1]))
    adj = adj + mid

    features = features + std_features * torch.randn(features.shape[0], features.shape[1])

    return adj, features

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1


def noise_adding(adj):
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device='cuda')


    if args.cuda:
        adj = adj.to_dense().int().clone().detach().cuda()
    else:
        adj = adj.to_dense().int().clone().detach()


    if args.cuda:
        m = Bernoulli(torch.tensor([args.training_noise_adj_std]).cuda())
    else:
        m = Bernoulli(torch.tensor([args.training_noise_adj_std]))

    random_noise = m.sample(adj.shape).squeeze(-1).int()
    adj_noise = torch.logical_xor(adj, random_noise).int()
    adj_noise = adj_noise.triu() + adj_noise.triu(1).transpose(0, 1)
    coo = adj_noise.to_sparse().coalesce()

    adj_noise_coo = sp.sparse.coo_matrix((coo.values().cpu().numpy(),
                            (coo.indices().cpu().numpy()[0], coo.indices().cpu().numpy()[1])),
                           shape=adj_noise.shape)

    del adj_noise
    del adj

    return adj_noise_coo


def filtering(adj, features, labels, idx_train, idx_val, idx_test, sens):
    indices = None
    label_0 = (1 - labels).sum()
    label_1 = labels.sum()

    if label_0 > label_1:
        indices = np.random.choice(torch.nonzero(1 - labels).squeeze().cpu().numpy().tolist(),
                                   size=int(label_0 - label_1), replace=False)
    else:
        indices = np.random.choice(torch.nonzero(labels).squeeze().cpu().numpy().tolist(),
                                   size=int(label_1 - label_0), replace=False)


    keep_indices = torch.ones_like(labels)
    keep_indices[indices] = 0

    print("Start filtering ...")

    filtered_adj = sp.sparse.csr_matrix(adj.A[keep_indices.bool(), :][:, keep_indices.bool()])
    filtered_feature = features[keep_indices.bool(), :]
    filtered_labels = labels[keep_indices.bool()]
    filtered_sens = sens[keep_indices.bool()]

    indices_new = np.array(list(range(filtered_adj.shape[0])))
    random.shuffle(indices_new)

    filtered_idx_train = indices_new[: int(0.6 * indices_new.shape[0])]
    filtered_idx_val = indices_new[int(0.6 * indices_new.shape[0]): int(0.9 * indices_new.shape[0])]
    filtered_idx_test = indices_new[int(0.9 * indices_new.shape[0]):]

    filtered_idx_train = torch.LongTensor(filtered_idx_train)
    filtered_idx_val = torch.LongTensor(filtered_idx_val)
    filtered_idx_test = torch.LongTensor(filtered_idx_test)

    return filtered_adj, filtered_feature, filtered_labels, filtered_idx_train, filtered_idx_val, filtered_idx_test, filtered_sens



def certify():

    print("CHECK   num of VUL  : "  + str(int(torch.randperm(int(idx_test.shape[0]))[:int(idx_test.shape[0] * args.test_ratio)].shape[0] * args.vul_ratio)))


    num_class, dim = labels.max().item() + 1, features.shape[1]
    smoothed_classifier = Smooth_Ber(model, num_class, dim, args.prob, sparse_mx_to_torch_sparse_tensor(adj).cuda(), features, args.cuda, labels)

    f = open(args.certifyfile, 'w')
    print("idx\tlabel\tpredict\tpABar\tcorrect\tRx\tbest_fair_value\tcor_acc_value\ttime", file=f, flush=True)

    cnt = 0
    cnt_certify = 0

    for i in tqdm(range(args.sample_times)):

        random_idx = torch.randperm(int(idx_test.shape[0]))[:int(idx_test.shape[0] * args.test_ratio)]

        before_time = time.time()
        prediction, pABar, Rx, best_fair_value, cor_acc_value = smoothed_classifier.certify_Ber(idx_test[random_idx], args.N0, args.N, args.alpha, args.num_x, args.gaussian_std, labels, sens, args.threshold, args.threshold_flag, args.vul_ratio)
        after_time = time.time()
        correct = int(int(prediction) == int(1))

        cnt += 1
        cnt_certify += correct

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(i, int(1), prediction, pABar, correct, Rx, best_fair_value, cor_acc_value, time_elapsed), file=f,
              flush=True)

    f.close()

    print("certify acc:", float(cnt_certify) / cnt)



def accuracy_new(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def train(epoch):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features_noise + args.training_noise_x_std * torch.randn(features_noise.shape[0], features_noise.shape[1]).cuda(), edge_index)
    loss_train = F.nll_loss(F.log_softmax(output, dim=1)[idx_train], labels[idx_train])
    acc_train = accuracy_new(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features_noise, edge_index)

    loss_val = F.nll_loss(F.log_softmax(output[idx_val], dim=1), labels[idx_val])
    acc_val = accuracy_new(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()

def tst():
    model.eval()
    output = model(features, edge_index)
    loss_test = F.nll_loss(F.log_softmax(output[idx_test], dim=1), labels[idx_test])
    acc_test = accuracy_new(output[idx_test], labels[idx_test])

    print(output[idx_test])
    print(labels[idx_test])

    pred_labels = output.argmax(1)

    parity, equality = fair_metric(pred_labels[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                   sens[idx_test].numpy())


    print("flagflag")

    print(pred_labels.sum())
    print(pred_labels.shape[0])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    print("Statistical Parity:  " + str(parity))
    print("Equality:  " + str(equality))


    if args.threshold_flag == 'parity':
        args.threshold = parity * 1.25
    elif args.threshold_flag == 'equality':
        args.threshold = equality * 1.25



if __name__ == '__main__':

    params = None 
    with open('params.json', 'r', encoding='utf-8') as jsonf:
        params = json.load(jsonf)

    mp.set_start_method('spawn')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_name = args.dataset
    dataset_gnn = "(" + args.dataset + ", " + args.gnn + ")"
    args.seed = random.choice(params[args.threshold_flag][dataset_gnn][2])

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] =str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic =True

    adj, features, labels, idx_train, idx_val, idx_test, sens = None, None, None, None, None, None, None
    
    if args.dataset == 'credit':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit()
        features = feature_norm(features)
    elif args.dataset == 'german':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_german()
        features = feature_norm(features)
        args.training_noise_adj_std = 0.0002
        args.training_noise_x_std = 1e-5
        args.vul_ratio = 0.05
        args.lr = 0.001
    elif args.dataset == 'bail':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail()
    else:
        print('Not implemented.')
        assert 1 == 0


    args.prob = params[args.threshold_flag][dataset_gnn][0][0]
    args.gaussian_std = params[args.threshold_flag][dataset_gnn][0][1]

    adj_vnl, features, labels, idx_train, idx_val, idx_test, sens = adj, features, labels, idx_train, idx_val, idx_test, sens

    print(adj_vnl.shape[0])
    print("CHECK   num of VUL  : "  + str(int(torch.randperm(int(idx_test.shape[0]))[:int(idx_test.shape[0] * args.test_ratio)].shape[0] * args.vul_ratio)))

    print(labels.sum())
    print(labels.shape[0] - labels.sum())

    print(labels[idx_train].sum())
    print(labels[idx_train].shape[0] - labels[idx_train].sum())

    print(labels[idx_val].sum())
    print(labels[idx_val].shape[0] - labels[idx_val].sum())

    print(labels[idx_test].sum())
    print(labels[idx_test].shape[0] - labels[idx_test].sum())


    model = None
    if args.gnn == 'gcn':
        model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    elif args.gnn == 'sage':
        model = SAGE(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    elif args.gnn == 'jk':
        model = JK(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
        if args.dataset == 'credit':
            args.training_noise_adj_std = 0.00
            args.training_noise_x_std = 0.00
            args.lr = 0.1
    else:
        print("Not implemented.")
        assert 1 == 0
    
    args.certifyfile = "exps_results/" + args.threshold_flag + '_' + str(args.gnn) + '_' + str(args.dataset) + "_" + str(int(args.prob * 10)) + "_" + str(int(args.gaussian_std * 1e3)) + ".txt"
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    final_epochs = 0
    loss_val_global = 1e10
    features_noise = None
    starting = time.time()
    for epoch in range(args.epochs):

        adj = noise_adding(adj_vnl)
        edge_index, _ = convert.from_scipy_sparse_matrix(adj)
        if args.cuda:
            edge_index = edge_index.cuda()

        features_noise = features

        loss_mid = train(epoch)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'trained_gnns/' + args.gnn + '_' + dataset_name + '.pth')
            print("Saved!")
            final_epochs = epoch


    ending = time.time()
    print("Time:", ending - starting, "s")
    model = torch.load('trained_gnns/' + args.gnn + '_' + dataset_name + '.pth')
    tst()

    adj = adj_vnl
    certify()