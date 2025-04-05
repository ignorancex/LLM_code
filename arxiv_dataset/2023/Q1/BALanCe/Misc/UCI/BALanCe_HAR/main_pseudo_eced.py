import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
#from Model import ALNet
import model

import Pool
from DataLoader import MyDataset
import torch.nn.functional as F
from torch import optim
#from consistent_dropout import patch_module
from Buffer import Buffer
import numpy as np
import copy
import eced

sample_num=20
al_num=6
hamming_dis_threshold=0.06
patience=3
dropout_rate=0.5
B = 3

batch_size=32
num_workers=0

hamming_pool_size = 10000
val_dataset_size = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight):
    net = model.BayesianNet(10).to(device)
    soft = torch.nn.Softmax(dim=2).to(device)
    opt = optim.Adam(list(net.parameters()), lr=0.001)
    best_acc = 0.0
    best_epoch = 0
    best_model = copy.deepcopy(net)
    weight = torch.Tensor(weight).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight).to(device)
     
    for epoch in range(500):
        net.train()
        for b_id, (x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            opt.zero_grad()
            pred = net(x, sample_num)
            pred = soft(pred)
            pred = torch.mean(pred, dim=1)
            label = label.view(-1)
            #loss = F.cross_entropy(pred, label, weight=weight)
            loss = ce_loss(pred, label)
            loss.backward()
            opt.step()
        
        net.eval()
        ground_truth_lt = []
        pred_lt = []
        for b_id, (x, label) in enumerate(dataloader_val):
            x, label = x.to(device), label.to(device)
            pred = net(x, sample_num)
            pred = soft(pred)
            pred = torch.mean(pred, dim=1)
            label = label.view(-1)
            pred_lt += list(pred.data.cpu().numpy())
            ground_truth_lt += list(label.data.cpu().numpy())

        pred_lt = np.array(pred_lt)
        ground_truth_lt = np.array(ground_truth_lt)

        pred_index = np.argmax(pred_lt, axis=1)
        acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
        if acc <= best_acc:
            if epoch - best_epoch > patience:
                print(f'patience ex;, val_acc: {best_acc}')
                break
        else:
            best_acc = acc
            best_epoch = epoch
            best_model = copy.deepcopy(net)

    net = best_model
    net.eval()
    ground_truth_lt = []
    pred_lt = []
    for b_id, (x, label) in enumerate(dataloader_test):
        x, label = x.to(device), label.to(device)
        pred = net(x, sample_num)
        pred = soft(pred)
        pred = torch.mean(pred, 1)
        label = label.view(-1)
        pred_lt += list(pred.data.cpu().numpy())
        ground_truth_lt += list(label.data.cpu().numpy())

    pred_lt = np.array(pred_lt)
    ground_truth_lt = np.array(ground_truth_lt)

    pred_index = np.argmax(pred_lt, axis=1)
    test_acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
    
    return net, best_acc, test_acc

def one_pass(pass_time):
    buff_val = Buffer()
    #buff_test = Buffer()
    buff_train = Buffer()
    buff_hamming = Buffer()

    pool = Pool.Pool('./data/MNIST/processed/training.pt')
    pool_test = Pool.Pool('./data/MNIST/processed/test.pt')

    feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
    buff_val.add_rows(feat_lt, label_lt)
    
    feat_lt, label_lt = pool.random_query_balanced_labels(K=2)
    buff_train.add_rows(feat_lt, label_lt)

    feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
    buff_hamming.add_rows(feat_lt, label_lt)

    dataset_val = MyDataset(buff_val)
    dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            )

    dataset_test = MyDataset(pool_test)
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            )

    test_acc_lt = []
    while len(buff_train.labels) < 260:
        dataset_train = MyDataset(buff_train)
        weight = dataset_train.get_weight()
        dataloader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                )
        net, val_acc, test_acc = train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight=weight)
        _, pos_lt = eced.delta_pseudo_eced(net, pool, buff_hamming, B=B, device=device, sample_num=sample_num, hamming_dis_threshold=hamming_dis_threshold)
        feat_lt, label_lt = pool.query_labels(pos_lt)
        buff_train.add_rows(feat_lt, label_lt)
        print(f'ECED Pseudo Pass:{pass_time}, train_size:{len(buff_train.labels)}, test_acc:{test_acc}, pos_lt: {pos_lt} ')
        test_acc_lt.append(test_acc)

    return test_acc_lt

acc_lts = []
for pass_time in range(al_num):
    test_acc_lt = one_pass(pass_time)
    acc_lts.append(test_acc_lt)

acc_lts = np.array(acc_lts)
np.savetxt(f'ECED-Pseudo_sample-num-{sample_num}_hamming-{hamming_dis_threshold}_B-{B}_patience-{patience}.out', acc_lts, delimiter=',')

