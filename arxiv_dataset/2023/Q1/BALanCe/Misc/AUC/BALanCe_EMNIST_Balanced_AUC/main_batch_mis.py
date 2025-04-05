import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import model

import Pool
from DataLoader import MyDataset
import torch.nn.functional as F
from torch import optim
from Buffer import Buffer
import numpy as np
import copy
import mis

sample_num=10
al_num=6
#hamming_dis_threshold=0.06
patience=3
dropout_rate=0.5
B = 3

batch_size=64
num_workers=5

#hamming_pool_size = 20000
#val_dataset_size = 20000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight):
    net = model.BayesianNet(47).to(device)
    soft = torch.nn.Softmax(dim=2).to(device)
    opt = optim.Adam(list(net.parameters()), lr=0.001)
    best_acc = 0.0
    best_epoch = 0
    best_model_state = net.state_dict()
    weight = torch.Tensor(weight).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(weight).to(device)
    
    if len(dataloader_train.dataset) > 0:
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
            #print(f'... val acc: {acc}')
            if acc <= best_acc:
                if epoch - best_epoch > patience:
                    print(f'patience ex; best epoch: {best_epoch}, best val acc: {best_acc}')
                    break
            else:
                best_acc = acc
                best_epoch = epoch
                best_model_state = net.state_dict()

        net.load_state_dict(best_model_state)

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

    pool = Pool.Pool('./data/EMNIST/fixed/training.pt')
    pool_test = Pool.Pool('./data/EMNIST/fixed/test_balanced.pt')

    #feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
    #buff_val.add_rows(feat_lt, label_lt)
    buff_val.add_from_pt('./data/EMNIST/fixed/val.pt')

    #feat_lt, label_lt = pool.random_query_balanced_labels(K=2)
    #buff_train.add_rows(feat_lt, label_lt)

    #feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
    #buff_hamming.add_rows(feat_lt, label_lt)
    buff_hamming.add_from_pt('./data/EMNIST/fixed/hamming.pt')

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
        sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=4096)
        dataloader_train = torch.utils.data.DataLoader(
                dataset_train,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                #shuffle=True,
                )
        net, val_acc, test_acc = train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight=weight)
        _, pos_lt = mis.delta_mis_batch_enu(net, pool, device=device, B=B, sample_num=sample_num)
        feat_lt, label_lt = pool.query_labels(pos_lt)
        buff_train.add_rows(feat_lt, label_lt)
        print(f'ECED Batch Pass:{pass_time}, train_size:{len(buff_train.labels)}, test_acc:{test_acc}, pos_lt: {pos_lt} ')
        test_acc_lt.append(test_acc)

    return test_acc_lt

acc_lts = []
for pass_time in range(al_num):
    test_acc_lt = one_pass(pass_time)
    acc_lts.append(test_acc_lt)

acc_lts = np.array(acc_lts)
np.savetxt(f'MIS-Batch_sample-num-{sample_num}_B-{B}_patience-{patience}.out', acc_lts, delimiter=',')

