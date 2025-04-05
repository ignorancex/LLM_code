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
import sampler_model

sample_num=200
#sample_each_class = 16

sample_each_class_lt = [1,28,28,28,28,
                        28,28,28,28,1]

al_num=6
hamming_dis_threshold=0.05
patience=3
dropout_rate=0.5

batch_size=64
num_workers=5

#hamming_pool_size = 10000
#val_dataset_size = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight):
    net = model.BayesianNet(10).to(device)
    opt = optim.Adam(list(net.parameters()), lr=0.001)
    
    train_net = sampler_model.SamplerModel(net, k=1)
    val_net = sampler_model.NoDropoutModel(net)
    test_net = sampler_model.SamplerModel(net, k=sample_num)

    best_acc = 0.0
    best_epoch = 0
    best_model_state = copy.deepcopy(net.state_dict())
    weight = torch.Tensor(weight).to(device)
     
    for epoch in range(500):
        net.train()
        for b_id, (x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            opt.zero_grad()
            pred = train_net(x)
            label = label.view(-1)
            loss = F.nll_loss(pred, label, weight=weight)
            loss.backward()
            opt.step()
        
        net.eval()
        ground_truth_lt = []
        pred_lt = []
        with torch.no_grad():
            for b_id, (x, label) in enumerate(dataloader_val):
                x, label = x.to(device), label.to(device)
                pred = val_net(x)
                label = label.view(-1)
                pred_lt += list(pred.data.cpu().numpy())
                ground_truth_lt += list(label.data.cpu().numpy())

        pred_lt = np.array(pred_lt)
        ground_truth_lt = np.array(ground_truth_lt)

        pred_index = np.argmax(pred_lt, axis=1)
        acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
        if acc <= best_acc:
            if epoch - best_epoch > patience:
                print(f'patience ex; best epoch: {best_epoch},  best val acc: {best_acc}')
                break
        else:
            best_acc = acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model_state)
    net.eval()
    ground_truth_lt = []
    pred_lt = []
    with torch.no_grad():
        for b_id, (x, label) in enumerate(dataloader_test):
            x, label = x.to(device), label.to(device)
            pred = test_net(x)
            label = label.view(-1)
            pred_lt += list(pred.data.cpu().numpy())
            ground_truth_lt += list(label.data.cpu().numpy())

    pred_lt = np.array(pred_lt)
    ground_truth_lt = np.array(ground_truth_lt)

    pred_index = np.argmax(pred_lt, axis=1)
    test_acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
    
    return net, best_acc, test_acc

buff_val = Buffer()
#buff_test = Buffer()
buff_train = Buffer()
buff_hamming = Buffer()

pool = Pool.Pool('./data/MNIST/fixed/training.pt')
pool_test = Pool.Pool('./data/MNIST/fixed/test.pt')

#feat_lt, label_lt = pool.random_query_labels(val_dataset_size)
#buff_val.add_rows(feat_lt, label_lt)
buff_val.add_from_pt('./data/MNIST/fixed/val.pt')

#feat_lt, label_lt = pool.random_query_balanced_labels(K=sample_each_class)
feat_lt, label_lt = pool.random_query_labels_spec(sample_each_class_lt)
buff_train.add_rows(feat_lt, label_lt)
#buff_train.add_from_pt('./data/MNIST/fixed/start.pt')

#feat_lt, label_lt = pool.random_query_labels(hamming_pool_size)
#buff_hamming.add_rows(feat_lt, label_lt)
buff_hamming.add_from_pt('./data/MNIST/fixed/hamming.pt')

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
    
dataset_train = MyDataset(buff_train)
weight = dataset_train.get_weight()
if len(buff_train.labels) < 260:
    sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=4096)
    dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            #shuffle=True,
            #drop_last=True
        )
else:
    dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            #sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True
        )

net, val_acc, test_acc = train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight=weight)

dataset1 = MyDataset(pool_test)
dataset2 = MyDataset(buff_hamming)

dataloader1 = torch.utils.data.DataLoader(
        dataset1,
        batch_size=32,
        shuffle=False
        )

dataloader2 = torch.utils.data.DataLoader(
        dataset2,
        batch_size=32,
        shuffle=False
        )


test_pool_pred_lt = []
test_ground_lt = []

hamming_pred_lt = []
hamming_ground_lt = []

net.eval()
with torch.no_grad():
    for b_id, (x, label) in enumerate(dataloader1):
        x, label = x.to(device), label.to(device)
        pred = net(x, sample_num)
        label_soft_pred = pred.exp_()
        test_pool_pred_lt += list(label_soft_pred.data.cpu().numpy())
        
        test_ground_lt += list(label.data.cpu().numpy())
    
    for b_id, (x, label) in enumerate(dataloader2):
        x, label = x.to(device), label.to(device)
        pred = net(x, sample_num)
        label_soft_pred = pred.exp_()
        hamming_pred_lt += list(label_soft_pred.data.cpu().numpy())
        
        hamming_ground_lt += list(label.data.cpu().numpy())
        
test_pool_pred_lt = np.array(test_pool_pred_lt)

hamming_pred_lt = np.array(hamming_pred_lt)
hamming_pred_lt = np.argmax(hamming_pred_lt, axis=2)

test_ground_lt = np.array(test_ground_lt)
test_ground_lt = np.squeeze(test_ground_lt)

hamming_ground_lt = np.array(hamming_ground_lt)
hamming_ground_lt = np.squeeze(hamming_ground_lt)

test_pred_index = np.argmax(test_pool_pred_lt, axis=2)

acc_lt = []
for k in range(0,sample_num):
    tmp_test_pred_index = test_pred_index[:,k]
    acc = np.sum(tmp_test_pred_index == test_ground_lt)/len(tmp_test_pred_index)
    
    #print(tmp_test_pred_index == test_ground_lt)
    print(test_ground_lt.shape)
    print(tmp_test_pred_index.shape)
    print(acc)
    acc_lt.append(acc)

acc_lt.append(1.0)
acc_lt = np.array(acc_lt)

hamming_pred_lt = hamming_pred_lt.T

hamming_pred_lt = np.concatenate((hamming_pred_lt, [hamming_ground_lt]), axis=0)

sample_num = np.sum(np.array(sample_each_class_lt))

np.savetxt(f'TeoAlg/acc_lt_{sample_num}.txt', acc_lt, delimiter=',')
np.savetxt(f'TeoAlg/hamming_pool_pred_{sample_num}.txt', hamming_pred_lt, delimiter=',')


