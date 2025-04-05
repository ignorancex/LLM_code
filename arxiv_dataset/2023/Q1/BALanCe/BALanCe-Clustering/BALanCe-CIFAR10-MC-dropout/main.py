import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import vgg_model
import Pool
from DataLoader import MyDataset
import torch.nn.functional as F
from torch import optim
from np_encoder import NpEncoder
import numpy as np
import random_acq
import bald_gpu, balance_gpu
import badge, core_set
import power_balance, power_bald
import sampler_model 
import copy
import json
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import log_loss
from optparse import OptionParser
import random
from datetime import datetime
import os

usage = "usage: python main.py "
parser = OptionParser(usage)

parser.add_option("-m", "--method", dest="method", type="string", default="Random",
        help="BADGE, CoreSet, Cluster-Margin, BatchBALD, BALD, Batch-BALanCe, BALanCe, Mean-STD, Variation-Ratio, Random")
parser.add_option("-l", "--learning_rate", dest="learning_rate", type="float", default=0.001,
        help="set learning rate for optimizor")
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=64,
        help="batch size")
parser.add_option("--sample_num", dest="sample_num", type="int", default=100,
        help="MC dropout sample number")
parser.add_option("--al_num", dest="al_num", type="int", default=6,
        help="AL trial number")
parser.add_option("--patience", dest="patience", type="int", default=20,
        help="model training patience")
parser.add_option("--dropout_rate", dest="dropout_rate", type="float", default=0.5,
        help="model dropout rate")
parser.add_option("--epoch_num", dest="epoch_num", type="int", default=5000,
        help="model training epoch limit")
parser.add_option("--budget", dest="budget", type="int", default=21000,
        help="AL budget")
parser.add_option("--B", dest="B", type="int", default=1000,
        help="AL acquisition batch size")
parser.add_option("--M", dest="M", type="int", default=10000,
        help="Y configuration sample number")
parser.add_option("--anneal_ratio", dest="anneal_ratio", type="float", default=0.125,
        help="anneal ratio for BALanCE, Batch-BALanCe, and Filtering-Batch-BALanCe")
parser.add_option("--num_workers", dest="num_workers", type="int", default=5,
        help="number of workers for data loader")
parser.add_option("--sampling_index", dest="sampling_index", type="int", default=3,
        help="starting index for sampling Y configurations")
parser.add_option("--downsample_num", dest="downsample_num", type="int", default=500,
        help="downsample_num")
parser.add_option("--coldness", dest="coldness", type="float", default=1.0,
        help="coldness for power distribution sampling")
parser.add_option("--seed", dest="seed", type="int", default=1,
        help="seed")
(options, args) = parser.parse_args()

method=options.method
sample_num=options.sample_num
al_num=options.al_num
patience=options.patience
dropout_rate=options.dropout_rate
epoch_num=options.epoch_num
budget=options.budget
B=options.B
M=options.M
anneal_ratio=options.anneal_ratio
batch_size=options.batch_size
num_workers=options.num_workers
sampling_index=options.sampling_index
seed=options.seed
learning_rate=options.learning_rate
downsample_num=options.downsample_num
coldness=options.coldness
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d~%H:%M:%S")

hype_parameters = {'method': method,
        'sample_num': sample_num, 'al_num':al_num, 'patience':patience,
        'dropout_rate':dropout_rate, 'B':B, 'M':M,
        'batch_size':batch_size, 'num_workers':num_workers,
        'sampling_index': sampling_index,  
        'budget': budget, 'epoch_num': epoch_num, 'anneal_ratio':anneal_ratio, 
        'downsample_num':downsample_num,  
        'coldness':coldness,
        'learning_rate':learning_rate, 'seed':seed}

print(hype_parameters)

checkpoint = {}
for key in hype_parameters:
    checkpoint[key] = hype_parameters[key]
checkpoint['trial_checkpoint_lt'] = []

#random.seed(seed)
#torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight):
    def train_onetime(dataloader_train, dataloader_val, dataloader_test, weight):
        #net = vgg_model.vgg11(pretrained_features_only=True, num_classes=10).to(device)
        net = vgg_model.vgg11_bn(num_classes=10).to(device)
        #opt = optim.Adam(list(net.parameters()), lr=learning_rate) 
        opt = optim.SGD(list(net.parameters()), lr=learning_rate, momentum=0.9) 
        train_net = sampler_model.SamplerModel(net, k=1)
        val_net = sampler_model.NoDropoutModel(net)
        test_net = sampler_model.SamplerModel(net, k=sample_num)

        best_acc = 0.0
        best_epoch = 0
        best_model_state = copy.deepcopy(net.state_dict())
        weight = torch.Tensor(weight).to(device)
        
        #print(f'train loader size: {dataloader_train.dataset.size()}')
        if dataloader_train.dataset.size() > 0:
            for epoch in range(epoch_num):
                net.train()
                for b_id, (pos, x, label) in enumerate(dataloader_train):
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
                    for b_id, (pos, x, label) in enumerate(dataloader_val):
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
                        #print(f'patience ex; best epoch: {best_epoch}, best val acc: {best_acc}')
                        break
                else:
                    best_acc = acc
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(net.state_dict())

                print(f'\t epoch: {epoch}, acc: {acc}')
        
        return best_acc, best_epoch, best_model_state, net, test_net

    best_acc, best_epoch, best_model_state, net, test_net = train_onetime(dataloader_train, dataloader_val, dataloader_test, weight)
    while best_acc < 0.5:
        best_acc, best_epoch, best_model_state, net, test_net = train_onetime(dataloader_train, dataloader_val, dataloader_test, weight)

    net.load_state_dict(best_model_state)
    net.eval()
    ground_truth_lt = []
    pred_lt = []
    with torch.no_grad():
        for b_id, (pos, x, label) in enumerate(dataloader_test):
            x, label = x.to(device), label.to(device)
            pred = test_net(x).exp_()
            label = label.view(-1)
            pred_lt += list(pred.data.cpu().numpy())
            ground_truth_lt += list(label.data.cpu().numpy())

    pred_lt = np.array(pred_lt)
    ground_truth_lt = np.array(ground_truth_lt)

    pred_index = np.argmax(pred_lt, axis=1)
    test_acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
    test_auc = roc_auc_score(ground_truth_lt, pred_lt, multi_class='ovr')
    test_f1 = f1_score(ground_truth_lt, pred_index, average='macro') 
    test_NLL = log_loss(ground_truth_lt, pred_lt) 

    test_result = {'test_acc': test_acc, 'test_auc': test_auc, 'test_f1': test_f1, 'test_NLL': test_NLL}

    return net, best_acc, test_result

def one_pass(pass_time):
    model_state_lt = []
    trial_checkpoint = {'trial_pass': pass_time,
            'test_result_lt': [],
            'acquired_pos_lt': []}
    
    pool = Pool.Pool('../data/', label_file='../data/train_valid_lt.txt', bitmap='../data/pool_index_lt.json')
    pool_train = Pool.Pool('../data/', label_file='../data/train_valid_lt.txt', bitmap='../data/start_index_lt.json')
    pool_val = Pool.Pool('../data/', label_file='../data/train_valid_lt.txt', bitmap='../data/val_index_lt.json')
    pool_hamming = Pool.Pool('../data/', label_file='../data/train_valid_lt.txt', bitmap='../data/hamming_index_lt.json')
    pool_test = Pool.Pool('../data/', label_file='../data/test_lt.txt')

    trial_checkpoint['starting_size'] = pool_train.size()

    dataset_val = MyDataset(pool_val)
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

    test_result_lt = []
    while pool_train.size() <= budget:
        dataset_train = MyDataset(pool_train, train=True)
        weight = dataset_train.get_weight()
        sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=4*8092)
        dataloader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    )
        
        net, val_acc, test_result = train_eval_model(dataloader_train, dataloader_val, dataloader_test, weight=weight)

        trial_checkpoint['test_result_lt'].append(test_result)
        model_state_lt.append(copy.deepcopy(net.state_dict()))

        if method == 'Random':
            pos_lt = random_acq.acquire(pool, B=B) 
        elif method == 'BALD':
            pos_lt = bald_gpu.acquire(net, pool, device=device, M=M, B=B, sample_num=sample_num)
        elif method == 'BALanCe':
            hamming_dis_threshold=(1.0-val_acc)*anneal_ratio
            pos_lt = balance_gpu.acquire(net, pool, pool_hamming, device=device, downsample_num=downsample_num, coldness=coldness, B=B, hamming_dis_threshold=hamming_dis_threshold, sample_num=sample_num)
        elif method == 'PowerBALD':
            pos_lt = power_bald.acquire(net, pool, device=device, coldness=coldness, B=B, sample_num=sample_num)
        elif method == 'PowerBALanCe':
            hamming_dis_threshold=(1.0-val_acc)*anneal_ratio
            pos_lt = power_balance.acquire(net, pool, pool_hamming, device=device, coldness=coldness, B=B, hamming_dis_threshold=hamming_dis_threshold, sample_num=sample_num)
        elif method == 'BADGE':
            pos_lt = badge.acquire(net, pool, device=device, B=B, sample_num=sample_num)
        elif method == 'CoreSet':
            pos_lt = core_set.acquire(net, pool, pool_train, device=device, B=B, sample_num=sample_num)
        else:
            print(f'{method} does not exist.')
            sys.exit()

        trial_checkpoint['acquired_pos_lt'].append(pos_lt)
        pool.query_labels(pos_lt)
        pool_train.add_samples(pos_lt)
        print(f'{method} Pass:{pass_time}, train_size:{dataset_train.size()-B},\n\ttest_result:{test_result}')

    return trial_checkpoint, model_state_lt

trial_model_state_lt = []
for pass_time in range(al_num):
    trial_checkpoint, model_state_lt = one_pass(pass_time)
    checkpoint['trial_checkpoint_lt'].append(trial_checkpoint)

    with open(f'{method}_checkpoint_{dt_string}.json', 'w') as outfile:
        json.dump(checkpoint, outfile, cls=NpEncoder)
