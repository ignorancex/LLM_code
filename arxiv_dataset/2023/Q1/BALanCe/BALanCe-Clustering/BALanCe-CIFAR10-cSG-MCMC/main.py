import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
#import vgg_model
import models
import Pool
from DataLoader import MyDataset
import torch.nn.functional as F
from torch import optim
from np_encoder import NpEncoder
import numpy as np
import random_acq, batch_balance_gpu, batch_bald_gpu, mean_std, variation_ratios
import cluster_margin, margin, badge, core_set
import bald_gpu, balance_gpu
#import sampler_model 
import copy
import json
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import log_loss
from optparse import OptionParser
import random, glob
from datetime import datetime
import power_bald, power_balance
import lin_partition
import os, shutil

usage = "usage: python main.py "
parser = OptionParser(usage)

parser.add_option("-m", "--method", dest="method", type="string", default="Random",
        help="BADGE, CoreSet, Cluster-Margin, BatchBALD, BALD, Batch-BALanCe, BALanCe, Mean-STD, Variation-Ratio, Random")
parser.add_option("-l", "--learning_rate", dest="learning_rate", type="float", default=0.0001,
        help="set learning rate for optimizor")
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=64,
        help="batch size")
parser.add_option("--sample_num", dest="sample_num", type="int", default=100,
        help="MC dropout sample number")
parser.add_option("--al_num", dest="al_num", type="int", default=6,
        help="AL trial number")
parser.add_option("--patience", dest="patience", type="int", default=10,
        help="model training patience")
parser.add_option("--dropout_rate", dest="dropout_rate", type="float", default=0.5,
        help="model dropout rate")
parser.add_option("--epoch_num", dest="epoch_num", type="int", default=400,
        help="model training epoch limit")
parser.add_option("--budget", dest="budget", type="int", default=12000,
        help="AL budget")
parser.add_option("--B", dest="B", type="int", default=2000,
        help="AL acquisition batch size")
parser.add_option("--downsample_num", dest="downsample_num", type="int", default=6000,
        help="downsample num")
parser.add_option("--M", dest="M", type="int", default=8,
        help="Y configuration sample number")
parser.add_option("--anneal_ratio", dest="anneal_ratio", type="float", default=0.125,
        help="anneal ratio for BALanCE, Batch-BALanCe, and Filtering-Batch-BALanCe")
parser.add_option("--num_workers", dest="num_workers", type="int", default=5,
        help="number of workers for data loader")
parser.add_option("--sampling_index", dest="sampling_index", type="int", default=3,
        help="starting index for sampling Y configurations")
parser.add_option("--cluster_method", dest="cluster_method", type="string", default="kcenter",
        help="kcenter, kmeans++, HAC")
parser.add_option("--cluster_num", dest="cluster_num", type="int", default=200,
        help="cluster_num")
parser.add_option('--alpha', type=int, default=1, help='1: SGLD')
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
anneal_ratio=options.anneal_ratio
batch_size=options.batch_size
num_workers=options.num_workers
sampling_index=options.sampling_index
seed=options.seed
learning_rate=options.learning_rate
cluster_method=options.cluster_method
cluster_num=options.cluster_num
alpha = options.alpha
downsample_num=options.downsample_num
M=options.M
temperature=1/50000
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d~%H:%M:%S")

hype_parameters = {'method': method,
        'sample_num': sample_num, 'al_num':al_num, 'patience':patience,
        'dropout_rate':dropout_rate, 'B':B,
        'batch_size':batch_size, 'num_workers':num_workers,
        'sampling_index': sampling_index,  
        'budget': budget, 'epoch_num': epoch_num, 'anneal_ratio':anneal_ratio, 
        'learning_rate':learning_rate, 'M': M, 
        'downsample_num': downsample_num,
        'seed':seed}

print(hype_parameters)

checkpoint = {}
for key in hype_parameters:
    checkpoint[key] = hype_parameters[key]
checkpoint['trial_checkpoint_lt'] = []

random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def noise_loss(net, lr, alpha):
    noise_loss_value = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).to(device)
        noise_loss_value += torch.sum(var * torch.normal(means, std = noise_std).to(device))
    return noise_loss_value

def adjust_learning_rate(optimizer, epoch, batch_idx, num_batch, T, lr_0):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_eval_model(dataloader_train, dataloader_val, dataloader_test, train_set_size, epoch_num):
    net = models.ResNet18().to(device)
    lr_0 = 0.5
    opt = optim.SGD(net.parameters(), lr=lr_0, momentum=1-alpha, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    num_batch = train_set_size//batch_size
    print(f'num batch: {num_batch}')
    T= epoch_num * num_batch
    print(f'T: {T}') 
    
    for epoch in range(epoch_num):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f'epoch: {epoch}')
        for b_id, (pos, x, label) in enumerate(dataloader_train):
            x, label = x.to(device), label.to(device)
            label = label.view(-1)
            opt.zero_grad()
            lr = adjust_learning_rate(opt, epoch, b_id, num_batch, T, lr_0)
            outputs, _ = net(x)
            if (epoch%50)+1>45:
                loss_noise_value = noise_loss(net, lr, alpha)*(temperature/train_set_size)**.5
                loss = criterion(outputs, label)+loss_noise_value
            else:
                loss = criterion(outputs, label)
            loss.backward()
            opt.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
            if b_id % 100==0:
                print(f'Epoch: {epoch}, BatchID: {b_id}, Train Loss: {train_loss/(b_id+1):.3f} | Acc: {100.*correct.item()/total:.3f}')
        
        net.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for b_id, (pos, x, label) in enumerate(dataloader_val):
                x, label = x.to(device), label.to(device)
                label = label.view(-1)
                outputs, _ = net(x)
                loss = criterion(outputs, label)
                val_loss += loss.data.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += predicted.eq(label.data).cpu().sum()
                if b_id % 30 == 0:
                    print(f'Val Loss: {val_loss/(b_id+1):.3f} | Acc: {100.*correct.item()/total:.3f}')
        
        acc = correct.item()/total
        if acc > best_acc:
            best_acc = acc

        print(f'Val Set: Average loss: {val_loss/len(dataloader_val):.4f}, Accuracy: {100. * correct.item() / total}')

        if (epoch % 50)+1 > 47:
            print('save model')
            net.cpu()
            torch.save(net.state_dict(), f'saved_models/{method}_saved_models/cifar_model_{epoch}.pt')
            net.to(device)
    
    pred_lt = []
    ground_truth_lt = []

    for path in glob.glob(f'saved_models/{method}_saved_models/cifar_model_*.pt'):
        net.cpu()
        net.load_state_dict(torch.load(path))
        net.to(device)
        net.eval()

        tmp_pred_lt = []
        tmp_ground_truth_lt = []
        with torch.no_grad():
             for b_id, (pos, x, label) in enumerate(dataloader_test):
                x, label = x.to(device), label.to(device)
                label = label.view(-1)
                outputs, _ = net(x)
                outputs = F.softmax(outputs, dim=1) 
                tmp_pred_lt += list(outputs.data.cpu().numpy())
                tmp_ground_truth_lt += list(label.data.cpu().numpy())
        
        tmp_pred_lt = np.array(tmp_pred_lt)
        tmp_ground_truth_lt = np.array(tmp_ground_truth_lt)
        tmp_pred_index = np.argmax(tmp_pred_lt, axis=1)
        tmp_test_acc = np.sum(tmp_pred_index == tmp_ground_truth_lt)/len(tmp_ground_truth_lt)
        print(f'checkpoint: {path}, acc: {tmp_test_acc}')

        pred_lt.append(tmp_pred_lt)
        ground_truth_lt.append(tmp_ground_truth_lt)
        
    # assert 
    if len(ground_truth_lt) > 1:
        assert np.all(ground_truth_lt[0] == ground_truth_lt[1])
    
    pred_lt = np.array(pred_lt)
    pred_lt = np.mean(pred_lt, axis=0)
    #print(f'pred_lt: {pred_lt.shape}')

    ground_truth_lt = np.array(ground_truth_lt[0])
    #print(f'ground_truth_lt: {ground_truth_lt.shape}')

    pred_index = np.argmax(pred_lt, axis=1)
    test_acc = np.sum(pred_index == ground_truth_lt)/len(ground_truth_lt)
    test_auc = roc_auc_score(ground_truth_lt, pred_lt, multi_class='ovr')
    test_f1 = f1_score(ground_truth_lt, pred_index, average='macro')
    test_NLL = log_loss(ground_truth_lt, pred_lt)

    test_result = {'test_acc': test_acc, 'test_auc': test_auc, 'test_f1': test_f1, 'test_NLL': test_NLL}

    return best_acc, test_result

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
            shuffle=False,
            )

    dataset_test = MyDataset(pool_test)
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            )

    test_result_lt = []
    while pool_train.size() <= budget:
        dataset_train = MyDataset(pool_train, train=True)
        weight = dataset_train.get_weight()
        #train_set_size = len(dataset_train) 
        train_set_size = 50000
        sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=train_set_size)
        dataloader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True
                    )
        
        val_acc, test_result = train_eval_model(dataloader_train, dataloader_val, dataloader_test, train_set_size, epoch_num)
        
        trial_checkpoint['test_result_lt'].append(test_result)
        #model_state_lt.append(copy.deepcopy(net.state_dict()))

        if method == 'Random':
            pos_lt = random_acq.acquire(pool, B=B) 
        elif method == 'BADGE':
            pos_lt = badge.acquire(f'saved_models/{method}_saved_models/', pool, device=device, B=B)
        elif method == 'CoreSet':
            pos_lt = core_set.acquire(f'saved_models/{method}_saved_models/', pool, pool_train, device=device, B=B)
        elif method == 'PowerBALD':
            pos_lt =  power_bald.acquire(f'saved_models/{method}_saved_models/', pool, device=device, B=B) 
        elif method == 'PowerBALanCe':
            hamming_dis_threshold = (1.0-val_acc)*anneal_ratio
            pos_lt =  power_balance.acquire(f'saved_models/{method}_saved_models/', pool, pool_hamming, hamming_dis_threshold=hamming_dis_threshold, device=device, B=B)  
        elif method == 'BALanCe':
            hamming_dis_threshold = (1.0-val_acc)*anneal_ratio
            pos_lt =  balance_gpu.acquire(f'saved_models/{method}_saved_models/', pool, pool_hamming, hamming_dis_threshold=hamming_dis_threshold, device=device, B=B, downsample_num=downsample_num)  
        else:
            print(f'{method} does not exist.')
            sys.exit()

        trial_checkpoint['acquired_pos_lt'].append(pos_lt)
        pool.query_labels(pos_lt)
        pool_train.add_samples(pos_lt)
        #print(f'{method} Pass:{pass_time}, train_size:{dataset_train.size()},\n\ttest_result:{test_result},\n\tpos_lt: {pos_lt} ')
        print(f'{method} Pass:{pass_time}, train_size:{dataset_train.size()-B},\n\ttest_result:{test_result}')

    return trial_checkpoint, model_state_lt

if os.path.exists(f'saved_models/{method}_saved_models/'):
    shutil.rmtree(f'saved_models/{method}_saved_models/')
    os.makedirs(f'saved_models/{method}_saved_models/')
else:
    os.makedirs(f'saved_models/{method}_saved_models/')

for pass_time in range(al_num):
    trial_checkpoint, model_state_lt = one_pass(pass_time)
    checkpoint['trial_checkpoint_lt'].append(trial_checkpoint)

    with open(f'{method}_checkpoint_{dt_string}.json', 'w') as outfile:
        json.dump(checkpoint, outfile, cls=NpEncoder)
