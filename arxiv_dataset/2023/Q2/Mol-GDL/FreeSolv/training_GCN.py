import torch
import torch.nn as nn
from models.gcn import GCNNet
from utils_test import *
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score

from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np

def train(model, device, loader_train, optimizer, epoch):
#    print('Training on {} samples...'.format(len(loader_train[0].dataset)))
    model.train()
    
    arr_data = []
    for batch_idx, data in enumerate(loader_train):
        arr_data_tmp = []
        for idx, data_data in enumerate(data):
            arr_data_tmp.append(data_data)
        arr_data.append(arr_data_tmp)
    for i in range(len(arr_data[0])):
        optimizer.zero_grad()
        arr_data_data = [a[i] for a in arr_data]
        y = arr_data_data[0].y.to(device)
        output = model(arr_data_data, device)
        loss = loss_fn(output, y)
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
        
def predicting(model, device, loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        arr_data = []
        for batch_idx, data in enumerate(loader_test):
            arr_data_tmp = []
            for idx, data_data in enumerate(data):
                arr_data_tmp.append(data_data)
            arr_data.append(arr_data_tmp)
            
        for i in range(len(arr_data[0])): 
            arr_data_data = [a[i] for a in arr_data] 
            output = model(arr_data_data, device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, arr_data[0][i].y.cpu()), 0)    
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

modeling = GCNNet

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

data_train = TestbedDataset(root='data', dataset='freesolv_train')
data_valid = TestbedDataset(root='data', dataset='freesolv_validation')
data_test = TestbedDataset(root='data', dataset='freesolv_test')

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1501
NUM_NET = 5

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

length_train = int(len(data_train)/NUM_NET)
length_valid = int(len(data_valid)/NUM_NET)
length_test = int(len(data_test)/NUM_NET)


print("length of training set:",length_train)
print("length of validation set:",length_valid)
print("length of testing set:",length_test)

train_num = np.linspace(0, length_train-1, length_train)
valid_num = np.linspace(0, length_valid-1, length_valid)
test_num = np.linspace(0, length_test-1, length_test)


loader_train = []
loader_valid = []
loader_test = []
for a in range(NUM_NET):
    train_num_tmp = [int(n * NUM_NET + a) for n in train_num]
    valid_num_tmp = [int(n * NUM_NET + a) for n in valid_num]
    test_num_tmp = [int(n * NUM_NET + a) for n in test_num]
    
    data_train_tmp = data_train[train_num_tmp]
    data_valid_tmp = data_valid[valid_num_tmp]
    data_test_tmp = data_test[test_num_tmp]
    
    loader_train.append(DataLoader(data_train_tmp, batch_size=TRAIN_BATCH_SIZE, shuffle=None))
    loader_valid.append(DataLoader(data_valid_tmp, batch_size=VALID_BATCH_SIZE, shuffle=None))
    loader_test.append(DataLoader(data_test_tmp, batch_size=TEST_BATCH_SIZE, shuffle=None))


for i in range(5):
    
    model = modeling().to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_rmse = 1000
    for epoch in range(NUM_EPOCHS):
        train(model, device, loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, loader_test)
        rmse = np.sqrt(mean_squared_error(T, S))
        if rmse < best_rmse:
            best_rmse = rmse
        if epoch % 100 == 0:
            print("-------------------------------------------------------")
            print("epoch:",epoch)
            print('best_rmse:', best_rmse)
