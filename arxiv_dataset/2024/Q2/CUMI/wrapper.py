from tqdm import tqdm
import torch
from utils import set_seed, get_kernelsize, calculate_MI, reyi_entropy, calculate_TC
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os, yaml
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import classification_report
from copy import deepcopy as copy
import warnings
import scipy.io as sio

warnings.filterwarnings("ignore")

set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE_loss = nn.CrossEntropyLoss()
recon_loss = nn.MSELoss()

def train_IBCI(args, trainloader, testloader, net):
    epochs = args.epochs
    learning_rate = args.lr
    beta = args.beta
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
    best_acc = 0

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data['views'], data['label']
            inputs, labels =[input.to(device) for input in inputs], labels.to(device) 
            optimizer.zero_grad()
            inputs, inputs_hat, com_feature, uni_features, outputs = net(inputs)
            sigma_com = get_kernelsize(com_feature, selected_param=0.15, select_type='meadian')
            sigma_uni_fea = [get_kernelsize(uni_fea, selected_param=0.15, select_type='meadian') for uni_fea in uni_features]

            loss_ce = CE_loss(outputs, labels)
            reconstruction_error = [recon_loss(input_i, input_hat_i) for (input_i, input_hat_i) in zip(inputs, inputs_hat)]
            H_zc = reyi_entropy(com_feature, sigma_com)
            TC_loss = calculate_TC(uni_features, sigma_uni_fea)
            loss = loss_ce +  + sum(reconstruction_error) + beta[0] * H_zc + beta[1] * TC_loss

            loss.backward()
            optimizer.step()
        print(f"The current epoch is {epoch}, the current accuracy is {best_acc}.")
        report = test_IBCI(args, testloader, net)
        if report[0] > best_acc:
            best_acc = report[0]
            best_result = report
    
    return best_result 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_IBCI(args, testloader, net):
    net.eval()
    with torch.no_grad():
        correct = 0 
        total = 0
        y_true = []
        y_pred = []
        for i, data in tqdm(enumerate(testloader, 0)):
            inputs, labels = data['views'], data['label']
            inputs, labels =[input.to(device) for input in inputs], labels.to(device) 
            _, _, _, _, outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true += labels.tolist()
            y_pred += predicted.tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        acc, precision, recall, f1_score   = report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']
        return [acc, precision, recall, f1_score]

