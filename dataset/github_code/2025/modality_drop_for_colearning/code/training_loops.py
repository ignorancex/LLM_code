#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:49:49 2023

@author: nicholasmagal

This code contains the training, validation, and testing loops. Note that this code is adapted from https://github.com/pliang279/MFN
"""

import numpy as np
from models import biEFLSTM, MFN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

def train_ef_IM_bi(X_train, y_train, X_valid, y_valid, X_test, y_test, config, total_classes, a_d, v_d, l_d):

    #Shuffle data, comes in N X T X Features    
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    #Swap axis so data is now in T X N X Features 
    X_train = X_train.swapaxes(0,1)
    X_valid = X_valid.swapaxes(0,1)
    X_test = X_test.swapaxes(0,1)

    #How much data we have in our training dataset 
    d = X_train.shape[2]

    #LSTM hidden size
    h = config["h"]

    #How many Time steps 
    t = X_train.shape[0]

    #Output size, 1 for regression 
    output_dim = total_classes

    #Configuration for training
    dropout = config["drop"]
    model = biEFLSTM(d,h,output_dim,dropout)
    optimizer = optim.Adam(model.parameters(),lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
    
    l_dim, a_dim, v_dim = config['input_dims']

    def train(model, batchsize, X_train, y_train, optimizer, criterion,a_d,v_d,l_d):
        epoch_loss = 0
        model.train()

        #Getting total batches
        total_n = X_train.shape[1]
        num_batches = int(total_n // batchsize)

        for batch in range(num_batches):

            optimizer.zero_grad()
            #Getting indexes of batch
            start = batch*batchsize
            end = (batch+1)*batchsize
            batch_X = torch.Tensor(X_train[:,start:end]).to(device)
            batch_y = torch.Tensor(y_train[start:end]).type(torch.LongTensor).to(device)
        
            #applying modality dropout
            draw = random.uniform(0,1)
            if draw <= a_d:
                batch_X[:,:,l_dim:l_dim+a_dim] = 0.0 

                
            if draw <= l_d:
                batch_X[:,:,:l_dim] = 0.0 
                
            if draw <= v_d:
                batch_X[:,:,l_dim + a_dim:] = 0.0 
            
            #Inference
            predictions = model.forward(batch_X)

            #Optimization
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid).to(device)
            batch_y = torch.Tensor(y_valid).type(torch.LongTensor).to(device)
            predictions = model.forward(batch_X)
            epoch_loss = criterion(predictions, batch_y).item()
        return epoch_loss

    def predict(model, X_test,y_test, loss_fn):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).to(device)
            batch_Y = torch.Tensor(y_test).type(torch.LongTensor).to(device)
            predictions = model.forward(batch_X)
            prediction_logits, prediction_classes = torch.max(predictions,1)
            prediction_classes = prediction_classes.cpu().data.numpy()
            
            loss = loss_fn(predictions, batch_Y)
            # print("test loss: ")
            # print(loss.cpu().data.numpy())
        return prediction_classes, loss

    best_valid = 999999.0
    rand = random.randint(0,100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion,a_d,v_d,l_d)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        scheduler.step(valid_loss)
        #wandb.log({"Training Loss": train_loss, "Valid Loss": valid_loss})
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')

            torch.save(model, '../saves/bi_ef_'+str(a_d)+"_"+ str(v_d)+"_"+ str(l_d)+"_"+'.pt' )
        else:
            #print(epoch, train_loss, valid_loss)
            pass
    model = torch.load('../saves/bi_ef_'+str(a_d)+"_"+ str(v_d)+"_"+ str(l_d)+"_"+'.pt')

    predictions, loss = predict(model, X_test, y_test,criterion)
    #print("Confusion Matrix :")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    #print("Classification Report :")
    #print(classification_report(y_test, predictions, digits=5))
    f_score = round(f1_score(predictions,y_test,average='weighted'),5)
    acc = accuracy_score(y_test, predictions)
    #print("Accuracy ", accuracy_score(y_test, predictions))
    
    print([a_d, l_d, v_d, acc, loss, f_score])
    return [a_d, l_d, v_d, acc, loss, f_score, cm] 

def train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, audio_dropout, video_dropout, language_dropout):
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    X_train = X_train.swapaxes(0,1)
    X_valid = X_valid.swapaxes(0,1)
    X_test = X_test.swapaxes(0,1)

    d = X_train.shape[2]
    h = 128
    t = X_train.shape[0]
    output_dim = 1
    dropout = 0.5

    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs

    model = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])

    criterion = nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
    l_dim, a_dim, v_dim = config['input_dims']
    
    
    def train(model, batchsize, X_train, y_train, optimizer, criterion, audio_dropout, language_dropout, video_dropout):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[1]
        num_batches = int(total_n // batchsize)
        for batch in range(num_batches):
            start = batch*batchsize
            end = (batch+1)*batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[:,start:end]).to(device)
            batch_y = torch.Tensor(y_train[start:end]).to(device)
            
            #applying modality dropout
            draw = random.uniform(0,1)
            if draw <= audio_dropout:
                batch_X[:,:,l_dim:l_dim+a_dim] = 0.0 
                
            if draw <= language_dropout:
                batch_X[:,:,:l_dim] = 0.0 
                
            if draw <= video_dropout:
                batch_X[:,:,l_dim + a_dim:] = 0.0 
 
            #This is from MFN network        
            # x_l = x[:,:,:self.d_l]
            # x_a = x[:,:,self.d_l:self.d_l+self.d_a]
            # x_v = x[:,:,self.d_l+self.d_a:]

            predictions = model.forward(batch_X).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid).to(device)
            batch_y = torch.Tensor(y_valid).to(device)
            predictions = model.forward(batch_X).squeeze(1)
            epoch_loss = criterion(predictions, batch_y).item()
        return epoch_loss

    def predict(model, X_test):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).to(device)
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions

    best_valid = 999999.0
    rand = random.randint(0,100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion, audio_dropout, language_dropout, video_dropout)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            torch.save(model, 'output/mfn_%d.pt' %rand)
        else:
            pass
            #print(epoch, train_loss, valid_loss)

    #print('model number is:', rand)
    model = torch.load('output/mfn_%d.pt' %rand)

    predictions = predict(model, X_test)
    mae = np.mean(np.absolute(predictions-y_test))
    #print("mae: ", mae)
    corr = np.corrcoef(predictions,y_test)[0][1]
    #print("corr: ", corr)
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    #print("mult_acc: ", mult)
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    #print("mult f_score: ", f_score)
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    #print("Confusion Matrix :")
    #print(confusion_matrix(true_label, predicted_label))
    #print("Classification Report :")
    #print(classification_report(true_label, predicted_label, digits=5))
    acc = accuracy_score(true_label, predicted_label)
    #print("Accuracy ",acc )
    
    return [audio_dropout, language_dropout, video_dropout, acc, mae, f_score]