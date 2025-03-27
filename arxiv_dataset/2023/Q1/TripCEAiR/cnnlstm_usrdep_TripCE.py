import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import math
from scipy import signal
import scipy.io
from scipy import signal
from scipy import interpolate
import mne
from sklearn.metrics import accuracy_score,confusion_matrix
from scipy.stats import zscore
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torchsummary import summary
from pytorch_metric_learning import distances, losses, miners, reducers, testers



device = 'cuda:0'

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.bn = nn.BatchNorm1d(5)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv1d(5, 128, kernel_size=10, stride=1,padding='valid') 
        self.pool1 = nn.MaxPool1d(3,3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=10, stride=1,padding='valid')
        self.pool2 = nn.MaxPool1d(3,3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=10, stride=1,padding='valid')
        self.pool3 = nn.MaxPool1d(3,3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=10, stride=1,padding='valid')
        self.pool4 = nn.MaxPool1d(3,3)
        self.blstm = nn.LSTM(256,512, bidirectional=False,batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(512,256)
        self.lin = nn.Linear(256,26)

    def forward(self, x):
        x = self.bn(x)
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act1(self.conv2(x))
        x = self.pool2(x)
        x = self.act1(self.conv3(x))
        x = self.pool3(x)
        x = self.act1(self.conv4(x))
        x = self.pool4(x)
        x = x.permute(0,2,1)
        x,(a,b) = self.blstm(x)
        x = x[:, -1, :]
        
        x = self.lin1(x)
        
        x1 = self.drop(x)
        x1 = self.lin(x1)
        return x,x1

def process_EMG(trial,duration,interp_type):

    X = np.zeros((2600,5,int(duration/4)))
    Y = np.zeros((2600,))
    testctr = 0
    
    for folder in os.listdir('/home/ayush/Documents/EMG_50/EMG_Data'):
        for rep in range (1,3):
            for num in range(65,91):
                emg_sig = abs(np.load('/home/ayush/Documents/EMG_50/EMG_Data/'+folder+'/Segmented/EMG_'+chr(num)+'_'+str(trial+1)+'_'+str(rep)+'.npy'))
                
                emg_signal_resampled = np.zeros((int(len(emg_sig)/4),5))
                
                emg_signal = np.zeros((int(duration/4),5))
                
                for i in range(0,5):
                    emg_signal_resampled[:,i] = signal.resample(emg_sig[:,i],int(len(emg_sig)/4))
                    
                if len(emg_signal_resampled)<=int(duration/4):
                    for i in range(0,5):
                        x = np.linspace(0, 1,len(emg_signal_resampled))
                        y = emg_signal_resampled[:,i]
                        f = interpolate.interp1d(x, y,kind=interp_type)
                        xnew = np.linspace(0, 1,int(duration/4))
                        emg_signal[:,i] = f(xnew)

                else:
                    emg_signal = emg_signal_resampled[0:int(duration/4),:]
 
                X[testctr,:,:] = np.transpose(emg_signal)                 
                Y[testctr] = num-65
                testctr = testctr+1
                
                
    #print(testctr)
    return X,Y
    

folderarr = []
for fo in os.listdir('/home/ayush/Documents/EMG_50/EMG_Data'):
    folderarr.append(fo)
    
duration = 8000

print('User Dependent: CNN-LSTM')

for loss_type in ['NP', 'CD', 'L2']:    
    print(loss_type)
    accarr = []
    
    for fold_no in range(0,5):
    
        X_train = np.zeros((1,5,int(duration/4)))
        X_test = np.zeros((1,5,int(duration/4)))
        Y_train = np.zeros((1,))
        Y_test = np.zeros((1,))
    
        #print('Fold number : ' + str(fold_no))
    
        for fold_id in range(0,5):
            if fold_id==fold_no:
                [X,Y] = process_EMG(fold_id,8000,'cubic')
                X_test = np.concatenate((X_test,X))
                Y_test = np.concatenate((Y_test,Y))
                
            else:
                [X,Y] = process_EMG(fold_id,8000,'cubic')
                X_train = np.concatenate((X_train,X))
                Y_train = np.concatenate((Y_train,Y))
                    
        X_train = X_train[1:,:,:]
        X_test = X_test[1:,:,:]   
        Y_test = Y_test[1:,] 
        Y_train = Y_train[1:,]    
        
        
        for i in range(0,len(X_train)):
            for j in range(0,5):
                X_train[i,:,j] = (X_train[i,:,j]-np.mean(X_train[i,:,j]))/np.std(X_train[i,:,j])
                
        for i in range(0,len(X_test)):
            for j in range(0,5):
                X_test[i,:,j] = (X_test[i,:,j]-np.mean(X_test[i,:,j]))/np.std(X_test[i,:,j])
                
        X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
        
        model = NeuralNet().to(device)
        
        
        #print(summary(model, (5,2000)))
        
        if loss_type == 'L2':
            loss_func = losses.TripletMarginLoss(margin=0.2,distance=distances.LpDistance(normalize_embeddings=True, p=2, power=2),smooth_loss=True)
            
        if loss_type == 'NP':
            loss_func = losses.NTXentLoss(temperature=0.2)
            
        if loss_type == 'CD':
            loss_func = losses.TripletMarginLoss(margin=0.1,distance=distances.CosineSimilarity(),smooth_loss=True)
        
        
        mining_func = miners.BatchEasyHardMiner(pos_strategy='semihard',neg_strategy='hard')
    
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters() , lr = 0.0005)
        
        best_val_acc = -1000
        best_val_model = None
        batch_size = 260
        val_acc_arr = -1000*np.ones((10,))
        es_patience=10+1
        num_epochs = 1000
        
        #print('Started training')
        for epoch in range(0,num_epochs):  
            model.train(True)
            running_loss = 0.0
            running_acc = 0
            running_val_loss = 0.0
            for i in range(0,len(X_train_final)//batch_size):
                inputs = torch.tensor(X_train_final[i*batch_size:(i+1)*batch_size,:,:], dtype=torch.float).to(device)
                labels = torch.tensor(Y_train_final[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
        
                optimizer.zero_grad()
                embeddings = model(inputs)[0]
                outputs = model(inputs)[1]
                indices_tuple = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, indices_tuple) + criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item() * inputs.size(0)
                out = torch.argmax(outputs.detach(),dim=1)
                assert out.shape==labels.shape
                running_acc += (labels==out).sum().item()
        
            correct = 0
            model.train(False)
            with torch.no_grad():
                for i in range(0,len(X_val_final)//batch_size):
                    inputs = torch.tensor(X_val_final[i*batch_size:(i+1)*batch_size,:,:], dtype=torch.float).to(device)
                    labels = torch.tensor(Y_val_final[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
                    embeddings = model(inputs)[0]
                    out = model(inputs)[1]
                    indices_tuple = mining_func(embeddings, labels)
                    val_loss = loss_func(embeddings, labels, indices_tuple) + criterion(out,labels)
                    out = torch.argmax(out,dim=1)
                    acc = (out==labels).sum().item()
                    correct += acc
                    running_val_loss += val_loss.item() * inputs.size(0)
            #print(f"Train loss {epoch+1}: {running_loss/len(X_train_final)},Train Acc:{running_acc*100/len(X_train_final)}")
            #print(f"Validation loss: {running_val_loss/len(X_val_final)},Validation Acc:{correct*100/len(X_val_final)}%")
            #print(f" ")
            
            '''
            if correct>best_val_acc:
                best_val_acc = correct
                best_val_model = deepcopy(model.state_dict())
            '''
        
            val_acc_arr = np.concatenate((val_acc_arr,np.asarray(np.reshape(correct*100/len(X_val_final),(1,)))))
            if val_acc_arr[-es_patience] == np.max(val_acc_arr[-es_patience:]):
                break
        
        #print('Finished Training')
        
        predictions = np.zeros((len(X_test),26))
        
        for i in range(0,len(X_test)):
            test_preds = model(torch.tensor(np.reshape(X_test[i,:,:],(1,5,2000)), dtype=torch.float).to(device))[1]
            predictions[i] = test_preds.cpu().detach().numpy()
            
        Y_pred = np.argmax(np.asarray(predictions),axis=1)
        acc = accuracy_score(Y_test,Y_pred)
        
        print(acc)
        accarr.append(acc)
    
        del model
                
    print('Overall accuracy: ' + str(np.mean(accarr))) 