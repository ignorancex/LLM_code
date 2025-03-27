#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:28:04 2023

@author: mary
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torchtuples as tt # Some useful functions
from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard

np.random.seed(121274)
_ = torch.manual_seed(121274)

fold2_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold2_indexes.csv')

train_index_fold2 = fold2_indexes['train'].tolist()

val_index_fold2 = fold2_indexes['validation'].tolist()
val_index_fold2 = [x for x in val_index_fold2 if np.isnan(x) == False]
val_index_fold2 = [int(x) for x in val_index_fold2]

test_index_fold2 = fold2_indexes['test'].tolist()
test_index_fold2 = [x for x in test_index_fold2 if np.isnan(x) == False]
test_index_fold2 = [int(x) for x in test_index_fold2]    

train_val_index_fold2 = sorted(train_index_fold2 + val_index_fold2)

featuredf = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/244_features_noneaugmented.csv')

num_durations = 15
labtrans = LogisticHazard.label_transform(num_durations)

df_train = featuredf.loc[train_val_index_fold2]
df_test = featuredf.loc[test_index_fold2]

scaler = StandardScaler()

imfeat_train = scaler.fit_transform(df_train).astype('float32')
imfeat_test = scaler.transform(df_test).astype('float32')


#Model1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
label_d = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/kits_label_244.csv')
df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
num_nodes = [32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.3

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/CT_74CI_survnet2.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

x_train = imfeat_train
x_test = imfeat_test

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_2(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_2, self).__init__()

        # self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(38, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        # self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train,
                col14_train, col15_train, col16_train, col17_train, col18_train,
                col19_train, col20_train, col21_train, col22_train, col23_train,
                col24_train, col25_train, col26_train, col27_train, col28_train,
                col29_train, col30_train,
                col31_train, col32_train, col33_train, col34_train,
                col35_train, col36_train, col37_train, col38_train
                ):
        
        
        merged_data = torch.cat((col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train, col13_train,
                                 col14_train, col15_train, col16_train, col17_train, col18_train,
                                 col19_train, col20_train, col21_train, col22_train, col23_train,
                                 col24_train, col25_train, col26_train, col27_train, col28_train,
                                 col29_train, col30_train,
                                 col31_train, col32_train, col33_train, col34_train,
                                 col35_train, col36_train, col37_train, col38_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13
    
label_d = pd.read_csv ('csv_files/kits_label_244_wholecols.csv')
df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_2(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/wholecols_72CI74auc.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()
col14_train = (df_train_target['cl14d'].astype('float32')).to_numpy()
col15_train = (df_train_target['cl15d'].astype('float32')).to_numpy()
col16_train = (df_train_target['cl16d'].astype('float32')).to_numpy()
col17_train = (df_train_target['cl17d'].astype('float32')).to_numpy()
col18_train = (df_train_target['cl18d'].astype('float32')).to_numpy()
col19_train = (df_train_target['cl19d'].astype('float32')).to_numpy()
col20_train = (df_train_target['cl20d'].astype('float32')).to_numpy()
col21_train = (df_train_target['cl21d'].astype('float32')).to_numpy()
col22_train = (df_train_target['cl22d'].astype('float32')).to_numpy()
col23_train = (df_train_target['cl23d'].astype('float32')).to_numpy()
col24_train = (df_train_target['cl24d'].astype('float32')).to_numpy()
col25_train = (df_train_target['cl25d'].astype('float32')).to_numpy()
col26_train = (df_train_target['cl26d'].astype('float32')).to_numpy()
col27_train = (df_train_target['cl27d'].astype('float32')).to_numpy()
col28_train = (df_train_target['cl28d'].astype('float32')).to_numpy()
col29_train = (df_train_target['cl29d'].astype('float32')).to_numpy()
col30_train = (df_train_target['cl30d'].astype('float32')).to_numpy()
col31_train = (df_train_target['cl31d'].astype('float32')).to_numpy()
col32_train = (df_train_target['cl32d'].astype('float32')).to_numpy()
col33_train = (df_train_target['cl33d'].astype('float32')).to_numpy()
col34_train = (df_train_target['cl34d'].astype('float32')).to_numpy()
col35_train = (df_train_target['cl35d'].astype('float32')).to_numpy()
col36_train = (df_train_target['cl36d'].astype('float32')).to_numpy()
col37_train = (df_train_target['cl37d'].astype('float32')).to_numpy()
col38_train = (df_train_target['cl38d'].astype('float32')).to_numpy()

col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()
col14_test = (df_test_target['cl14d'].astype('float32')).to_numpy()
col15_test = (df_test_target['cl15d'].astype('float32')).to_numpy()
col16_test = (df_test_target['cl16d'].astype('float32')).to_numpy()
col17_test = (df_test_target['cl17d'].astype('float32')).to_numpy()
col18_test = (df_test_target['cl18d'].astype('float32')).to_numpy()
col19_test = (df_test_target['cl19d'].astype('float32')).to_numpy()
col20_test = (df_test_target['cl20d'].astype('float32')).to_numpy()
col21_test = (df_test_target['cl21d'].astype('float32')).to_numpy()
col22_test = (df_test_target['cl22d'].astype('float32')).to_numpy()
col23_test = (df_test_target['cl23d'].astype('float32')).to_numpy()
col24_test = (df_test_target['cl24d'].astype('float32')).to_numpy()
col25_test = (df_test_target['cl25d'].astype('float32')).to_numpy()
col26_test = (df_test_target['cl26d'].astype('float32')).to_numpy()
col27_test = (df_test_target['cl27d'].astype('float32')).to_numpy()
col28_test = (df_test_target['cl28d'].astype('float32')).to_numpy()
col29_test = (df_test_target['cl29d'].astype('float32')).to_numpy()
col30_test = (df_test_target['cl30d'].astype('float32')).to_numpy()
col31_test = (df_test_target['cl31d'].astype('float32')).to_numpy()
col32_test = (df_test_target['cl32d'].astype('float32')).to_numpy()
col33_test = (df_test_target['cl33d'].astype('float32')).to_numpy()
col34_test = (df_test_target['cl34d'].astype('float32')).to_numpy()
col35_test = (df_test_target['cl35d'].astype('float32')).to_numpy()
col36_test = (df_test_target['cl36d'].astype('float32')).to_numpy()
col37_test = (df_test_target['cl37d'].astype('float32')).to_numpy()
col38_test = (df_test_target['cl38d'].astype('float32')).to_numpy()

col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)
col14_train = col14_train.reshape(len(col14_train),1)
col15_train = col15_train.reshape(len(col15_train),1)
col16_train = col16_train.reshape(len(col16_train),1)
col17_train = col17_train.reshape(len(col17_train),1)
col18_train = col18_train.reshape(len(col18_train),1)
col19_train = col19_train.reshape(len(col19_train),1)
col20_train = col20_train.reshape(len(col20_train),1)
col21_train = col21_train.reshape(len(col21_train),1)
col22_train = col22_train.reshape(len(col22_train),1)
col23_train = col23_train.reshape(len(col23_train),1)
col24_train = col24_train.reshape(len(col24_train),1)
col25_train = col25_train.reshape(len(col25_train),1)
col26_train = col26_train.reshape(len(col26_train),1)
col27_train = col27_train.reshape(len(col27_train),1)
col28_train = col28_train.reshape(len(col28_train),1)
col29_train = col29_train.reshape(len(col29_train),1)
col30_train = col30_train.reshape(len(col30_train),1)
col31_train = col31_train.reshape(len(col31_train),1)
col32_train = col32_train.reshape(len(col32_train),1)
col33_train = col33_train.reshape(len(col33_train),1)
col34_train = col34_train.reshape(len(col34_train),1)
col35_train = col35_train.reshape(len(col35_train),1)
col36_train = col36_train.reshape(len(col36_train),1)
col37_train = col37_train.reshape(len(col37_train),1)
col38_train = col38_train.reshape(len(col38_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)
col14_test = col14_test.reshape(len(col14_test),1)
col15_test = col15_test.reshape(len(col15_test),1)
col16_test = col16_test.reshape(len(col16_test),1)
col17_test = col17_test.reshape(len(col17_test),1)
col18_test = col18_test.reshape(len(col18_test),1)
col19_test = col19_test.reshape(len(col19_test),1)
col20_test = col20_test.reshape(len(col20_test),1)
col21_test = col21_test.reshape(len(col21_test),1)
col22_test = col22_test.reshape(len(col22_test),1)
col23_test = col23_test.reshape(len(col23_test),1)
col24_test = col24_test.reshape(len(col24_test),1)
col25_test = col25_test.reshape(len(col25_test),1)
col26_test = col26_test.reshape(len(col26_test),1)
col27_test = col27_test.reshape(len(col27_test),1)
col28_test = col28_test.reshape(len(col28_test),1)
col29_test = col29_test.reshape(len(col29_test),1)
col30_test = col30_test.reshape(len(col30_test),1)
col31_test = col31_test.reshape(len(col31_test),1)
col32_test = col32_test.reshape(len(col32_test),1)
col33_test = col33_test.reshape(len(col33_test),1)
col34_test = col34_test.reshape(len(col34_test),1)
col35_test = col35_test.reshape(len(col35_test),1)
col36_test = col36_test.reshape(len(col36_test),1)
col37_test = col37_test.reshape(len(col37_test),1)
col38_test = col38_test.reshape(len(col38_test),1)

x_train = (col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train,
           col14_train, col15_train, col16_train, col17_train, col18_train,
           col19_train, col20_train, col21_train, col22_train, col23_train,
           col24_train, col25_train, col26_train, col27_train, col28_train,
           col29_train, col30_train,
           col31_train, col32_train, col33_train, col34_train,
           col35_train, col36_train, col37_train, col38_train)
x_test = (col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test,
          col14_test, col15_test, col16_test, col17_test, col18_test,
          col19_test, col20_test, col21_test, col22_test, col23_test,
          col24_test, col25_test, col26_test, col27_test, col28_test,
          col29_test, col30_test,
          col31_test, col32_test, col33_test, col34_test,
          col35_test, col36_test, col37_test, col38_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model3
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_3(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_3, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+38, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train,
                col14_train, col15_train, col16_train, col17_train, col18_train,
                col19_train, col20_train, col21_train, col22_train, col23_train,
                col24_train, col25_train, col26_train, col27_train, col28_train,
                col29_train, col30_train,
                col31_train, col32_train, col33_train, col34_train,
                col35_train, col36_train, col37_train, col38_train
                ):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train, col13_train,
                                 col14_train, col15_train, col16_train, col17_train, col18_train,
                                 col19_train, col20_train, col21_train, col22_train, col23_train,
                                 col24_train, col25_train, col26_train, col27_train, col28_train,
                                 col29_train, col30_train,
                                 col31_train, col32_train, col33_train, col34_train,
                                 col35_train, col36_train, col37_train, col38_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13
    
label_d = pd.read_csv ('csv_files/kits_label_244_wholecols.csv')
df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_3(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_wholecols_82CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()
col14_train = (df_train_target['cl14d'].astype('float32')).to_numpy()
col15_train = (df_train_target['cl15d'].astype('float32')).to_numpy()
col16_train = (df_train_target['cl16d'].astype('float32')).to_numpy()
col17_train = (df_train_target['cl17d'].astype('float32')).to_numpy()
col18_train = (df_train_target['cl18d'].astype('float32')).to_numpy()
col19_train = (df_train_target['cl19d'].astype('float32')).to_numpy()
col20_train = (df_train_target['cl20d'].astype('float32')).to_numpy()
col21_train = (df_train_target['cl21d'].astype('float32')).to_numpy()
col22_train = (df_train_target['cl22d'].astype('float32')).to_numpy()
col23_train = (df_train_target['cl23d'].astype('float32')).to_numpy()
col24_train = (df_train_target['cl24d'].astype('float32')).to_numpy()
col25_train = (df_train_target['cl25d'].astype('float32')).to_numpy()
col26_train = (df_train_target['cl26d'].astype('float32')).to_numpy()
col27_train = (df_train_target['cl27d'].astype('float32')).to_numpy()
col28_train = (df_train_target['cl28d'].astype('float32')).to_numpy()
col29_train = (df_train_target['cl29d'].astype('float32')).to_numpy()
col30_train = (df_train_target['cl30d'].astype('float32')).to_numpy()
col31_train = (df_train_target['cl31d'].astype('float32')).to_numpy()
col32_train = (df_train_target['cl32d'].astype('float32')).to_numpy()
col33_train = (df_train_target['cl33d'].astype('float32')).to_numpy()
col34_train = (df_train_target['cl34d'].astype('float32')).to_numpy()
col35_train = (df_train_target['cl35d'].astype('float32')).to_numpy()
col36_train = (df_train_target['cl36d'].astype('float32')).to_numpy()
col37_train = (df_train_target['cl37d'].astype('float32')).to_numpy()
col38_train = (df_train_target['cl38d'].astype('float32')).to_numpy()

col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()
col14_test = (df_test_target['cl14d'].astype('float32')).to_numpy()
col15_test = (df_test_target['cl15d'].astype('float32')).to_numpy()
col16_test = (df_test_target['cl16d'].astype('float32')).to_numpy()
col17_test = (df_test_target['cl17d'].astype('float32')).to_numpy()
col18_test = (df_test_target['cl18d'].astype('float32')).to_numpy()
col19_test = (df_test_target['cl19d'].astype('float32')).to_numpy()
col20_test = (df_test_target['cl20d'].astype('float32')).to_numpy()
col21_test = (df_test_target['cl21d'].astype('float32')).to_numpy()
col22_test = (df_test_target['cl22d'].astype('float32')).to_numpy()
col23_test = (df_test_target['cl23d'].astype('float32')).to_numpy()
col24_test = (df_test_target['cl24d'].astype('float32')).to_numpy()
col25_test = (df_test_target['cl25d'].astype('float32')).to_numpy()
col26_test = (df_test_target['cl26d'].astype('float32')).to_numpy()
col27_test = (df_test_target['cl27d'].astype('float32')).to_numpy()
col28_test = (df_test_target['cl28d'].astype('float32')).to_numpy()
col29_test = (df_test_target['cl29d'].astype('float32')).to_numpy()
col30_test = (df_test_target['cl30d'].astype('float32')).to_numpy()
col31_test = (df_test_target['cl31d'].astype('float32')).to_numpy()
col32_test = (df_test_target['cl32d'].astype('float32')).to_numpy()
col33_test = (df_test_target['cl33d'].astype('float32')).to_numpy()
col34_test = (df_test_target['cl34d'].astype('float32')).to_numpy()
col35_test = (df_test_target['cl35d'].astype('float32')).to_numpy()
col36_test = (df_test_target['cl36d'].astype('float32')).to_numpy()
col37_test = (df_test_target['cl37d'].astype('float32')).to_numpy()
col38_test = (df_test_target['cl38d'].astype('float32')).to_numpy()

col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)
col14_train = col14_train.reshape(len(col14_train),1)
col15_train = col15_train.reshape(len(col15_train),1)
col16_train = col16_train.reshape(len(col16_train),1)
col17_train = col17_train.reshape(len(col17_train),1)
col18_train = col18_train.reshape(len(col18_train),1)
col19_train = col19_train.reshape(len(col19_train),1)
col20_train = col20_train.reshape(len(col20_train),1)
col21_train = col21_train.reshape(len(col21_train),1)
col22_train = col22_train.reshape(len(col22_train),1)
col23_train = col23_train.reshape(len(col23_train),1)
col24_train = col24_train.reshape(len(col24_train),1)
col25_train = col25_train.reshape(len(col25_train),1)
col26_train = col26_train.reshape(len(col26_train),1)
col27_train = col27_train.reshape(len(col27_train),1)
col28_train = col28_train.reshape(len(col28_train),1)
col29_train = col29_train.reshape(len(col29_train),1)
col30_train = col30_train.reshape(len(col30_train),1)
col31_train = col31_train.reshape(len(col31_train),1)
col32_train = col32_train.reshape(len(col32_train),1)
col33_train = col33_train.reshape(len(col33_train),1)
col34_train = col34_train.reshape(len(col34_train),1)
col35_train = col35_train.reshape(len(col35_train),1)
col36_train = col36_train.reshape(len(col36_train),1)
col37_train = col37_train.reshape(len(col37_train),1)
col38_train = col38_train.reshape(len(col38_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)
col14_test = col14_test.reshape(len(col14_test),1)
col15_test = col15_test.reshape(len(col15_test),1)
col16_test = col16_test.reshape(len(col16_test),1)
col17_test = col17_test.reshape(len(col17_test),1)
col18_test = col18_test.reshape(len(col18_test),1)
col19_test = col19_test.reshape(len(col19_test),1)
col20_test = col20_test.reshape(len(col20_test),1)
col21_test = col21_test.reshape(len(col21_test),1)
col22_test = col22_test.reshape(len(col22_test),1)
col23_test = col23_test.reshape(len(col23_test),1)
col24_test = col24_test.reshape(len(col24_test),1)
col25_test = col25_test.reshape(len(col25_test),1)
col26_test = col26_test.reshape(len(col26_test),1)
col27_test = col27_test.reshape(len(col27_test),1)
col28_test = col28_test.reshape(len(col28_test),1)
col29_test = col29_test.reshape(len(col29_test),1)
col30_test = col30_test.reshape(len(col30_test),1)
col31_test = col31_test.reshape(len(col31_test),1)
col32_test = col32_test.reshape(len(col32_test),1)
col33_test = col33_test.reshape(len(col33_test),1)
col34_test = col34_test.reshape(len(col34_test),1)
col35_test = col35_test.reshape(len(col35_test),1)
col36_test = col36_test.reshape(len(col36_test),1)
col37_test = col37_test.reshape(len(col37_test),1)
col38_test = col38_test.reshape(len(col38_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train,
           col14_train, col15_train, col16_train, col17_train, col18_train,
           col19_train, col20_train, col21_train, col22_train, col23_train,
           col24_train, col25_train, col26_train, col27_train, col28_train,
           col29_train, col30_train,
           col31_train, col32_train, col33_train, col34_train,
           col35_train, col36_train, col37_train, col38_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test,
          col14_test, col15_test, col16_test, col17_test, col18_test,
          col19_test, col20_test, col21_test, col22_test, col23_test,
          col24_test, col25_test, col26_test, col27_test, col28_test,
          col29_test, col30_test,
          col31_test, col32_test, col33_test, col34_test,
          col35_test, col36_test, col37_test, col38_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

# Model4
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_4(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_4, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+4, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, col4_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13


label_d = pd.read_csv ('csv_files/kits_label_244_spear0.1.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_4(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_spearman0.1_79CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)



col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)


col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])


surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model5
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_5(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_5, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+13, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train, col13_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13
    
label_d = pd.read_csv ('csv_files/kits_label_244_spear0.05.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_5(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_spearman0.05_83CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()


col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()

col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])


surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model6
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_6(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_6, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+30, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train,
                col14_train, col15_train, col16_train, col17_train, col18_train,
                col19_train, col20_train, col21_train, col22_train, col23_train,
                col24_train, col25_train, col26_train, col27_train, col28_train,
                col29_train, col30_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train, col13_train,
                                 col14_train, col15_train, col16_train, col17_train, col18_train,
                                 col19_train, col20_train, col21_train, col22_train, col23_train,
                                 col24_train, col25_train, col26_train, col27_train, col28_train,
                                 col29_train, col30_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13
    
label_d = pd.read_csv ('csv_files/kits_label_244_spear0.01.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_6(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_spearman0.01_81CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()
col14_train = (df_train_target['cl14d'].astype('float32')).to_numpy()
col15_train = (df_train_target['cl15d'].astype('float32')).to_numpy()
col16_train = (df_train_target['cl16d'].astype('float32')).to_numpy()
col17_train = (df_train_target['cl17d'].astype('float32')).to_numpy()
col18_train = (df_train_target['cl18d'].astype('float32')).to_numpy()
col19_train = (df_train_target['cl19d'].astype('float32')).to_numpy()
col20_train = (df_train_target['cl20d'].astype('float32')).to_numpy()
col21_train = (df_train_target['cl21d'].astype('float32')).to_numpy()
col22_train = (df_train_target['cl22d'].astype('float32')).to_numpy()
col23_train = (df_train_target['cl23d'].astype('float32')).to_numpy()
col24_train = (df_train_target['cl24d'].astype('float32')).to_numpy()
col25_train = (df_train_target['cl25d'].astype('float32')).to_numpy()
col26_train = (df_train_target['cl26d'].astype('float32')).to_numpy()
col27_train = (df_train_target['cl27d'].astype('float32')).to_numpy()
col28_train = (df_train_target['cl28d'].astype('float32')).to_numpy()
col29_train = (df_train_target['cl29d'].astype('float32')).to_numpy()
col30_train = (df_train_target['cl30d'].astype('float32')).to_numpy()

col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()
col14_test = (df_test_target['cl14d'].astype('float32')).to_numpy()
col15_test = (df_test_target['cl15d'].astype('float32')).to_numpy()
col16_test = (df_test_target['cl16d'].astype('float32')).to_numpy()
col17_test = (df_test_target['cl17d'].astype('float32')).to_numpy()
col18_test = (df_test_target['cl18d'].astype('float32')).to_numpy()
col19_test = (df_test_target['cl19d'].astype('float32')).to_numpy()
col20_test = (df_test_target['cl20d'].astype('float32')).to_numpy()
col21_test = (df_test_target['cl21d'].astype('float32')).to_numpy()
col22_test = (df_test_target['cl22d'].astype('float32')).to_numpy()
col23_test = (df_test_target['cl23d'].astype('float32')).to_numpy()
col24_test = (df_test_target['cl24d'].astype('float32')).to_numpy()
col25_test = (df_test_target['cl25d'].astype('float32')).to_numpy()
col26_test = (df_test_target['cl26d'].astype('float32')).to_numpy()
col27_test = (df_test_target['cl27d'].astype('float32')).to_numpy()
col28_test = (df_test_target['cl28d'].astype('float32')).to_numpy()
col29_test = (df_test_target['cl29d'].astype('float32')).to_numpy()
col30_test = (df_test_target['cl30d'].astype('float32')).to_numpy()

col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)
col14_train = col14_train.reshape(len(col14_train),1)
col15_train = col15_train.reshape(len(col15_train),1)
col16_train = col16_train.reshape(len(col16_train),1)
col17_train = col17_train.reshape(len(col17_train),1)
col18_train = col18_train.reshape(len(col18_train),1)
col19_train = col19_train.reshape(len(col19_train),1)
col20_train = col20_train.reshape(len(col20_train),1)
col21_train = col21_train.reshape(len(col21_train),1)
col22_train = col22_train.reshape(len(col22_train),1)
col23_train = col23_train.reshape(len(col23_train),1)
col24_train = col24_train.reshape(len(col24_train),1)
col25_train = col25_train.reshape(len(col25_train),1)
col26_train = col26_train.reshape(len(col26_train),1)
col27_train = col27_train.reshape(len(col27_train),1)
col28_train = col28_train.reshape(len(col28_train),1)
col29_train = col29_train.reshape(len(col29_train),1)
col30_train = col30_train.reshape(len(col30_train),1)


col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)
col14_test = col14_test.reshape(len(col14_test),1)
col15_test = col15_test.reshape(len(col15_test),1)
col16_test = col16_test.reshape(len(col16_test),1)
col17_test = col17_test.reshape(len(col17_test),1)
col18_test = col18_test.reshape(len(col18_test),1)
col19_test = col19_test.reshape(len(col19_test),1)
col20_test = col20_test.reshape(len(col20_test),1)
col21_test = col21_test.reshape(len(col21_test),1)
col22_test = col22_test.reshape(len(col22_test),1)
col23_test = col23_test.reshape(len(col23_test),1)
col24_test = col24_test.reshape(len(col24_test),1)
col25_test = col25_test.reshape(len(col25_test),1)
col26_test = col26_test.reshape(len(col26_test),1)
col27_test = col27_test.reshape(len(col27_test),1)
col28_test = col28_test.reshape(len(col28_test),1)
col29_test = col29_test.reshape(len(col29_test),1)
col30_test = col30_test.reshape(len(col30_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train,
           col14_train, col15_train, col16_train, col17_train, col18_train,
           col19_train, col20_train, col21_train, col22_train, col23_train,
           col24_train, col25_train, col26_train, col27_train, col28_train,
           col29_train, col30_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test,
          col14_test, col15_test, col16_test, col17_test, col18_test,
          col19_test, col20_test, col21_test, col22_test, col23_test,
          col24_test, col25_test, col26_test, col27_test, col28_test,
          col29_test, col30_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model7
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_7(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_7, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+4, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, col4_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13

label_d = pd.read_csv ('csv_files/kits_label_244_importance0.1.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_7(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_77CI_survnet2_IF.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

df_train_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.fit_transform(df_train_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')
df_test_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.transform(df_test_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()


col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()


col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)


col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Model8
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_8(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_8, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+18, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train,
                col14_train, col15_train, col16_train, col17_train, col18_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train,
                                 col13_train,col14_train, col15_train, col16_train, 
                                 col17_train, col18_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13

label_d = pd.read_csv ('csv_files/kits_label_244_importance0.01.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_8(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_IF0.01_84CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

df_train_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.fit_transform(df_train_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')
df_test_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.transform(df_test_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')

col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()
col14_train = (df_train_target['cl14d'].astype('float32')).to_numpy()
col15_train = (df_train_target['cl15d'].astype('float32')).to_numpy()
col16_train = (df_train_target['cl16d'].astype('float32')).to_numpy()
col17_train = (df_train_target['cl17d'].astype('float32')).to_numpy()
col18_train = (df_train_target['cl18d'].astype('float32')).to_numpy()


col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()
col14_test = (df_test_target['cl14d'].astype('float32')).to_numpy()
col15_test = (df_test_target['cl15d'].astype('float32')).to_numpy()
col16_test = (df_test_target['cl16d'].astype('float32')).to_numpy()
col17_test = (df_test_target['cl17d'].astype('float32')).to_numpy()
col18_test = (df_test_target['cl18d'].astype('float32')).to_numpy()

col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)
col14_train = col14_train.reshape(len(col14_train),1)
col15_train = col15_train.reshape(len(col15_train),1)
col16_train = col16_train.reshape(len(col16_train),1)
col17_train = col17_train.reshape(len(col17_train),1)
col18_train = col18_train.reshape(len(col18_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)
col14_test = col14_test.reshape(len(col14_test),1)
col15_test = col15_test.reshape(len(col15_test),1)
col16_test = col16_test.reshape(len(col16_test),1)
col17_test = col17_test.reshape(len(col17_test),1)
col18_test = col18_test.reshape(len(col18_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train,
           col14_train, col15_train, col16_train, col17_train, col18_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test,
          col14_test, col15_test, col16_test, col17_test, col18_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])
        
indexes_alive = []
for i, item in enumerate(alive_person_test[1]):
    if item>2000:
        indexes_alive.append(i)
    

surv_test_cont.iloc[:, alive_person_test[0][10:20]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

surv_test_cont.iloc[:, died_person_test[0][2]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

#Violin Plot
#******************************************************************************
data_1 = np.asarray(surv_test_disc.iloc[-1, died_person_test[0][:]])
data_2 = np.asarray(surv_test_disc.iloc[-1, alive_person_test[0][:]])
data_3 = np.asarray(surv_train_disc.iloc[-1, died_person_train[0][:]])
data_4 = np.asarray(surv_train_disc.iloc[-1, alive_person_train[0][:]])
data = list([data_1, data_2, data_3, data_4])
fig, ax = plt.subplots()
ax.violinplot(data, showmeans=True, showmedians=False)
# ax.set_title('violin graph')
# ax.set_xlabel('Different Kinds of Dataset')
ax.set_ylabel('Survival Probabilities')
xticklabels = ['Dead_Test', 'Censored_Test', 'Dead_Train', 'Censored_Train']
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
red = '#B60D0D'
blue = '#23BDC4'
green = '#23C47B'
khaki = '#D2A03D'
violet = '#B86FCD'

violin_parts = ax.violinplot(data, showmeans=True, showmedians=False)

i=0
for vp in violin_parts['bodies']:
    if i==0:
        vp.set_facecolor(blue)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==1:
        vp.set_facecolor(green)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
        
    if i ==2:
        vp.set_facecolor(violet)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    
    if i ==3:
        vp.set_facecolor(khaki)
        vp.set_edgecolor(red)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
    i+=1    
plt.savefig('Violin_plot.eps', format='eps')

#Model9
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class survnet_2_9(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
       
        super(survnet_2_9, self).__init__()

        self.fc1 = torch.nn.Linear(in_features, 1000)
        self.fc2 = torch.nn.Linear(1000+30, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, out_features)
        
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm1d(1000)
        self.BN2 = torch.nn.BatchNorm1d(500)
        self.BN3 = torch.nn.BatchNorm1d(100)

        self.dropout = torch.nn.Dropout(0.1)
        
    
    def forward(self, imfeat_train, col1_train, col2_train, col3_train, 
                col4_train, col5_train, col6_train, col7_train, col8_train,
                col9_train, col10_train, col11_train, col12_train, col13_train,
                col14_train, col15_train, col16_train, col17_train, col18_train,
                col19_train, col20_train, col21_train, col22_train, col23_train,
                col24_train, col25_train, col26_train, col27_train, col28_train,
                col29_train, col30_train):
        
        x1 = self.fc1(imfeat_train)
        
        x2 = self.relu(x1)
        
        x3 = self.BN1(x2)
        
        x4 = self.dropout(x3)
        
        merged_data = torch.cat((x4, col1_train, col2_train, col3_train, col4_train,
                                 col5_train, col6_train, col7_train, col8_train,
                                 col9_train, col10_train, col11_train, col12_train,
                                 col13_train,col14_train, col15_train, col16_train, 
                                 col17_train, col18_train, col19_train, col20_train, col21_train, col22_train, col23_train,
                                 col24_train, col25_train, col26_train, col27_train, col28_train,
                                 col29_train, col30_train), dim=1)
        
        x5 = self.fc2(merged_data)
        
        x6 = self.relu(x5)
        
        x7 = self.BN2(x6)
        
        x8 = self.dropout(x7)
        
        x9 = self.fc3(x8)
        
        x10 = self.relu(x9)
        
        x11 = self.BN3(x10)
        
        x12 = self.dropout(x11)
        
        x13 = self.fc4(x12)
        
        return x13
    
label_d = pd.read_csv ('csv_files/kits_label_244_importance0.001.csv')

df_train_target = label_d.loc[train_val_index_fold2]
df_test_target = label_d.loc[test_index_fold2]

get_target = lambda label_d: (label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train_target))
target_test = labtrans.transform(*get_target(df_test_target))

in_features = imfeat_train.shape[1]
out_features = labtrans.out_features
net = survnet_2_9(in_features, out_features)
model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model.load_model_weights('models/merged_IF0.001_85CI.pt')

n_get_target = lambda label_d: (label_d['case_id'], label_d['survival'].values.astype(int), label_d['event'].values.astype(int))
caseid_test, durations_test, events_test = n_get_target(df_test_target)
caseid_train, durations_train, events_train = n_get_target(df_train_target)

df_train_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.fit_transform(df_train_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')
df_test_target[['cl1d','cl2d','cl3d','cl4d']] = scaler.transform(df_test_target[['cl1d','cl2d','cl3d','cl4d']]).astype('float32')


# In[313]:


col1_train = (df_train_target['cl1d'].astype('float32')).to_numpy()
col2_train = (df_train_target['cl2d'].astype('float32')).to_numpy()
col3_train = (df_train_target['cl3d'].astype('float32')).to_numpy()
col4_train = (df_train_target['cl4d'].astype('float32')).to_numpy()
col5_train = (df_train_target['cl5d'].astype('float32')).to_numpy()
col6_train = (df_train_target['cl6d'].astype('float32')).to_numpy()
col7_train = (df_train_target['cl7d'].astype('float32')).to_numpy()
col8_train = (df_train_target['cl8d'].astype('float32')).to_numpy()
col9_train = (df_train_target['cl9d'].astype('float32')).to_numpy()
col10_train = (df_train_target['cl10d'].astype('float32')).to_numpy()
col11_train = (df_train_target['cl11d'].astype('float32')).to_numpy()
col12_train = (df_train_target['cl12d'].astype('float32')).to_numpy()
col13_train = (df_train_target['cl13d'].astype('float32')).to_numpy()
col14_train = (df_train_target['cl14d'].astype('float32')).to_numpy()
col15_train = (df_train_target['cl15d'].astype('float32')).to_numpy()
col16_train = (df_train_target['cl16d'].astype('float32')).to_numpy()
col17_train = (df_train_target['cl17d'].astype('float32')).to_numpy()
col18_train = (df_train_target['cl18d'].astype('float32')).to_numpy()
col19_train = (df_train_target['cl19d'].astype('float32')).to_numpy()
col20_train = (df_train_target['cl20d'].astype('float32')).to_numpy()
col21_train = (df_train_target['cl21d'].astype('float32')).to_numpy()
col22_train = (df_train_target['cl22d'].astype('float32')).to_numpy()
col23_train = (df_train_target['cl23d'].astype('float32')).to_numpy()
col24_train = (df_train_target['cl24d'].astype('float32')).to_numpy()
col25_train = (df_train_target['cl25d'].astype('float32')).to_numpy()
col26_train = (df_train_target['cl26d'].astype('float32')).to_numpy()
col27_train = (df_train_target['cl27d'].astype('float32')).to_numpy()
col28_train = (df_train_target['cl28d'].astype('float32')).to_numpy()
col29_train = (df_train_target['cl29d'].astype('float32')).to_numpy()
col30_train = (df_train_target['cl30d'].astype('float32')).to_numpy()



col1_test = (df_test_target['cl1d'].astype('float32')).to_numpy()
col2_test = (df_test_target['cl2d'].astype('float32')).to_numpy()
col3_test = (df_test_target['cl3d'].astype('float32')).to_numpy()
col4_test = (df_test_target['cl4d'].astype('float32')).to_numpy()
col5_test = (df_test_target['cl5d'].astype('float32')).to_numpy()
col6_test = (df_test_target['cl6d'].astype('float32')).to_numpy()
col7_test = (df_test_target['cl7d'].astype('float32')).to_numpy()
col8_test = (df_test_target['cl8d'].astype('float32')).to_numpy()
col9_test = (df_test_target['cl9d'].astype('float32')).to_numpy()
col10_test = (df_test_target['cl10d'].astype('float32')).to_numpy()
col11_test = (df_test_target['cl11d'].astype('float32')).to_numpy()
col12_test = (df_test_target['cl12d'].astype('float32')).to_numpy()
col13_test = (df_test_target['cl13d'].astype('float32')).to_numpy()
col14_test = (df_test_target['cl14d'].astype('float32')).to_numpy()
col15_test = (df_test_target['cl15d'].astype('float32')).to_numpy()
col16_test = (df_test_target['cl16d'].astype('float32')).to_numpy()
col17_test = (df_test_target['cl17d'].astype('float32')).to_numpy()
col18_test = (df_test_target['cl18d'].astype('float32')).to_numpy()
col19_test = (df_test_target['cl19d'].astype('float32')).to_numpy()
col20_test = (df_test_target['cl20d'].astype('float32')).to_numpy()
col21_test = (df_test_target['cl21d'].astype('float32')).to_numpy()
col22_test = (df_test_target['cl22d'].astype('float32')).to_numpy()
col23_test = (df_test_target['cl23d'].astype('float32')).to_numpy()
col24_test = (df_test_target['cl24d'].astype('float32')).to_numpy()
col25_test = (df_test_target['cl25d'].astype('float32')).to_numpy()
col26_test = (df_test_target['cl26d'].astype('float32')).to_numpy()
col27_test = (df_test_target['cl27d'].astype('float32')).to_numpy()
col28_test = (df_test_target['cl28d'].astype('float32')).to_numpy()
col29_test = (df_test_target['cl29d'].astype('float32')).to_numpy()
col30_test = (df_test_target['cl30d'].astype('float32')).to_numpy()
# In[314]:


col1_train = col1_train.reshape(len(col1_train),1)
col2_train = col2_train.reshape(len(col2_train),1)
col3_train = col3_train.reshape(len(col3_train),1)
col4_train = col4_train.reshape(len(col4_train),1)
col5_train = col5_train.reshape(len(col5_train),1)
col6_train = col6_train.reshape(len(col6_train),1)
col7_train = col7_train.reshape(len(col7_train),1)
col8_train = col8_train.reshape(len(col8_train),1)
col9_train = col9_train.reshape(len(col9_train),1)
col10_train = col10_train.reshape(len(col10_train),1)
col11_train = col11_train.reshape(len(col11_train),1)
col12_train = col12_train.reshape(len(col12_train),1)
col13_train = col13_train.reshape(len(col13_train),1)
col14_train = col14_train.reshape(len(col14_train),1)
col15_train = col15_train.reshape(len(col15_train),1)
col16_train = col16_train.reshape(len(col16_train),1)
col17_train = col17_train.reshape(len(col17_train),1)
col18_train = col18_train.reshape(len(col18_train),1)
col19_train = col19_train.reshape(len(col19_train),1)
col20_train = col20_train.reshape(len(col20_train),1)
col21_train = col21_train.reshape(len(col21_train),1)
col22_train = col22_train.reshape(len(col22_train),1)
col23_train = col23_train.reshape(len(col23_train),1)
col24_train = col24_train.reshape(len(col24_train),1)
col25_train = col25_train.reshape(len(col25_train),1)
col26_train = col26_train.reshape(len(col26_train),1)
col27_train = col27_train.reshape(len(col27_train),1)
col28_train = col28_train.reshape(len(col28_train),1)
col29_train = col29_train.reshape(len(col29_train),1)
col30_train = col30_train.reshape(len(col30_train),1)

col1_test = col1_test.reshape(len(col1_test),1)
col2_test = col2_test.reshape(len(col2_test),1)
col3_test = col3_test.reshape(len(col3_test),1)
col4_test = col4_test.reshape(len(col4_test),1)
col5_test = col5_test.reshape(len(col5_test),1)
col6_test = col6_test.reshape(len(col6_test),1)
col7_test = col7_test.reshape(len(col7_test),1)
col8_test = col8_test.reshape(len(col8_test),1)
col9_test = col9_test.reshape(len(col9_test),1)
col10_test = col10_test.reshape(len(col10_test),1)
col11_test = col11_test.reshape(len(col11_test),1)
col12_test = col12_test.reshape(len(col12_test),1)
col13_test = col13_test.reshape(len(col13_test),1)
col14_test = col14_test.reshape(len(col14_test),1)
col15_test = col15_test.reshape(len(col15_test),1)
col16_test = col16_test.reshape(len(col16_test),1)
col17_test = col17_test.reshape(len(col17_test),1)
col18_test = col18_test.reshape(len(col18_test),1)
col19_test = col19_test.reshape(len(col19_test),1)
col20_test = col20_test.reshape(len(col20_test),1)
col21_test = col21_test.reshape(len(col21_test),1)
col22_test = col22_test.reshape(len(col22_test),1)
col23_test = col23_test.reshape(len(col23_test),1)
col24_test = col24_test.reshape(len(col24_test),1)
col25_test = col25_test.reshape(len(col25_test),1)
col26_test = col26_test.reshape(len(col26_test),1)
col27_test = col27_test.reshape(len(col27_test),1)
col28_test = col28_test.reshape(len(col28_test),1)
col29_test = col29_test.reshape(len(col29_test),1)
col30_test = col30_test.reshape(len(col30_test),1)

x_train = (imfeat_train, col1_train, col2_train, col3_train, col4_train,
           col5_train, col6_train, col7_train, col8_train,
           col9_train, col10_train, col11_train, col12_train, col13_train,
           col14_train, col15_train, col16_train, col17_train, col18_train,
           col19_train, col20_train, col21_train, col22_train, col23_train,
           col24_train, col25_train, col26_train, col27_train, col28_train,
           col29_train, col30_train)
x_test = (imfeat_test, col1_test, col2_test, col3_test, col4_test,
          col5_test, col6_test, col7_test, col8_test,
          col9_test, col10_test, col11_test, col12_test, col13_test,
          col14_test, col15_test, col16_test, col17_test, col18_test,
          col19_test, col20_test, col21_test, col22_test, col23_test,
          col24_test, col25_test, col26_test, col27_test, col28_test,
          col29_test, col30_test)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

surv_train_disc = model.predict_surv_df(x_train)
surv_test_disc = model.predict_surv_df(x_test)

surv_test_cont = model.interpolate(40).predict_surv_df(x_test)
surv_train_cont = model.interpolate(40).predict_surv_df(x_train)

ev = EvalSurv(surv_test_cont, durations_test, events_test, censor_surv='km')
ctd = ev.concordance_td('antolini')
print(f'C_td Score for test: {ctd}')

ev_train = EvalSurv(surv_train_cont, durations_train, events_train, censor_surv='km')
ctd_train = ev_train.concordance_td('antolini')
print(f'C_td Score for train: {ctd_train}')


died_person_train = [[],[],[]]
alive_person_train = [[],[],[]]
for i,item in enumerate(events_train):
    if item == 1:
        died_person_train[0].append(i)
        died_person_train[1].append(durations_train[i])
        died_person_train[2].append(list(caseid_train)[i])
    else :
        alive_person_train[0].append(i)
        alive_person_train[1].append(durations_train[i])
        alive_person_train[2].append(list(caseid_train)[i])

died_person_test = [[],[],[]]
alive_person_test = [[],[],[]]
for i,item in enumerate(events_test):
    if item == 1:
        died_person_test[0].append(i)
        died_person_test[1].append(durations_test[i])
        died_person_test[2].append(list(caseid_test)[i])
    else :
        alive_person_test[0].append(i)
        alive_person_test[1].append(durations_test[i])
        alive_person_test[2].append(list(caseid_test)[i])

surv_test_cont.iloc[:, died_person_test[0][-3]].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')