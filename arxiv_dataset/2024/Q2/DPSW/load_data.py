import numpy as np
import pandas as pd
import torch
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

import time

def get_train_val_test(data, tr_ind, val_ind, te_ind):
    return data[tr_ind], data[val_ind], data[te_ind]

class DataLoader(object):
    def __init__(self, args, idx):
        self.args = args
        self._load(idx)

    def _load(self, idx):
        print('----- Loading data -----')
        if self.args.data == 'news':
            """ NEWS DATA LOAD """
            X_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) + '.csv.x')
            AY_data = pd.read_csv('data/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_'+ str(idx) +'.csv.y', header=None)

            X = np.zeros((5000, 3477))
            for doc_id, word, freq in zip(tqdm(X_data['5000']), X_data['3477'], X_data['0']):
                X[doc_id-1][word-1] = freq
           
            A = AY_data[0].values
            Y = AY_data[1].values
            Y0 = AY_data[3].values
            Y1 = AY_data[4].values

            ## add some noise to outcome Y
            noise1 = self.args.noise*np.random.normal(0,1,size=len(X))
            noise0 = self.args.noise*np.random.normal(0,1,size=len(X))
            Y = (Y1 + noise1) * A + (Y0 + noise0) * (1 - A)

            ind = np.arange(len(X))

            tr_ind = ind[:3000]
            val_ind = ind[3000:4000]
            te_ind = ind[4000:]
        elif self.args.data == 'synthH':
            D_tr = np.loadtxt('data/synthH/' + str(idx) + '_' + str(idx) + '_' + str(idx) + '/4_0.csv', delimiter=',')
            D_te = np.loadtxt('data/synthH/' + str(idx) + '_' + str(idx) + '_' + str(idx) + '/4_0.csv', delimiter=',')
            D = np.vstack((D_tr, D_te))
            A = D[:, 0]
            Y0 = D[:, 3]
            Y1 = D[:, 4]
            X = D[:, 5:]
            Y = (1 - A) * D[:, 3] + A * D[:, 4]
            #ind = np.arange(D.shape[0])
            np.random.seed(seed=int(time.time()))
            ind = np.random.permutation(D.shape[0]) ## randomly permute np.arange(0, 20000)
            tr_ind = ind[:10000]
            val_ind = ind[10000:15000]
            te_ind = ind[15000:20000]            



        elif self.args.data == 'LBIDD':
            D = np.load('data/LBIDD/LBIDD.npz')
            A = D['t'][:, idx-1]
            Y = D['yf'][:, idx-1]
            Y0 = D['y0'][:, idx-1]
            Y1 = D['y1'][:, idx-1]
            X = D['x'][:, :, idx-1] # 
            ind = np.arange(len(X)) # n_sample = 10000, n_features = 177, n_experiments = 20
            tr_ind = ind[:7000]
            val_ind = ind[7000:9000]
            te_ind = ind[9000:10000]            


        Y = Y.reshape(-1,1)
        self.scalerY = StandardScaler()
        self.scalerY.fit(Y)
        Y = self.scalerY.transform(Y).squeeze()
        Y0 = self.scalerY.transform(Y0.reshape(-1,1)).squeeze()
        Y1 = self.scalerY.transform(Y1.reshape(-1,1)).squeeze()
        
        self.scalerX = StandardScaler()
        self.scalerX.fit(X)
        X = self.scalerX.transform(X)


        A_tr, A_val, A_te = A[tr_ind], A[val_ind], A[te_ind]
        X_tr, X_val, X_te = X[tr_ind], X[val_ind], X[te_ind]
        Y_tr, Y_val, Y_te = Y[tr_ind], Y[val_ind], Y[te_ind]
        Y0_tr, Y0_val, Y0_te = Y0[tr_ind], Y0[val_ind], Y0[te_ind]
        Y1_tr, Y1_val, Y1_te = Y1[tr_ind], Y1[val_ind], Y1[te_ind]

        self.A_tr = A_tr
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.in_dim = len(X[0])
        num_samples_1 = len(np.where(A_tr == 1)[0])
        num_samples_0 = len(np.where(A_tr == 0)[0])
        self.num_tr = num_samples_0 + num_samples_1
        self.p_tr = float(num_samples_1) / (float(num_samples_0) + float(num_samples_1))

        self.A_te = A_te
        self.X_te = X_te
        self.Y_te = Y_te
        self.Y0_te = Y0_te
        self.Y1_te = Y1_te
        
        ## np.array -> torch.Tensor -> torch.utils.data.DataLoader
        tr_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(tr_ind), torch.Tensor(A_tr), torch.Tensor(X_tr), torch.Tensor(Y_tr))
        val_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(val_ind), torch.Tensor(A_val), torch.Tensor(X_val), torch.Tensor(Y_val))
        val2_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(val_ind), torch.Tensor(X_val), torch.Tensor(Y0_val), torch.Tensor(Y1_val))
        te_data_torch = torch.utils.data.TensorDataset(torch.from_numpy(te_ind), torch.Tensor(X_te), torch.Tensor(Y0_te), torch.Tensor(Y1_te))
        in_data_torch = torch.utils.data.TensorDataset(
            torch.cat((torch.from_numpy(tr_ind),torch.from_numpy(val_ind))),
            torch.Tensor(np.concatenate((X_tr, X_val))), 
            torch.Tensor(np.concatenate((Y0_tr, Y0_val))),
            torch.Tensor(np.concatenate((Y1_tr, Y1_val)))
            )

        self.tr_loader = torch.utils.data.DataLoader(tr_data_torch, batch_size=self.args.batch_size, shuffle=self.args.shuffle, worker_init_fn=np.random.seed(self.args.data_seed))#, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(val_data_torch, batch_size=self.args.batch_size, shuffle=False)#, num_workers=2)
        self.val2_loader = torch.utils.data.DataLoader(val2_data_torch, batch_size=self.args.batch_size, shuffle=False)#, num_workers=2)
        self.te_loader = torch.utils.data.DataLoader(te_data_torch, batch_size=self.args.batch_size, shuffle=False)#, num_workers=2)
        self.in_loader = torch.utils.data.DataLoader(in_data_torch, batch_size=self.args.batch_size, shuffle=False)#, num_workers=2)
        print('----- Finished loading data -----')
