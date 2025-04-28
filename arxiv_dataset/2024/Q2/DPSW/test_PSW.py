from catenets.models.jax import SNet1
from catenets.models.torch import TARNet
from catenets.models.torch.base import PropensityNet
from arviz import psislw as PSIS
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from config import args
from load_data import DataLoader
import torch

print(args)
method='PSW'


def main(idx, method):

    Data = DataLoader(args=args, idx=idx)
    X = Data.X_tr
    Y = Data.Y_tr
    T = Data.A_tr
    X_te = Data.X_te
    Y_te = Data.Y_te
    T_te = Data.A_te
    Y0_te = Data.Y0_te
    Y1_te = Data.Y1_te

    data_scale = Data.scalerY.scale_.item()

    if method == 'PSW':
        if args.data == 'news' or args.data == 'LBIDD':
            model_p = PropensityNet(n_unit_in=X.shape[1], n_unit_out = 100, name='IPW', weighting_strategy = 'ipw', n_layers_out_prop=3, n_units_out_prop=100, n_iter=300, batch_size=5000)
            X = torch.from_numpy(X).to(torch.float32)
            T = torch.from_numpy(T).to(torch.float32)
            X_te = torch.from_numpy(X_te).to(torch.float32)
            T_te = torch.from_numpy(T_te).to(torch.float32)
            model_p.fit(X, T)
            ipw = model_p.get_importance_weights(X, T).detach().numpy()
            ipw_te =  model_p.get_importance_weights(X_te, T_te).detach().numpy()
            # print(ipw)
            # print(ipw_te)
            pipw, _ = PSIS(np.log(ipw))
            pipw_te, _ = PSIS(np.log(ipw_te))
            pipw = np.exp(pipw)
            pipw_te = np.exp(pipw_te)
            
            print("PSIS")
            print(np.max(pipw))
            print(np.max(pipw_te))
            # sys.exit()

    return pipw, pipw_te



if __name__ == '__main__':
    dataname = args.data
    indices = args.data_index.strip().split(',')
    indices = [int(idx) for idx in indices]
    flag_notall = indices[0]
    if args.data == 'news':
        num_of_datasets = 50
    elif args.data == 'LBIDD':
        num_of_datasets = 20

    if args.data == 'synthH':
        num_of_datasets = 1
        indices = [args.data_index] * num_of_datasets


    if args.data == 'synthH':
        num_of_datasets = 20
        indices = [args.data_index] * num_of_datasets
    else:
        indices = range(1, num_of_datasets + 1)

    if args.data == 'news':
        dir_name = 'data/NEWS_csv/'
        num_tr = 3000; num_te = 1000
    elif args.data == 'LBIDD':
        dir_name = 'data/LBIDD/'
        num_tr = 7000; num_te = 1000
    
    W_tr = np.zeros((num_tr, num_of_datasets))
    W_te = np.zeros((num_te, num_of_datasets))
    

    for i in range(len(indices)):
        pipw, pipw_te = main(indices[i], method)
        W_tr[:, i] = pipw
        W_te[:, i] = pipw_te
        
    np.savetxt(dir_name + 'pipw_tr.csv', W_tr, delimiter=',')
    np.savetxt(dir_name + 'pipw_te.csv', W_te, delimiter=',')

