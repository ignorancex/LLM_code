import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import NoInterNet_model as NIN
import NoInterNet_fraction_model as NINf
import argparse

import torch

from torch.utils.data import DataLoader
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def main(ns):
    #preparing data
    print(ns)
    print('Running on', device)
       
    n_load = None if ns.n_load == -99 else ns.n_load

    if ns.PB:
        Pks, labels, lenk, lenB = NINf.load_PBf(ns.input, ns.fraction_path, ns.k_max, ns.k_min, ns.log, norm_fs=ns.norm_fs, norm_c=ns.norm_c, n_load=n_load)
    elif ns.B:
        Pks, labels, lenB, lenk = NINf.load_PBf_B(ns.input, ns.fraction_path, ns.k_max, ns.k_min, ns.log, norm_fs=ns.norm_fs, norm_c=ns.norm_c, n_load=n_load)
        #technically lenB is lenk, but I swap the names not to change the building of the NN
    else:
        Pks, labels, lenk = NINf.load_Pkf(ns.input, ns.fraction_path, ns.k_max, ns.k_min, ns.log, ns.multipole, ns.norm_fs, ns.norm_c, n_load=n_load)
        lenB = 0

    #add cosmology prior
    Pk_paths = pd.read_csv(ns.input)
    sP = {'sigma_8': 0.0060, 'Omega_m': 0.0056, 'h': 0.0042, 'n_s': 0.0038, 'Omega_b': 0.0008} #from Table 2 in Planck 2018 (1807.06209)
    len_priors = len(ns.priors)
    r = np.random.normal(size=(len(Pks), len_priors))
    cosmo_p = np.zeros((len(Pks), len_priors))

    for i, prior in enumerate(ns.priors):
        if len_priors == 1:
            cosmo_p = np.reshape(Pk_paths[prior],((len(Pks),1))) + r * sP[prior] * ns.NPlanck #N times Planck
        else:
            cosmo_p[:,i] = Pk_paths[prior][:n_load] + r[:,i] * sP[prior] * ns.NPlanck #N times Planck

    print(cosmo_p)
    Pks = np.hstack((Pks, cosmo_p)) #I also normalize cosmo_p between 0 and 1
    
    #maximin normalization
    max_Pks = Pks.max(axis=0)
    min_Pks = Pks.min(axis=0)
    Pks = NINf.maxmin_corr(Pks, max_Pks, min_Pks)
    norm = np.ones(len(Pks))

    #There was no information leakage in outbox, a bit of it in inbox
    #So here there is an overcomplicated way to remove it, enjoy
    fs = labels[:,-1]

    _, ind = np.unique(fs, return_index=True)
    f_u = np.array([fs[indx] for indx in sorted(ind)]) #non ordered unique interloper fraction array
    
    n_train = int(len(f_u)*ns.train_fraction)
    n_val = int(len(f_u)*ns.val_fraction)
    
    fs_train = (f_u[:n_train])
    fs_val = (f_u[n_train:n_val])
    #fs_test = (f_u[n_val:])

    sel_train = np.array([], dtype=int)
    sel_val = np.array([], dtype=int)
    
    for f in fs_train: sel_train= np.append(sel_train, np.where(fs == f)[0])
    for f in fs_val: sel_val= np.append(sel_val, np.where(fs == f)[0])
    
    Pks_train = Pks[sel_train]
    Pks_val = Pks[sel_val]
    
    labels_train = labels[sel_train]
    labels_val = labels[sel_val]
    
    norm_train = np.array(norm[sel_train, np.newaxis])
    norm_val   = np.array(norm[sel_val, np.newaxis])

    train = NIN.PkDataset(Pks_train, labels_train, norm_train)
    val = NIN.PkDataset(Pks_val, labels_val, norm_val)
    
    train_dataloader = DataLoader(train, batch_size=ns.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=ns.batch_size, shuffle=True)
    
    #input and output dimensions
    input_size = lenk + lenB + len_priors
    output_size = lenk
    n_out = ns.neurons if ns.multipole == 0 else ns.neurons/2
    
    #preparing model
    model = NINf.NoInterNet_fraction_compress_inference(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ns.learning_rate)
    
    #training
    train_history, val_history = NINf.training_inference(train_dataloader, val_dataloader, model, optimizer, ns.epochs, ns.patience, ns.output, lenk, ns.neurons)

    #results
    plt.plot(train_history, label="training set")
    plt.plot(val_history, label="test set")
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.savefig(ns.output+"loss_PT.pdf")
    plt.close()
    
    out_name = "params.dat"
    out_path = ns.output + out_name
    
    with open(out_path, 'w') as f:
        for k in vars(ns):
            print(k, getattr(ns, k), file=f)
        print('Input_size', input_size, file=f)
        print('Output_size', output_size, file=f)
        print('N_out', n_out, file=f)
        print('Total_trainng_epochs', len(train_history), file=f)
        
    history_name = "history.dat"
    history_path = ns.output + history_name
    
    np.savetxt(history_path, np.vstack([train_history, val_history]).transpose(), header="Train Val")

    maxmin_name = "maxmin.dat"
    maxmin_path = ns.output + maxmin_name

    np.savetxt(maxmin_path, np.vstack([min_Pks, max_Pks]).transpose(), header="Min Max")

    if ns. opt:
        return val_history
    
if __name__ == '__main__':
    desc = 'Train a dense NN that takes as inputs the interloper-contaminated Pk and outputs the Pk correction and the outlier fraction'
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'input to csv file containg the path to the Pk (first column contaminated, second column not contaminated)'
    group.add_argument('--input', type=str, help=h, required=True)

    h = 'fractions file'
    group.add_argument('--fraction-path', type=str, help=h, required=True)

    h = 'output folder path'
    group.add_argument('--output', type=str, help=h, required=True)    

    #optional arguments

    h = 'maximum k in the Pk'
    parser.add_argument('--k-max', type=float, default=.8, help=h)

    h = 'minimum k in the Pk'
    parser.add_argument('--k-min', type=float, default=0, help=h)

    h = 'cosmological parameters to sample'
    parser.add_argument('--priors', type=str, nargs='+', default=['sigma_8'], help=h)

    h = 'fraction of P(k)s used for training'
    parser.add_argument('--train-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.75, help=h)
    
    h = 'fraction of P(k)s used for validation (must be greater than tranin-fraction) '
    parser.add_argument('--val-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.90, help=h)

    h = 'number of neurons of hidden layers, or of the first hidden layers'
    parser.add_argument('--neurons', type=int, default=60, help=h)
    
    h = 'minimum number of neurons'
    parser.add_argument('--n-min', type=int, default=8, help=h)
    
    h = 'learning rate'
    parser.add_argument('--learning-rate', type=float, default=1e-3, help=h)
    
    h = 'batch size'
    parser.add_argument('--batch-size', type=int, default=64, help=h)

    h = 'maximum number of epochs for training'
    parser.add_argument('--epochs', type=int, default=400, help=h)
    
    h = 'patience for earlystopping'
    parser.add_argument('--patience', type=int, default=20, help=h)

    h = 'multipole to use, either 0 or 0 and 2'
    parser.add_argument('--multipole', type=int, default=0, choices=[0,2], help=h)

    h = 'number of times we enlarge Planck'
    parser.add_argument('--NPlanck', type=float, default=5., help=h)

    h = 'number of data to load, if -99 it loads all the data in the csv'
    parser.add_argument('--n-load', type=int, default=-99, help=h)
    
    #flags

    h = 'input log10(Pk)'
    parser.add_argument('--log', action='store_true', help=h)

    h = 'use LogMSELoss?'
    parser.add_argument('--logLoss', action='store_true', help=h)

    h = 'hyper-parameter optimization flag'
    parser.add_argument('--opt', action='store_true', help=h)

    h = 'add log10(normalization) to input variables'
    parser.add_argument('--norm-feature', action='store_true', help=h)

    h = 'rescale f in [0,1]'
    parser.add_argument('--norm-fs', action='store_true', help=h)

    h = 'rescale correction in [0,1]'
    parser.add_argument('--norm-c', action='store_true', help=h)

    h = 'use B0 as well'
    parser.add_argument('--PB', action='store_true', help=h)

    h = 'correct B0'
    parser.add_argument('--B', action='store_true', help=h)

    # and go!
    main(parser.parse_args())