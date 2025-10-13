import numpy as np
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
    
    if ns.f_feat:
        Pks, labels, lenk = NINf.load_Pkf_feat(ns.input, ns.fraction_path, ns.k_max, 0, ns.log, ns.multipole, ns.norm_fs, ns.norm_c)
        Pks[:,:-1], norm = NIN.normalize_Pk(Pks[:,:-1])
    else:
        Pks, labels, lenk = NIN.load_Pk(ns.input, ns.k_max, ns.log, ns.multipole)
        Pks, norm = NIN.normalize_Pk(Pks)
    
    n_train = int(len(labels)*ns.train_fraction)
    n_val = int(len(labels)*ns.val_fraction)
    
    Pks_train = Pks[:n_train]
    Pks_val = Pks[n_train:n_val]
    #Pks_test = Pks[n_val:]
    
    labels_train = labels[:n_train]
    labels_val = labels[n_train:n_val]
    #labels_test = labels[n_val:]

    norm_train = np.array(norm[:n_train, np.newaxis])
    norm_val   = np.array(norm[n_train:n_val, np.newaxis])

    train = NIN.PkDataset(Pks_train, labels_train, norm_train)
    val = NIN.PkDataset(Pks_val, labels_val, norm_val)
    #test = NIN.PkDataset(Pks_test, labels_test)
    
    train_dataloader = DataLoader(train, batch_size=ns.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=ns.batch_size, shuffle=True)
    #test_dataloader = DataLoader(test, batch_size=ns.batch_size, shuffle=True)
    
    #input and output dimensions
    input_size = lenk if ns.multipole == 0 else lenk*2
    output_size = lenk
    n_out = ns.neurons if ns.multipole == 0 else ns.neurons/2

    #preparing model
    if ns.f_feat:
        model = NIN.NoInterNet_fraction_compress_knowf(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)
    else:
        model = NIN.NoInterNet_compress(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)
    loss_fn = NIN.LogMSELoss() if ns.logLoss else nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=ns.learning_rate)
    
    #training
    train_history, val_history = NIN.training(train_dataloader, val_dataloader, model, loss_fn, optimizer, ns.epochs, ns.patience, ns.output, ns.neurons)
    
    #results
    plt.semilogy(train_history, label="training set")
    plt.semilogy(val_history, label="test set")
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.savefig(ns.output+"loss_PT.pdf")
    plt.close()
    
    out_name = "params.dat"
    out_path = ns.output + out_name
    
    with open(out_path, 'w') as f:
        for k in vars(ns):
            print(k, getattr(ns, k), file=f)
        print('Total_trainng_epochs', len(train_history), file=f)
        
    history_name = "history.dat"
    history_path = ns.output + history_name
    
    np.savetxt(history_path, np.vstack([train_history, val_history]).transpose(), header="Train Val")
    
if __name__ == '__main__':
    desc = 'Train a dense NN that takes as inputs the interloper-contaminated Pk and outputs the Pk correction'
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'input to csv file containg the path to the Pk (first column contaminated, second column not contaminated)'
    group.add_argument('--input', type=str, help=h, required=True)

    h = 'output folder path'
    group.add_argument('--output', type=str, help=h, required=True)    

    #optional arguments

    h = 'maximum k in the Pk'
    parser.add_argument('--k-max', type=float, default=.8, help=h)

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

    h = 'fractions file'
    parser.add_argument('--fraction-path', type=str, default='', help=h)
    
    #flags

    h = 'input log10(Pk)'
    parser.add_argument('--log', action='store_true', help=h)

    h = 'use LogMSELoss?'
    parser.add_argument('--logLoss', action='store_true', help=h)

    h = 'f as feature'
    parser.add_argument('--f-feat', action='store_true', help=h)

    h = 'rescale f in [0,1]'
    parser.add_argument('--norm-fs', action='store_true', help=h)

    h = 'rescale correction in [0,1]'
    parser.add_argument('--norm-c', action='store_true', help=h)
    
    # and go!
    main(parser.parse_args())
