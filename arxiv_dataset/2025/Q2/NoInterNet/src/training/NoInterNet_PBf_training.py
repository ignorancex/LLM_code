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

    PBs, labels, lenP, lenB = NINf.load_PBf(ns.input, ns.fraction_path, ns.k_max, ns.k_min, ns.log, norm_fs=ns.norm_fs, norm_c=ns.norm_c)
    
    max_PBs = PBs.max(axis=0)
    min_PBs = PBs.min(axis=0)
    PBs = NINf.maxmin_corr(PBs, max_PBs, min_PBs)
    norm = np.ones(len(PBs))
    
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
    #sel_test = np.array([], dtype=int)
    
    for f in fs_train: sel_train= np.append(sel_train, np.where(fs == f)[0])
    for f in fs_val: sel_val= np.append(sel_val, np.where(fs == f)[0])
    #for f in fs_test: sel_test= np.append(sel_test, np.where(fs == f)[0])
    
    PBs_train = PBs[sel_train]
    PBs_val = PBs[sel_val]
    #Pks_test = Pks[sel_test]
    
    labels_train = labels[sel_train]
    labels_val = labels[sel_val]
    #labels_test = labels[sel_test]

    norm_train = np.array(norm[sel_train, np.newaxis])
    norm_val   = np.array(norm[sel_val, np.newaxis])

    train = NINf.PBDataset(PBs_train, labels_train, norm_train)
    val = NINf.PBDataset(PBs_val, labels_val, norm_val)
    #test = NIN.PkDataset(Pks_test, labels_test)
    
    train_dataloader = DataLoader(train, batch_size=ns.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=ns.batch_size, shuffle=True)
    #test_dataloader = DataLoader(test, batch_size=ns.batch_size, shuffle=True)
    
    #input and output dimensions 
    input_size = lenP + lenB
    output_size = lenP
    n_out = ns.n_out

    #preparing model
    if ns.moments:
        print("Moments model")
        model = NINf.NoInterNet_fraction_compress_inference(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)
    else:
        model = NINf.NoInterNet_fraction_compress(input_size, ns.neurons, output_size, n_out, ns.n_min).to(device)
    
    loss_fn = NINf.LogFractionLoss() if ns.logLoss else nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=ns.learning_rate)
    
    #training
    if ns.moments:
        train_history, val_history = NINf.training_inference(train_dataloader, val_dataloader, model, optimizer, ns.epochs, ns.patience, ns.output, lenP, ns.neurons)
    else:
        train_history, val_history = NIN.training(train_dataloader, val_dataloader, model, loss_fn, optimizer, ns.epochs, ns.patience, ns.output, ns.neurons)
    
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

    if ns. opt:
        return val_history
    
if __name__ == '__main__':
    desc = 'Train a dense NN that takes as inputs the interloper-contaminated P+B and outputs the Pk correction and the outlier fraction'
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'input to csv file containg the path to the Pk (first column contaminated, second column not contaminated)'
    group.add_argument('--input', type=str, help=h, required=True)

    h = 'fractions file'
    group.add_argument('--fraction-path', type=str, required=True)

    h = 'output folder path'
    group.add_argument('--output', type=str, help=h, required=True)    

    #optional arguments

    h = 'maximum k in the Pk'
    parser.add_argument('--k-max', type=float, default=.8, help=h)

    h = 'minimum k in the Pk'
    parser.add_argument('--k-min', type=float, default=0, help=h)

    h = 'fraction of P(k)s used for training'
    parser.add_argument('--train-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.75, help=h)
    
    h = 'fraction of P(k)s used for validation (must be greater than tranin-fraction) '
    parser.add_argument('--val-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.90, help=h)

    h = 'number of neurons of hidden layers, or of the first hidden layers'
    parser.add_argument('--neurons', type=int, default=60, help=h)
    
    h = 'minimum number of neurons'
    parser.add_argument('--n-min', type=int, default=8, help=h)

    h = 'number of neurons for the last hidden layer'
    parser.add_argument('--n-out', type=int, default=64, help=h)
    
    h = 'learning rate'
    parser.add_argument('--learning-rate', type=float, default=1e-3, help=h)
    
    h = 'batch size'
    parser.add_argument('--batch-size', type=int, default=64, help=h)

    h = 'maximum number of epochs for training'
    parser.add_argument('--epochs', type=int, default=400, help=h)
    
    h = 'patience for earlystopping'
    parser.add_argument('--patience', type=int, default=20, help=h)

    #flags

    h = 'input log10(Pk)'
    parser.add_argument('--log', action='store_true', help=h)

    h = 'use LogMSELoss?'
    parser.add_argument('--logLoss', action='store_true', help=h)

    h = 'hyper-parameter optimization flag'
    parser.add_argument('--opt', action='store_true', help=h)

    h = 'rescale f in [0,1]'
    parser.add_argument('--norm-fs', action='store_true', help=h)

    h = 'rescale correction in [0,1]'
    parser.add_argument('--norm-c', action='store_true', help=h)

    h = 'moment network'
    parser.add_argument('--moments', action='store_true', help=h)

    # and go!
    main(parser.parse_args())
