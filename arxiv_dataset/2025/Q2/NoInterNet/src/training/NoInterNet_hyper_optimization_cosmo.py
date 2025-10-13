import argparse
import os

import optuna
import torch

from NoInterNet_cosmologyprior_training import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_terminal():
    desc = 'Hyper-parameter optimization of NoInterNet_fraction_compress'
    parser = argparse.ArgumentParser(description=desc)

    # required arguments
    group = parser.add_argument_group('required arguments')

    h = 'input to csv file containg the path to the Pk (first column contaminated, second column not contaminated)'
    group.add_argument('--input', type=str, help=h, required=True)

    h = 'fractions file'
    group.add_argument('--fraction-path', type=str, required=True)

    h = 'output folder path'
    group.add_argument('--output-hyp', type=str, help=h, required=True)

    h = 'study storage'
    group.add_argument('--storage', type=str, required=True, help=h)

    #optional arguments

    h = 'maximum k in the Pk'
    parser.add_argument('--k-max', type=float, default=.8, help=h)

    h = 'minimum k in the Pk'
    parser.add_argument('--k-min', type=float, default=0, help=h)

    h = 'fraction of P(k)s used for training'
    parser.add_argument('--train-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.75, help=h)
    
    h = 'fraction of P(k)s used for validation (must be greater than tranin-fraction) '
    parser.add_argument('--val-fraction', type=float, choices=NIN.Range(0.0, 1.0), default=0.90, help=h)

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

    h = 'cosmological parameters to sample'
    parser.add_argument('--priors', type=str, nargs='+', default=['sigma_8'], help=h)
    
    #flags

    h = 'input log10(Pk)'
    parser.add_argument('--log', action='store_true', help=h)

    h = 'use LogMSELoss?'
    parser.add_argument('--logLoss', action='store_true', help=h)

    h = 'add log10(normalization) to input variables'
    parser.add_argument('--norm-feature', action='store_true', help=h)

    h = 'rescale f in [0,1]'
    parser.add_argument('--norm-fs', action='store_true', help=h)

    h = 'rescale correction in [0,1]'
    parser.add_argument('--norm-c', action='store_true', help=h)

    h = 'moment network'
    parser.add_argument('--moments', action='store_true', help=h)

    h = 'use B0 as well'
    parser.add_argument('--PB', action='store_true', help=h)
    
    return parser.parse_args()

def objective(trial):

    n_min = trial.suggest_categorical("n_min", [2, 4, 8, 16, 32, 64, 128])
    neurons = trial.suggest_categorical("neurons", [32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])

    log_name = '-log' if params.log else '-nolog'
    loss_name = '-Logloss' if params.logLoss else ''
    output_folder = os.path.join(params.output_hyp, 'hyper-opt', f'NoInterNet{log_name}-{params.k_max:.1f}{loss_name}-{n_min:d}-{neurons:d}-{learning_rate:1.2e}-b{batch_size:d}/')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    params.output = output_folder

    params.n_min = n_min
    params.neurons = neurons
    params.learning_rate = learning_rate
    params.batch_size = batch_size
    params.opt = True

    min_loss = min(main(params))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return min_loss

if __name__ == '__main__':
    params = read_terminal()

    # Optuna parameters
    study_name = "hyper-opt" + params.storage
    n_trials   = 20
    source = "sqlite:///hyper-opt-" + params.storage + ".db"

    # Define sampler and start optimization
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=source, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)
    
    
    trial = study.best_trial
    
    log_name = '-log' if params.log else '-nolog'
    loss_name = '-Logloss' if params.logLoss else ''
    output_folder = os.path.join(params.output_hyp, 'hyper-opt', f'NoInterNet{log_name}-{params.k_max:.1f}{loss_name}-{trial.params["n_min"]:d}-{trial.params["neurons"]:d}-{trial.params["learning_rate"]:1.2e}-b{trial.params["batch_size"]:d}/')

    out = os.path.join(params.output_hyp, 'hyper-opt', 'best-trial-params.txt')
    
    with open(out, 'w') as f:
        print("Best trial:", file=f)
        print("  Value: ", trial.value, file=f)
        print("  Params: ", file=f)
        for key, value in trial.params.items():
            print(f"    {key}: {value}", file=f)
    
    best_name = os.path.join(params.output_hyp, 'hyper-opt', f'best-NoInterNet{log_name}-{params.k_max:.1f}{loss_name}/')

    if os.path.exists(output_folder):
        cmd = "cp -r " + output_folder + " " + best_name
        print(cmd)
        os.system(cmd)