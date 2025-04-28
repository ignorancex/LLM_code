import torch.multiprocessing as mp
import numpy as np
import argparse
import pickle
import pandas as pd
import torch
from gen_data import simulate_dag, simulate_lsem
from isl import isl
from utils import add_int
from sim_settings import settings

#========================================
# Configurations
#========================================

parser = argparse.ArgumentParser()

# ----------- Data parameters ------------
parser.add_argument('--infile', type = str, default = '',
                    help = 'File containing data to use ISL on.')
parser.add_argument('--scenario', type = str, default = 'S1',
                    help = 'Simulation scenario (choice of S1-S6).')
parser.add_argument('--p',type=int,default=2,
                    help='number of potential moderators')
parser.add_argument('--s',type=int,default=6,
                    help='number of potential mediators')
parser.add_argument('--sample_size', type = int, default = 50,
                    help = 'The number of samples of data, ignored if infile set.')
parser.add_argument('--num_cores',type=int,default=1,help="number of cores to use")
parser.add_argument('--outfile',type=str,default='',
                    help='Output file name')
parser.add_argument('--dir',type=str,default='.',
                    help='Output folder to save to')
parser.add_argument('--dgnn', action='store_true',
                   help="Disables use of structural constraints")
parser.add_argument('--XM', action='store_true')

# ----------- Training hyperparameters ------
parser.add_argument('--start', type = int, default = 0, help = 'Start replication in case of restart')
parser.add_argument('--rep_number', type = int, default = 1,
                    help = 'Number of replications, ignored if end set.')
parser.add_argument('--end', type = int, default = 0,
                    help = 'End replication in case of restarts.')
parser.add_argument('--epochs', type = int, default = 200,
                    help ='Number of epochs to train.')
parser.add_argument('--batch_size', type = int, default = 128,
                    help = 'Number of samples per batch.')
parser.add_argument('--k_max_iter', type = int, default = 1e2,
                    help = 'the max iteration number for searching parameters')
parser.add_argument('--original_lr', type = float, default = 3e-3, help = 'Initial learning rate.')

args = parser.parse_args()

# load real data if applicable
if args.infile != '':
    X = pd.read_csv(args.infile).to_numpy()
    # add in XA interaction
    X = add_int(X, args.p, args.XM)[:,:,np.newaxis]
    s = args.s
    p = args.p
# else load simulation graph
else:
    G = simulate_dag(**settings[args.scenario],XM=args.XM)[0]
    s = settings[args.scenario]["s"]
    p = settings[args.scenario]["p"]

outfile = args.outfile
if outfile == '':
    if args.infile != '':
        outfile = args.infile.split('.')[0]
    else:
        outfile = '%s_%d' %(args.scenario + (('_'+str(args.XM+1)) if args.XM else ''),args.sample_size)

if args.infile == '':
    def Worker(seed):
        torch.set_num_threads(1)
        with open('%s/%s_%d.data' %(args.dir,outfile,seed), 'wb') as filehandle:
            pickle.dump(isl(simulate_lsem(G,args.sample_size,p,s=s,seed=seed),
                            s,p,args.epochs,args.batch_size, args.k_max_iter,
                            args.original_lr,seed=seed,no_constraints=args.dgnn, inter_XM=args.XM)[0],filehandle,
                        protocol=pickle.HIGHEST_PROTOCOL)
else:
    def Worker(seed):
        torch.set_num_threads(1)
        with open('%s/%s_%d.data' %(args.dir,outfile,seed), 'wb') as filehandle:
            pickle.dump(isl(X,s,p,args.epochs,args.batch_size, args.k_max_iter,
                            args.original_lr,no_constraints=args.dgnn,seed=seed, inter_XM=args.XM)[0],filehandle,
                        protocol=pickle.HIGHEST_PROTOCOL)

seeds = list(range(args.start, args.end+1 if args.end else args.start+args.rep_number))

if __name__ == '__main__':
    mp.set_start_method("spawn",force=True)
    
    if args.num_cores == 1:
        for seed in seeds:
            Worker(seed)
    else:
        with mp.Pool(args.num_cores) as pool:
            pool.map(Worker, seeds)