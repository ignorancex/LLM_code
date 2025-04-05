import gc
import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import phasefinder as pf

# command-line arguments
parser=argparse.ArgumentParser()
# datasets
parser.add_argument('--data_dir', '-d', required=True, type=str, help='Master directory of the data.')
parser.add_argument('--L','-l', required=True, type=int, help='Linear size of lattice for training.')
# network architecture
parser.add_argument('--encoder_hidden', '-eh', default=4, type=int, help='Hidden neurons in the encoder.')
parser.add_argument('--seed', '-sd', default=0, type=int, help='Pytorch RNG seed.')
# validation
parser.add_argument('--fold', '-f', default=0, type=int, help='Index of training-validation fold.')
parser.add_argument('--n_train_val', '-n', default=256, type=int, help='Number of training + validation samples.')
parser.add_argument('--val_size','-vs', default=0.5, type=float, help='Fraction of data to use as validation set.')
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
parser.add_argument('--load_model', '-lm', default="", type=str, help='Load pretrained encoder and decoder from given directory.')
parser.add_argument('--results_dir', '-rd', required=True, type=str, help='Master results directory.')
parser.add_argument('--exist_ok', '-ok', action="store_true", help='Allow overwriting the results_dir directory.')
args=parser.parse_args()


# fix the random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# record initial time
time_start = time.time()

# create results directory
results_dir = args.results_dir
if os.path.exists(results_dir):
	import sys; sys.exit(0)
os.makedirs(results_dir, exist_ok=args.exist_ok)

# temperatures
temperatures = np.array([1.04+0.04*i for i in range(25)] \
	+ [2.01+0.01*i for i in range(50)] \
	+ [2.54+0.04*i for i in range(25)], dtype=np.float32)


# load data
print("Loading data . . . ")
data_filename = "states.npz"
with np.load(os.path.join(args.data_dir, "L{:d}".format(args.L), "aggregate", data_filename), mmap_mode="r") as states_L:
	X = states_L["train"][:,256*args.fold:256*args.fold+args.n_train_val].reshape((-1, states_L["train"].shape[2]))
T = np.repeat(temperatures, args.n_train_val)[:,None]
stratifier = np.copy(T)
X_train, _, T_train, _ = train_test_split(X, T, stratify=stratifier, test_size=args.val_size, random_state=0)
del X; del T; del stratifier; gc.collect()
train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train), torch.as_tensor(T_train)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)


# build model
print("Building model . . . ")
input_dim = args.L**2
encoder = pf.models.MLP(input_dim, args.encoder_hidden, 1)

# load pretrained parameters
encoder.load_state_dict(torch.load( os.path.join(args.load_model, "encoder.pth") ))


# function to evaluate encoder
def encode(transform=None):
	encodings = []
	with torch.no_grad():
		for (X_batch, _) in train_loader:
			X_batch = X_batch.to(args.device)
			if transform is not None:
				X_batch = transform(X_batch)
			encodings_batch = encoder(X_batch)
			encodings.append( encodings_batch.cpu().numpy() )
	encodings = np.concatenate(encodings, 0).reshape((-1))
	return encodings

# transforms
flatten = lambda x: x.reshape(-1, args.L**2)
unflatten = lambda x: x.reshape(-1, args.L, args.L)
transforms = {
	"alpha": lambda x: torch.roll(x, args.L, dims=1), 
	"rho": lambda x: flatten(torch.rot90(unflatten(x), dims=(1, 2))), 
	"tau": lambda x: flatten(torch.flip(unflatten(x), [2])), 
	"sigma": lambda x: -x
}


# measure equivariance
print("Measuring equivariance . . . ")
Z = encode(transform=None)
norm = np.linalg.norm(Z)
Zs_transformed = {gen: encode(transform=transform) for (gen, transform) in transforms.items()}
norms_transformed = {gen: np.linalg.norm(z) for (gen, z) in Zs_transformed.items()}
norm_ratios = {gen: float( n/norm ) for (gen, n) in norms_transformed.items()}
cos_sims = {gen: float( Z.dot(Zs_transformed[gen])/(norm*norms_transformed[gen]) ) for gen in Zs_transformed.keys()}
with open(os.path.join(results_dir, "norm_ratios.json"), "w") as fp:
	json.dump(norm_ratios, fp, indent=2)
with open(os.path.join(results_dir, "cos_sims.json"), "w") as fp:
	json.dump(cos_sims, fp, indent=2)

# collect results
results_dict = {}
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(results_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)


print("Done!")
