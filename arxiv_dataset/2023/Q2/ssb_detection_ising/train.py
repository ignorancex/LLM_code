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
parser.add_argument('--Ls','-ls', type=lambda s: [int(i) for i in s.split(",") if i != ""], default="", help='Multiple lattice sizes for training (comma separated).')
parser.add_argument('--L_test','-lt', type=int, default=0, help='Lattice size for testing; default is set equal to --L.')
# network architecture
parser.add_argument('--encoder_hidden', '-eh', default=4, type=int, help='Hidden neurons in the encoder.')
parser.add_argument('--decoder_hidden', '-dh', default=64, type=int, help='Hidden neurons in the decoder.')
parser.add_argument('--latent_dim', '-ld', default=1, type=int, help='Latent dimension of the autoencoder.')
parser.add_argument('--symmetric', '-s', action="store_true", help='Enforce symmetries resulting from latent dimension.')
parser.add_argument('--seed', '-sd', default=0, type=int, help='Pytorch RNG seed.')
# SGD hyperparameters
parser.add_argument('--batch_size', '-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs for training.')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='Learning rate.')
# equivariance regularization
parser.add_argument('--equivariance_reg', '-er', default=0, type=float, help='Coefficient of equivariance-enforcing terms in loss function.')
parser.add_argument('--equivariance_pre', '-ep', default=0, type=int, help='Number of training epochs ignoring --equivariance_reg.')
# validation
parser.add_argument('--fold', '-f', default=0, type=int, help='Index of training-validation fold.')
parser.add_argument('--n_train_val', '-n', default=256, type=int, help='Number of training + validation samples.')
parser.add_argument('--n_test', '-nt', default=2048, type=int, help='Number of test samples.')
parser.add_argument('--val_size','-vs', default=0.5, type=float, help='Fraction of data to use as validation set.')
parser.add_argument('--val_batch_size','-vbs', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
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

# if multiple lattice sizes
n_Ls = len(args.Ls)
if n_Ls > 0:
	assert args.symmetric, "--Ls may be set only if --symmetric is also set."
	assert args.n_train_val%n_Ls == 0, "--n_train_val must be a multiple of len(--Ls)."
	assert args.n_train_val//n_Ls >= 2, "--n_train_val must be at least twice len(--Ls)."

# set lattice size for testing
if args.L_test == 0:
	assert n_Ls == 0, "If using multiple lattice sizes with --Ls, then --L_test must be set explicitly."
	args.L_test = args.L
if args.L_test != args.L:
	assert args.symmetric, "--L and --L_test can be different only if --symmetric is set."
	results_dir = os.path.join(args.results_dir, "L{:d}".format(args.L_test))
	os.makedirs(results_dir, exist_ok=True)

# make no training data equivalent to no training
if args.n_train_val == 0:
	args.n_train_val = 2
	args.epochs = 0

# temperatures
temperatures = np.array([1.04+0.04*i for i in range(25)] \
	+ [2.01+0.01*i for i in range(50)] \
	+ [2.54+0.04*i for i in range(25)], dtype=np.float32)

# load data
if args.symmetric:
	assert args.latent_dim <= 2, "Symmetric architecture is supported only for latent dimension 1 or 2."
	data_filename = "states_symmetric.npz" if args.latent_dim==1 else "states_symmetric_2D.npz"
else:
	data_filename = "states.npz"
if n_Ls > 0:
	X = []
	for L in args.Ls:
		with np.load(os.path.join(args.data_dir, "L{:d}".format(L), "aggregate", data_filename), mmap_mode="r") as states_L:
			X.append( states_L["train"][:,256*args.fold:256*args.fold+args.n_train_val//n_Ls].reshape((-1, states_L["train"].shape[2])) )
	X = np.concatenate(X, 0)
else:
	with np.load(os.path.join(args.data_dir, "L{:d}".format(args.L), "aggregate", data_filename), mmap_mode="r") as states_L:
		X = states_L["train"][:,256*args.fold:256*args.fold+args.n_train_val].reshape((-1, states_L["train"].shape[2]))
T = np.repeat(temperatures, args.n_train_val)[:,None]
if n_Ls > 0:
	stratifier = np.concatenate((T, \
		np.tile(np.repeat(np.array(args.Ls), args.n_train_val//n_Ls), len(T)//args.n_train_val)[:,None] \
	), 1)
else:
	stratifier = np.copy(T)
X_train, X_val, T_train, T_val = train_test_split(X, T, stratify=stratifier, test_size=args.val_size, random_state=0)
del X; del T; del stratifier; gc.collect()
train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train), torch.as_tensor(T_train)), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(TensorDataset(torch.as_tensor(X_val), torch.as_tensor(T_val)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)

# build model
input_dim = 2*args.latent_dim if args.symmetric else args.L**2
encoder = pf.models.MLP(input_dim, args.encoder_hidden, args.latent_dim)
decoder = pf.models.MLP(args.latent_dim+1, args.decoder_hidden, input_dim)

# initialize parameters
input_dim = 2*args.latent_dim
torch.manual_seed(args.seed)
torch.nn.init.uniform_(encoder.linear1.bias, -1/np.sqrt(input_dim), 1/np.sqrt(input_dim))
torch.nn.init.uniform_(encoder.linear2.weight, -1/np.sqrt(args.encoder_hidden), 1/np.sqrt(args.encoder_hidden))
torch.nn.init.uniform_(encoder.linear2.bias, -1/np.sqrt(args.encoder_hidden), 1/np.sqrt(args.encoder_hidden))
torch.nn.init.uniform_(decoder.linear1.weight, -1/np.sqrt(args.latent_dim+1), 1/np.sqrt(args.latent_dim+1))
torch.nn.init.uniform_(decoder.linear1.bias, -1/np.sqrt(args.latent_dim+1), 1/np.sqrt(args.latent_dim+1))
encoder_weight = 2/np.sqrt(input_dim)*torch.rand(args.encoder_hidden, input_dim) - 1/np.sqrt(input_dim)
decoder_weight = 2/np.sqrt(args.decoder_hidden)*torch.rand(input_dim, args.decoder_hidden) - 1/np.sqrt(args.decoder_hidden)
decoder_bias = 2/np.sqrt(args.decoder_hidden)*torch.rand(input_dim) - 1/np.sqrt(args.decoder_hidden)
if not args.symmetric:
	if args.latent_dim == 1:
		def checkerboardify(x, L):
			y = torch.tile(x, [1, L//2])
			return torch.tile(torch.cat([y, torch.flip(y, [1])], 1), [1, L//2])
	if args.latent_dim == 2:
		def checkerboardify(x, L):
			y = torch.cat([torch.tile(x[:,:2], [1, L//2]), torch.tile(x[:,2:], [1, L//2])], 1)
			return torch.tile(y, [1, L//2])
	encoder_weight = input_dim/args.L**2*checkerboardify(encoder_weight, args.L)
	decoder_weight = torch.t( checkerboardify(torch.t(decoder_weight), args.L) )
	decoder_bias = checkerboardify(decoder_bias.unsqueeze(0), args.L).squeeze(0)
encoder.linear1.weight.data.copy_(encoder_weight)
decoder.linear2.weight.data.copy_(decoder_weight)
decoder.linear2.bias.data.copy_(decoder_bias)
del encoder_weight; del decoder_weight; del decoder_bias; gc.collect()
torch.manual_seed(0)

# create trainer and callbacks
trainer = pf.trainers.Autoencoder(encoder, decoder, epochs=args.epochs, lr=args.lr, rescale_lr=not args.symmetric, equivariance_reg=args.equivariance_reg, equivariance_pre=args.equivariance_pre, device=args.device)
callbacks = [pf.callbacks.Training()]
if args.val_interval > 0:
	callbacks.append( pf.callbacks.Validation(trainer, val_loader, epoch_interval=args.val_interval) )

# load pretrained parameters
if len(args.load_model) > 0:
	trainer.load_encoder(os.path.join(args.load_model, "encoder.pth"))
	trainer.load_decoder(os.path.join(args.load_model, "decoder.pth"))

# train model
trainer.fit(train_loader, callbacks)

# estimate symmetry generator reps
if args.equivariance_reg > 0:
	train_loader = DataLoader(train_loader.dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	if args.latent_dim == 1:
		flip_rep, neg_rep = trainer.generator_reps(train_loader)
		generator_reps = {"spatial": float(flip_rep), "internal": float(neg_rep)}
		with open(os.path.join(results_dir, "generator_reps.json"), "w") as fp:
			json.dump(generator_reps, fp, indent=2)
	if args.latent_dim == 2:
		reps = trainer.generator_reps_2D(train_loader)
		np.savez(os.path.join(results_dir, "generator_reps.npz"), **reps)

# generate encodings
del train_loader; del val_loader; gc.collect()
if args.latent_dim == 1:
	with np.load(os.path.join(args.data_dir, "L{:d}".format(args.L_test), "aggregate", data_filename), mmap_mode="r") as states_L:
		X = states_L["test"][:,:args.n_test].reshape((-1, states_L["test"].shape[2]))
	T = np.repeat(temperatures, args.n_test)[:,None]
	test_loader = DataLoader(TensorDataset(torch.as_tensor(X), torch.as_tensor(T)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	encodings = trainer.encode(test_loader)
	assert encodings.shape[1] == 1, "This script currently only supports 1D encodings."
	measurements = encodings.squeeze(-1).reshape((-1, args.n_test))
	np.savez(os.path.join(results_dir, "measurements.npz"), temperatures=temperatures, measurements=measurements)

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

# collect results
results_dict = {
#	"data_shapes": {name: list(A.shape) for (name, A) in [("X_1", X_1), ("X_2", X_2)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
}

# add command-line arguments and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(results_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
trainer.save_encoder(os.path.join(results_dir, "encoder.pth"))
trainer.save_decoder(os.path.join(results_dir, "decoder.pth"))


print("Done!")
