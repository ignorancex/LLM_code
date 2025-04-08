import sys, os, time, glob
dir = os.getcwd()
dir = os.path.dirname(dir)
sys.path.append(dir)


from module.nn import Atomistic_Model, Atomwise
import torch
import torch.nn.functional as F
from torch.optim import Adam

import module
from module.nn import *
from module.data import *
from module.input.input_generator import generator_predict
import train.trainer as trainer

#1=single conformer, 5=multiconformer
n_conformer = 1
input_path = '../dataset/training_set/npz'
checkpoint_path = './output/training'
#generator_predict(input_path)


data = MoleculeSet(input_path, checkpoint_path, n_conformer)

# split: train 80% and val 20% 
train, val = split.split_train_val(data, 0.8) 
print (len(train.subset),len(val.subset))
#batch size = 2
train_loader = MoleculeLoader(train, batch_size=2, num_workers=0)
val_loader = MoleculeLoader(val)
print (len(train_loader))
print (len(val_loader))
#create model
reps = C3Net()
output = Atomwise(n_in=64)
model = Atomistic_Model(reps, output)
#create trainer
opt = Adam(model.parameters(), lr=1e-4)
loss = lambda b, p: F.mse_loss(p["y"], b['dG'])
device = 'cpu'
trainer = trainer.Trainer(checkpoint_path, model, loss, opt, train_loader, val_loader, device)

# start training
trainer.train(torch.device(device, 0))
