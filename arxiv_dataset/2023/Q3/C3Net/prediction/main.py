import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.append(_parent)
del _here, _parent

import torch

import module
import prediction.predictor as predictor
from module.data import *
from module.input.input_generator import generator_predict
from module.nn import *
from module.nn import Atomistic_Model, Atomwise

# Property to predict: "water" (for solvation free energy) | "logp" | "pampa"
solvent = "water"
# Input molecule .sdf to predict:
solute = "example/MAN_ideal.sdf"
# Directory to store temporary .npz files:
npz_dir = "npz"
# Directory to write prediction result:
output_dir = "output/prediction"
# ====================================================================

os.makedirs(npz_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

generator_predict(solvent, solute, npz_dir)
# 1=single conformer, 5=multiconformer
n_conformer = 1
data = MoleculeSet(npz_dir, output_dir, n_conformer)
loader = MoleculeLoader(data, batch_size=2, num_workers=0)

# create model
reps = C3Net()
output = Atomwise(n_in=64)
model = Atomistic_Model(reps, output)

device = "cpu"
predictor = predictor.Predictor(output_dir, model, loader, device)

# start training
predictor.run(torch.device(device, 0))
data.clean_npzs()
