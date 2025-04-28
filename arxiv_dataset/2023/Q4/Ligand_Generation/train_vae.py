import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from tdc.chem_utils import MolConvert
from models.pl_conditional_model import ThreeD_Conditional_VAE
from dataset.pdbbind_dataset import PDBBindDataset, collate_fn
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 12345)
parser.add_argument("--num_epoch", type = int, default = 30)
parser.add_argument("--lr", type = float, default = 5e-4)
parser.add_argument("--latent_dim", type = int, default = 128)
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--ckpt_path", type = str, default = "checkpoint/vae.pt")
parser.add_argument("--device","--list", nargs="+")
args  = parser.parse_args()


device = [int(x) for x in args.device]

# set seed 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

inverter = MolConvert(src='SELFIES', dst = 'SMILES')
model = ThreeD_Conditional_VAE(
    latent_dim = args.latent_dim
)
print("Start training")
train_dataset = PDBBindDataset("train")
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
trainer = pl.Trainer(accelerator="gpu", devices=device, max_epochs=args.num_epoch, logger=pl.loggers.CSVLogger('logs'),
                     enable_checkpointing=False, strategy='ddp_find_unused_parameters_true')
trainer.fit(model, train_loader)
print('Saving..')
torch.save(model.state_dict(), args.ckpt_path)