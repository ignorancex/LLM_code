import os 
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import pandas as pd

from models.supervised import BindingAffinityPredictor
#from pdbbind_dataset import collate_fn_with_label
from dataset.davis_and_kiba_data import DavisDataset, collate_fn_with_label
from torch.utils.data import DataLoader
from utils.train_test import train_one_epoch_davis_and_kiba, eval_one_epoch_davis_and_kiba
from utils.metrics import evaluate_reg
import datetime
import json

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type = int, default = 12345)
parser.add_argument("--num_epoch", type = int, default = 200)
parser.add_argument("--lr", type = float, default = 5e-3)
parser.add_argument("--hidden_dim", type = int, default = 256)
parser.add_argument("--data_root", type = str, default = "./data/kiba")
parser.add_argument("--fold_idx", type = int, default = 0)
args  = parser.parse_args()


# set seed 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

exp_name = f"kiba_{args.fold_idx}"
result_dict = {}
result_dict.update(vars(args))
device = "cuda"
model = BindingAffinityPredictor(hidden_dim = args.hidden_dim).to(device)
loss_fn = nn.MSELoss()
evaluator = evaluate_reg
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5)

########
csv_file = os.path.join(args.data_root, "full.csv")
df = pd.read_csv(csv_file)

test_fold = json.load(open(f"{args.data_root}/folds/test_fold_setting1.txt"))
folds = json.load(open(f"{args.data_root}/folds/train_fold_setting1.txt"))

val_fold = folds[args.fold_idx]
df_train = df[~ df.index.isin(test_fold)]
df_val = df_train[df_train.index.isin(val_fold)]
df_train = df_train[~ df_train.index.isin(val_fold)]
df_test = df[df.index.isin(test_fold)]

train_dataset = DavisDataset(args.data_root, df_train)
valid_dataset = DavisDataset(args.data_root, df_val)
test_dataset = DavisDataset(args.data_root, df_test)


#######
train_loader = DataLoader(train_dataset, batch_size = 32, collate_fn = collate_fn_with_label, shuffle = True, num_workers = 6)
valid_loader = DataLoader(valid_dataset, batch_size = 32, collate_fn = collate_fn_with_label, shuffle = False, num_workers = 6)
test_loader = DataLoader(test_dataset, batch_size = 32, collate_fn = collate_fn_with_label, shuffle = False, num_workers = 6)
    

best_val_loss = 1000

for epoch in range(args.num_epoch):
    train_loss = train_one_epoch_davis_and_kiba(model, train_loader, optimizer, loss_fn, device)
    valid_score = eval_one_epoch_davis_and_kiba(model, valid_loader, device, evaluator, loss_fn)
    test_score = eval_one_epoch_davis_and_kiba(model, test_loader, device, evaluator)
    
    print(f"Train Loss: {train_loss}")

    for key in valid_score.keys():
        if key == 'avg_loss':
            val_loss = valid_score['mse']
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss= val_loss
                result_dict['train_loss'] = train_loss
                result_dict['val_loss'] = val_loss
                result_dict['epoch'] = epoch
                result_dict['best_val_loss'] = best_val_loss
                for key in test_score.keys():
                    result_dict[f"valid_{key}"] = valid_score[key]
                    result_dict[f"test_{key}"] = test_score[key]
                json_object = json.dumps(result_dict, indent = 4)
                with open(f"log_dir/{exp_name}.json", "w") as f:
                    f.write(json_object)
        else:
            print(f"\n{key} | Validation: {valid_score[key]} | Test: {test_score[key]}") 
        

    print(f"---- Done Epoch {epoch} ----- \n")