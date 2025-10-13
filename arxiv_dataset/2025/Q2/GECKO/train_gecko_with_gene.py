# --> General imports
import os
import numpy as np
from tqdm import tqdm
import json 
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random
import pickle
import argparse
import pandas as pd
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR

from contrastive_loss import InfoNCE, InfoNCE_with_false_negative_elimination
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(SEED, disable_cudnn=False):
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)        # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True  
    else:
        torch.backends.cudnn.enabled = False 


class GeckoDataset(Dataset):
    def __init__(self, args, features_deep_path, features_path, gene_expression_path, dataset_dict_path, max_n_tokens=None):

        with open(dataset_dict_path, 'rb') as f:
            self.dataset_dict = pickle.load(f)

        empty_keys_list = []
        for i in self.dataset_dict:
            if self.dataset_dict[i][1]-self.dataset_dict[i][0] == 0:
                empty_keys_list.append(i)
            
        print('empty_keys_list: ', empty_keys_list)

        for i in empty_keys_list:
            del self.dataset_dict[i]


        with open(gene_expression_path, 'rb') as f:
            self.wsi_gene_expression_dict = pickle.load(f)

        self.max_n_tokens = 0
        for i in self.dataset_dict.keys():
            if self.max_n_tokens < self.dataset_dict[i][1] - self.dataset_dict[i][0]:
                self.max_n_tokens = self.dataset_dict[i][1] - self.dataset_dict[i][0]
        
        print('max number of tokens', self.max_n_tokens)

        self.max_n_tokens = min(max_n_tokens, self.max_n_tokens)

        print('max number of tokens set to minimum of (dataset, fixed value of 2048)', self.max_n_tokens)

        self.features_deep_array = np.array(torch.load(features_deep_path).cpu().detach())
        print(self.features_deep_array.shape)
        
        self.features_array = pd.read_csv(features_path)
        self.features_array = self.features_array.set_index('Unnamed: 0')
        print(self.features_array.shape)
        
        self.feats_size = self.features_array.shape[1]
        self.feats_size_deep = self.features_deep_array.shape[1]

        self.bags_path = list(self.dataset_dict.keys())
        print(len(self.bags_path))

        # only select training WSIs with matched gene data for pretraining. 
        self.bags_path = [x for x in self.bags_path if x in list(self.wsi_gene_expression_dict.keys())]
        print(len(self.bags_path))

        # from ConcepPath paper
        train_path = []

        fold_split = np.array(pd.read_csv(args.split_path + '/fold' + str(args.cross_val_fold) + '.csv').iloc[:,1:])

        for i in self.bags_path:
            for j in fold_split:
                if i in j[0]:
                    if j[-1] == 'train':
                        train_path.append(i)
                    break

        self.bags_path = train_path

        train_index_select_list = []
        for i in train_path:  # from train and val both
            train_index_select_list.extend(list(np.arange(self.dataset_dict[i][0], self.dataset_dict[i][1])))

        self.features_array = np.array(self.features_array)

        for i in range(self.features_array.shape[1]):
            self.features_array[:, i] = (self.features_array[:, i] - self.features_array[train_index_select_list, i].min()) / (self.features_array[train_index_select_list, i].max() - self.features_array[train_index_select_list, i].min() + 1e-6)


    def __len__(self):
        return len(self.bags_path)

    def __getitem__(self, idx):
        split_info = self.dataset_dict[self.bags_path[idx]]
        wsi_gene_expression = torch.tensor(self.wsi_gene_expression_dict[self.bags_path[idx]], dtype=torch.float32).to(DEVICE)

        feats_deep = self.features_deep_array[split_info[0]:split_info[1]].copy()
        feats = self.features_array[split_info[0]:split_info[1]].copy()	
    
        bag_feats_deep = torch.tensor(np.array(feats_deep), dtype=torch.float32)
        bag_feats_deep = bag_feats_deep.view(-1, self.feats_size_deep)
        
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        bag_feats = bag_feats.view(-1, self.feats_size)
                                    
        if bag_feats.shape[0]>self.max_n_tokens:
            patch_indices = np.random.choice(np.arange(bag_feats.shape[0]), self.max_n_tokens, replace=False)
        elif bag_feats.shape[0]==self.max_n_tokens:
            patch_indices = np.arange(bag_feats.shape[0])
        else:
            patch_indices = np.concatenate([np.arange(bag_feats.shape[0]), np.random.choice(np.arange(bag_feats.shape[0]), self.max_n_tokens-bag_feats.shape[0])])

        bag_feats_deep = bag_feats_deep[patch_indices].to(DEVICE)
        bag_feats = bag_feats[patch_indices].to(DEVICE)

        return bag_feats_deep, bag_feats, wsi_gene_expression


def train_loop(args, loss_fn_interMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, max_n_tokens):
        
    ssl_model.train()

    ep_loss = 0.
    
    for b_idx, (patch_emb_deep, patch_emb, rna_seq) in enumerate(dataloader):
        
        losses = []    

        random_select = int(np.random.choice(np.arange(args.min_n_tokens, max_n_tokens), 1)[0])

        wsi_emb, _, wsi_emb_deep_proj, wsi_emb_deep, rna_emb_proj, rna_emb, _, _ = ssl_model(patch_emb[:, :random_select], patch_emb_deep[:, :random_select], rna_seq, training='yes')
        
        losses.append(loss_fn_interMod(query=wsi_emb_deep_proj, positive_key=wsi_emb, symmetric=args.symmetric_cl))
        losses.append(loss_fn_interMod(query=rna_emb_proj, positive_key=wsi_emb, symmetric=args.symmetric_cl))
        losses.append(loss_fn_interMod(query=wsi_emb_deep, positive_key=rna_emb, symmetric=args.symmetric_cl))

        loss = sum(losses)/3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch <= args.warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")

        ep_loss += loss.item()
        
    return ep_loss





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configurations for GECKO pretraining')
    #----> training args
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs. 5')
    parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature_infonce', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size 64')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    parser.add_argument('--num_workers', type=int, default=10, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')    

    # -------- to check setting this yes
    parser.add_argument('--top_k', default=10, type=int, help='')

    parser.add_argument('--dataset_dict_path', default='', type=str, help='Dataset folder name')
    parser.add_argument('--gene_expression_path', default='', type=str, help='Dataset folder name')

    parser.add_argument('--features_path', default='', type=str, help='Dataset folder name')
    parser.add_argument('--features_deep_path', default='', type=str, help='Dataset folder name')

    parser.add_argument('--min_n_tokens', default=512, type=int, help='128 default')
    parser.add_argument('--max_n_tokens', default=2048, type=int, help='')

    parser.add_argument('--save_path', default='', type=str, help='Dataset folder name')

    parser.add_argument('--cross_val_fold', default=0, type=int, help='top features to select')
    parser.add_argument('--split_path', default='', type=str, help='Dataset folder name')

    parser.add_argument('--keep_ratio', type=float, default=0.7, help='use all samples in batch for negatives')

    args = parser.parse_args()
    set_seed(args.seed)

    # paths 
    ROOT_SAVE_DIR = args.save_path + '_with_gene/' + args.split_path.split('/')[-2] + '_keepratio' + str(args.keep_ratio)

    EXP_CODE = "topk{}_mintokensize{}_maxtokensize{}_lr{}_epochs{}_bs{}_temperatureNCE{}".format(
        args.top_k,
        args.min_n_tokens,
        args.max_n_tokens,
        args.learning_rate, 
        args.epochs, 
        args.batch_size, 
        args.temperature_infonce
    )

    RESULTS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE, str(args.cross_val_fold))
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

    with open(os.path.join(RESULTS_SAVE_PATH, "config.json"), 'w') as jsonfile:
        json.dump(vars(args), jsonfile, indent=4)

    print()
    print(f"Running experiment {EXP_CODE}...")
    print()
    
    # Create a SummaryWriter
    log_dir = os.path.join(ROOT_SAVE_DIR, 'logs', EXP_CODE, str(args.cross_val_fold))
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'log.txt')),
            logging.StreamHandler(sys.stdout)  # Keep console output
        ]
    )

    # Redirect print statements to logging
    class PrintToLogger:
        def __init__(self, original_stdout):
            self.orig_stdout = original_stdout
            
        def write(self, text):
            if text.strip():  # Avoid empty lines
                logging.info(text.strip())
            self.orig_stdout.write(text)  # Keep original stdout output

        def flush(self):
            self.orig_stdout.flush()

    sys.stdout = PrintToLogger(sys.stdout)

    print("* Setup dataset...")
    dataset = GeckoDataset(
        args,
        features_deep_path=args.features_deep_path, 
        features_path=args.features_path, 
        gene_expression_path=args.gene_expression_path, 
        dataset_dict_path=args.dataset_dict_path,
        max_n_tokens=args.max_n_tokens,
    )

    # set up dataloader
    print("* Setup dataloader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
    )

    print("* Setup model...")
    import batched_simil_with_gene as mil	
    
    b_classifier = mil.BClassifier(input_size=dataset.feats_size, input_size_deep=dataset.feats_size_deep, top_k=args.top_k).to(DEVICE)
    ssl_model = mil.MILNet(b_classifier).to(DEVICE)
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(ssl_model.parameters(), lr=args.learning_rate)
    
    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args.epochs - args.warmup_epochs) * len(dataloader) if args.warmup else args.epochs * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=T_max,
        eta_min=args.end_learning_rate
    )
    
    if args.warmup:
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.00001,
            total_iters=args.warmup_epochs * len(dataloader)
    )
    else:
        scheduler_warmup = None
    
    # set up losses
    print("* Setup losses...")

    if args.keep_ratio == 1:
        print('using InfoNCE')
        loss_fn_interMod = InfoNCE(temperature=args.temperature_infonce)
    else:
        print('using InfoNCE_with_false_negative_elimination')
        loss_fn_interMod = InfoNCE_with_false_negative_elimination(temperature=args.temperature_infonce, percent=args.keep_ratio)


    # main training loop
    for epoch in range(args.epochs):
        
        print()
        print(f"Training for epoch {epoch}...")
        print()
        
        # train
        start = time.time()
        ep_loss = train_loop(args, loss_fn_interMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, dataset.max_n_tokens)
        end = time.time()

        print()
        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print("Total time = {:.3f} seconds".format(end-start))

    torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "checkpoint.pth"))
