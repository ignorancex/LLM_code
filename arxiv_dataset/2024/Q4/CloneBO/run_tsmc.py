import os
import sys
import gc
import random
import pandas as pd
import numpy as np

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from pools import get_oracle_and_pool
from clonebo.mcmc_proposers import proposer_local_multi, scorer_local
from clonebo.seq_tools import print_difs, remove_spaces, is_alph
from clonebo import tsmc
from clonebo import importance_sample 


import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    if cfg.run.wandb:
        wandb.init(project="CloneBO", config=OmegaConf.to_container(cfg, resolve=True))

    ############### lset seeds ###############
    seed = cfg.run.seed
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ############### load oracles ###############    
    (cost_func, labeled_seqs, labels, (start_mean, start_std), only_cdr
    ) = get_oracle_and_pool(OmegaConf.to_container(cfg.oracle))

    ############### load models ###############
    model = AutoModelForCausalLM.from_pretrained("CloneBO/CloneLM-Heavy")
    tokenizer = AutoTokenizer.from_pretrained("CloneBO/CloneLM-Heavy")
    tokenizer.seq_sep_token = "[ClSep]"
    tokenizer.seq_sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.seq_sep_token)
    if torch.cuda.is_available():
        model.to('cuda')

    ############### settings for sampling ###############
    args = cfg.likelihoods
    
    max_steps = cfg.sample.max_steps
    n_cond = cfg.sample.n_cond
    clone_size = cfg.sample.clone_size
    n_particles = cfg.sample.n_particles
    n_resample = cfg.sample.n_resample
    total_mcmc_steps = cfg.sample.total_mcmc_steps

    ############### save files, reload ###############
    data_path = 'data/mcmc_runs'
    start_name = f'{cfg.oracle.oracle_name}_tm_heavy_r{cfg.run.seed}_nmut{cfg.oracle.n_labelled_mut}'
    model_fname = f'{cfg.run.name}_n_cond{cfg.sample.n_cond}\
_cl{cfg.sample.clone_size}_ms{cfg.sample.max_steps}_nr{cfg.sample.n_resample}\
{'_naive' if cfg.sample.importance_sample else ''}\
_np{cfg.sample.n_particles}_tsmc_sig{args.label_noise_sigma}'
    print("Running:", start_name, model_fname)
    os.system(f'mkdir {os.path.join(data_path)}')
    os.system(f'mkdir {os.path.join(data_path, start_name)}')
    os.system(f'mkdir {os.path.join(data_path, start_name, model_fname)}')
    if "labeled_seqs.npy" in os.listdir(os.path.join(data_path, start_name, model_fname)) and not cfg.run.redo:
        try:
            n_start_labels = len(labels)
            clone_path = os.path.join(data_path, start_name, model_fname, 'log_sampled_clones.npy')
            log_sampled_clones = np.load(clone_path).tolist()
            load_labeled_seqs = np.load(os.path.join(data_path, start_name, model_fname, "labeled_seqs.npy"))
            load_labels = np.load(os.path.join(data_path, start_name, model_fname, "labels.npy"))
            n_loaded = len(load_labels)-n_start_labels
            print(f"Loading! Found {n_loaded} previous samples.")
            for i in range(n_loaded):
                wandb.log({
                    "labeled_seq": wandb.Html(load_labeled_seqs[n_start_labels:][i]),
                    "label": load_labels[n_start_labels:][i],
                    "best_label": load_labels[:n_start_labels + i].max()
                })
            labels = load_labels
            labeled_seqs = load_labeled_seqs
            # fix clones if broken
            
            if len(log_sampled_clones) < n_loaded:
                print("Fixing clone at", clone_path)
                log_sampled_clones = ((n_loaded-len(log_sampled_clones)) * ['A']
                                      + log_sampled_clones)
                np.save(clone_path, log_sampled_clones)
            print("N clones:", len(log_sampled_clones), "at", clone_path)
        except Exception as e:
            print("Error loading. Starting from scratch.\nError:", e)
            steps = 0
            log_sampled_clones = []
    else:
        steps = 0
        log_sampled_clones = []

    ############### Set up tsmc ###############
    proposer = proposer_local_multi
        
    def get_con_inds(start_seq, labeled_seqs, labels, args, model, tokenizer, n_cond):
        label_liks = scorer_local([start_seq], labeled_seqs, labeled_seqs, labels,
                                                 args, model, tokenizer)
        return np.argsort(label_liks)[-n_cond:]
    
    ############### TSMC ###############
    steps = len(log_sampled_clones)
    while steps < total_mcmc_steps:
        print(f"\nRound {steps}:")
        torch.cuda.empty_cache()
        gc.collect()
    
        ##### get start and seqs to cond on #####
        start = np.random.choice(labeled_seqs[np.argsort(labels)[-n_resample:]])
        cond_inds = get_con_inds(start, labeled_seqs, labels, args, model, tokenizer, n_cond) # most likely given start
        cond_labels = labels[cond_inds]
        cond_seqs = labeled_seqs[cond_inds]
    
        ##### pick first seq in sampled clone #####
        init_clone = start
    
        ##### run SMC #####
        if not cfg.sample.importance_sample:
            smc = tsmc.smc(args, model, tokenizer, n_particles*[init_clone],
                           cond_seqs, (cond_labels - start_mean) / start_std,
                           sample_one_seq=True, keep_on_gpu=(n_cond<=75))
            for i in range(clone_size-1):
                print("Sampling sequence ", i+1)
                ess = smc.run_smc(150, steps_per_update=1 if n_cond<=75 else 20)
                smc = tsmc.smc.refresh_smc(args, smc, energy_resample=True)
            sampled_clone = smc.get_clones()[0]
        else:
            smc = importance_sample.smc(args, model, tokenizer, n_particles*[init_clone],
                         cond_seqs, (cond_labels - start_mean) / start_std,
                         batch_size=75)
            for l in range(clone_size-1):
                smc.sample_seq()
            sampled_clone = smc.importance_sample()

        valid_clone = np.all(is_alph(remove_spaces(sampled_clone.split(tokenizer.seq_sep_token))))
        if valid_clone:
            ##### propose seq #####
            print("Proposing sequence.")
            proposal = proposer([sampled_clone], labeled_seqs, labels,
                                args, model, tokenizer,
                                best_seq=labeled_seqs[np.argsort(labels)[-n_resample:]],
                                only_cdr=only_cdr, max_steps=max_steps)[0]
            y_new = cost_func(proposal)
            steps = steps + 1
    
            ##### log #####
            labels = np.r_[labels, [y_new]]        
            labeled_seqs = np.r_[labeled_seqs, [proposal]]
            log_sampled_clones.append(sampled_clone)
            np.save(os.path.join(data_path, start_name, model_fname, 'log_sampled_clones.npy'), log_sampled_clones)
            np.save(os.path.join(data_path, start_name, model_fname, 'labeled_seqs.npy'), labeled_seqs)
            np.save(os.path.join(data_path, start_name, model_fname, 'labels.npy'), labels)
            if cfg.run.wandb:
                wandb.log({
                    "sampled_clone": wandb.Table(data=[[line] for line in
                    sampled_clone.split(tokenizer.seq_sep_token)], columns=["Sampled clone"]),
                    "labeled_seq": wandb.Html(labeled_seqs[-1]),
                    "label": labels[-1],
                    "best_label": labels.max()
                })
    
            ##### print results #####
            x0 = remove_spaces([start])[0]
            print("Proposal:")
            print_difs(remove_spaces([proposal])[0], x0, color='black')
            print("Value of proposal:", y_new, "; vs. X0:", cost_func(start), "; vs. previous best:", labels[:-1].max())
            print("Sampled clone:")
            for seq in remove_spaces(sampled_clone.split(tokenizer.seq_sep_token)):
                print_difs(seq, x0, color='black')
        else:
            print("Invalid clone, restarting.")

if __name__ == "__main__":
    train()
