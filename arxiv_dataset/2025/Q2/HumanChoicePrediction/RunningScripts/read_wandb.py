import pandas as pd 
import wandb
from tqdm import tqdm
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

class wandb_results:
    def __init__(self, project_id, wandb_username="eilamshapira"): 
        self.api = wandb.Api(timeout=60)
        self.project_id = project_id
        self.wandb_username = wandb_username

    def get_sweep_results(self, sweep_id, metric="accuracy_all", best_epoch=False, get_servers=False, reset_api=False, read_csv_if_exist=True, save_to_csv=True):
        if reset_api:
            self.reset_api()

        print(f"Download {sweep_id=} data...")
        runs = self.api.sweep(f"{self.wandb_username}/{self.project_id}/{sweep_id}").runs
        n_runs = len(runs)
        path = f"sweeps_csvs/{sweep_id}_{n_runs}.csv"
        if read_csv_if_exist and os.path.exists(path):
            return pd.read_csv(path, index_col=0)
        summary_list, config_list, name_list = [], [], []
        for run in tqdm(runs): 
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k,v in run.config.items()
                  if not k.startswith('_')})
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })
        config_cols = pd.json_normalize(runs_df['config'])
        config_cols.columns = [f"config_{c}" for c in config_cols.columns]
        summary_cols = pd.json_normalize(runs_df['summary'])
        runs_df = pd.concat([runs_df, config_cols, summary_cols], axis=1)
        runs_df.drop(['config', 'summary'], axis=1, inplace=True)
        hpt = [c for c in config_cols.columns if c not in ["config_seed", "config_run_hash"]]
        if save_to_csv: runs_df.to_csv(path)
        return runs_df
    
    def get_sweeps_results(self, sweeps, metric="accuracy_all", best_epoch=False, get_servers=False,  read_csv_if_exist=True, save_to_csv=True):
        print("Total number of sweeps:", len(sweeps))
        j = pd.concat([self.get_sweep_results(sweep, metric=metric, best_epoch=best_epoch,  get_servers=get_servers, save_to_csv=save_to_csv, read_csv_if_exist=read_csv_if_exist) for sweep in sweeps])
        j = j.reset_index(drop=True)
        return j
    
    def reset_api(self):
        self.api = wandb.Api()