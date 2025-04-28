import os
import random
import warnings

import urllib
import zipfile
warnings.filterwarnings("ignore")
import numpy as np
import torchvision
import wandb
import torch
torchvision.disable_beta_transforms_warning()
import constants as cst
import hydra
from config.config import Config
from run import run_wandb, run, sweep_init
from preprocessing.lobster import LOBSTERDataBuilder
from constants import Dataset
from config.config import MLPLOB, TLOB

@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    if (cst.DEVICE == "cpu"):
        accelerator = "cpu"
    else:
        accelerator = "gpu"
    if config.experiment.dataset_type == Dataset.FI_2010:
        config.experiment.batch_size = 32
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 144
    else:
        config.experiment.batch_size = 128 
        if config.model.type.value == "MLPLOB" or config.model.type.value == "TLOB":
            config.model.hyperparameters_fixed["hidden_dim"] = 46

    if config.experiment.dataset_type.value == "LOBSTER" and not config.experiment.is_data_preprocessed:
        # prepare the datasets, this will save train.npy, val.npy and test.npy in the data directory
        data_builder = LOBSTERDataBuilder(
            stocks=config.experiment.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=cst.DATE_TRADING_DAYS,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.experiment.sampling_type,
            sampling_time=config.experiment.sampling_time,
            sampling_quantity=config.experiment.sampling_quantity,
        )
        data_builder.prepare_save_datasets()
        
    elif config.experiment.dataset_type.value == "FI_2010" and not config.experiment.is_data_preprocessed:
        try:
            #take the .zip files name in data/FI_2010
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall(dir)  # Extracts to the current directory           
            print("Data extracted.")
        except Exception as e:
            raise(f"Error downloading or extracting data: {e}")
    if config.experiment.is_wandb:
        if config.experiment.is_sweep:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"])
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    else:
        run(config, accelerator)
    

def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision('high')
    
if __name__ == "__main__":
    set_torch()
    hydra_app()
    
