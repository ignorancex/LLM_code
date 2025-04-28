"""This file finetunes a base model on a subset of data.
Can be used for BART base/zsRE or Distil GPT2/WikiText
"""
import hydra
from pathlib import Path
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import os
import logging
import transformers
from src.utils import get_tokenizer, get_model
from src.training import training
from src.data_zsre import extract_data_zsre
from src.data_wiki import extract_data_wiki


# Set directory
cwd = Path(__file__).parent
os.chdir(cwd)

# Set Config file
OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

# Set default to float64
torch.set_default_dtype(torch.float64)


@hydra.main(config_path='config', config_name='config_train_model')
def run(config):
    logging.info(f"Configuration: {config}")
    base_dir = hydra.utils.get_original_cwd()
    logging.info(f"Project base directory: {base_dir}")

    # Set seed for reproducibility
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)

    # Download original model and tokenizer
    logging.info("Load Original Model")
    tokenizer = get_tokenizer(
        config.model.tokenizer_name, config.model.name, config.model.tokenizer_class)
    if config.task == 'wiki':
        tokenizer.pad_token = tokenizer.eos_token
    model_original = get_model(tokenizer, config.model.name, transformers,
                               config.model.class_name, config.model.pt, config.dropout, base_dir).to(config.device)

    # Download train/test data
    if config.task == "zsre":
        train_data, test_data = extract_data_zsre(
            config, model_original, tokenizer, base_dir)
    else:
        train_data, test_data = extract_data_wiki(
            config, tokenizer, base_dir)

    # Finetune model
    trained_model, training_logs = training(
        config, model_original, tokenizer, train_data, test_data, base_dir, lr=config.ft.lr)

    # Save Results
    results_dir = base_dir+config.results_dir
    results_path = f"{results_dir}/results_{config.task}_{config.data_seed}_{config.n}_{config.n_test}.pt"
    torch.save(training_logs, results_path)
    print("Results outputed to: ", results_path)


if __name__ == "__main__":
    run()
