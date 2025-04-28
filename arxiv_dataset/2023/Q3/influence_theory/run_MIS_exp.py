"""This file finds the Most Influential Subsets approximation using Arnoldi approximation on zsRE data
"""
import os
import logging
import transformers
import torch
import numpy as np
import random
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from src.loss_functions import multi_loss_fn
from src.superquantile import reduce_superquantile
from src.utils import get_tokenizer, get_model, create_dataloader
from src.data_zsre import extract_data_zsre, Create_dataset
from src.approximation_alg import arnoldi_iter, distill, compute_influence_on_loss


# Set directory
cwd = Path(__file__).parent
os.chdir(cwd)


# Set Config file
OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


@hydra.main(config_path='config', config_name='config_MIS_exp')
def run(config):
    logs = []

    logging.info(f"Configuration: {config}")
    base_dir = hydra.utils.get_original_cwd()
    logging.info(f"Project base directory: {base_dir}")

    # Set seed for reproducibility
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)

    # Load model and tokenizer
    tokenizer = get_tokenizer(
        config.model.tokenizer_name, config.model.name, config.model.tokenizer_class)
    if config.task == 'wiki':
        tokenizer.pad_token = tokenizer.eos_token
    model_original = get_model(tokenizer, config.model.name, transformers,
                               config.model.class_name, config.model.pt, config.dropout, base_dir).to(config.device)

    # Download zsRE data set
    logging.info("Create train/test datasets")
    # Extract Subset of Training Set
    train_data, test_data = extract_data_zsre(
        config, model_original, tokenizer, base_dir)
    # Training data
    train_dataloader = create_dataloader(train_data, config.tr_bsz)

    # Handpicked testing points
    # Test_subset index's = [0, 7,8,12,18,24]
    te_raw_inpt = torch.load(
        base_dir+"/data/zsre_selected_test_raw_inpt.pt", map_location=config.device)
    te_raw_pred = torch.load(
        base_dir+"/data/zsre_selected_test_raw_pred.pt", map_location=config.device)
    te_tok_pred = torch.load(
        base_dir+"/data/zsre_selected_test_tok_pred.pt", map_location=config.device)

    test_data = Create_dataset(
        te_raw_inpt, te_tok_pred, te_raw_pred, tokenizer, config)

    test_dataloader = create_dataloader(test_data, 1)

    # Removed datapoint
    # Only want to use 1 removed point at a time
    removed_dataloader = create_dataloader(train_data, 1)

    logging.info("Start Approximation Calculations")
    # Compute eigenvalues and eigenvectors
    start_vector = [torch.randn_like(v).cpu()
                    for v in model_original.parameters()]
    result = arnoldi_iter(start_vector, model_original, config.device, train_dataloader,  config.method.regularization_param, config.method.num_epochs,
                          task=config.task, verbose=False)
    eigvals, eigvecs = distill(
        result, config.method.arnoldi.top_k, verbose=False)
    logging.info("Complete arnoldi approximation")
    logging.info("Start Test Point Gradient Loss")
    influence_ls = {}
    MIS = []
    # For 5 chosen test points find gradient of loss
    grad_loss_test_ls = []
    for i, te in enumerate(test_dataloader):
        print(i)
        # Loss over test set
        model_original.zero_grad()
        loss_test_pt = multi_loss_fn(model_original, te, config.task)
        grad_loss_test_pt = torch.autograd.grad(
            loss_test_pt, model_original.parameters())
        grad_loss_test_ls.append(grad_loss_test_pt)
    logging.info("Start Training Point Influence")
    # Find influence of each train_pt on each test_pt
    for j, grad_te in enumerate(grad_loss_test_ls):
        logs = []
        for rd in removed_dataloader:
            # Loss over removed point (training point)
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())

            # Find influence of 1 training point on 1 test point
            influence = compute_influence_on_loss(
                grad_loss_removed_pt, grad_te, eigvals, eigvecs)
            # Need to multiply by negative 1 (for superquantile formula)
            logs.append(-1*influence)
        logging.info("Complete one test point")
        MIS.append(reduce_superquantile(torch.tensor(logs),
                   superquantile_tail_fraction=1-config.alpha))
        logging.info(f"MIS: {MIS}")
        influence_ls[j] = torch.tensor(logs)

# Save Results
    results_dir = base_dir+config.results_dir
    results_path_infl = f"{results_dir}/results_MISinfluence_{config.task}_{config.n}_{config.method.num_epochs}_{config.alpha}.pt"
    torch.save(influence_ls, results_path_infl)
    results_path_MIS = f"{results_dir}/results_MIS_{config.task}_{config.n}_{config.method.num_epochs}_{config.alpha}.pt"
    torch.save(MIS, results_path_MIS)
    print("Results outputed to: ", results_path_MIS)


if __name__ == "__main__":
    run()
