"""This file runs influence function approximation for each method using zsRE or WikiText data
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
from src.utils import get_tokenizer, get_model, create_dataloader, avg_grad, arnoldi_loss, gather_flat_grad
from src.data_zsre import extract_data_zsre
from src.data_wiki import extract_data_wiki
from src.loss_functions import multi_loss_fn
from src.approximation_alg import conjugate_gradient, sgd, identity, arnoldi_iter, distill, variance_reduction, compute_influence_on_loss


# Set directoryP
cwd = Path(__file__).parent
os.chdir(cwd)

# Set Config file
OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


@hydra.main(config_path='config', config_name='config_IF_exp')
def run(config):
    logs = []
    loss_results = []

    logging.info(f"Configuration: {config}")
    base_dir = hydra.utils.get_original_cwd()
    # base_dir = "/home/jrfish/nlp/AISTAT_2022_InfuenceFunction_original"
    logging.info(f"Project base directory: {base_dir}")

    # Set seed for reproducibility
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)

    # Download original model and tokenizer
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

    # Process data
    logging.info("Create train/test datasets")
    # Training data
    train_dataloader = create_dataloader(train_data, config.tr_bsz)
    # Testing data
    test_dataloader = create_dataloader(test_data, config.te_bsz)
    # Removed data
    # Only want to use 1 removed point at a time
    removed_dataloader = create_dataloader(train_data, 1)

    # Get eigenvalues and eignevectors if using Arnoldi approx (only need to find this once)
    if config.approx_method == 'arnoldi':
        logging.info(
            "Start Eigenvalue/Eigenvector Calculations for Arnoldi Method")
        start_vector = [torch.randn_like(v).cpu()
                        for v in model_original.parameters()]
        result = arnoldi_iter(start_vector, model_original, config.device, train_dataloader,  config.method.regularization_param, config.method.num_epochs,
                              task=config.task, verbose=False)
        eigvals, eigvecs = distill(
            result, config.method.arnoldi.top_k, verbose=False)

    logging.info(f"Starting approximation uisng {config.approx_method}")
    count = 0
    for rd in removed_dataloader:
        if config.approx_method == 'conjugate_gradient':
            z_lst = [torch.zeros_like(v) for v in model_original.parameters()]
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            IF, loss_results_1run = conjugate_gradient(z_lst,  grad_loss_removed_pt, model_original, config.device, train_dataloader,
                                                       config.method.regularization_param, config.method.conjugate_grad.eps, config.method.num_epochs, config.task, config.method.loss_at_epoch, config.break_early)
        elif config.approx_method == 'arnoldi':
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            loss_results_1run = arnoldi_loss(
                eigvecs, eigvals, grad_loss_removed_pt)
        elif config.approx_method == 'svrg':
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            IF, loss_results_1run = variance_reduction(grad_loss_removed_pt, model_original, config.device, train_dataloader, config.method.regularization_param,
                                                       config.method.lr, config.method.num_epochs, config.task, config.method.loss_at_epoch)
        elif config.approx_method == 'sgd':
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            IF, loss_results_1run = sgd(grad_loss_removed_pt, model_original, config.device, train_dataloader, config.method.regularization_param,
                                        config.method.lr, config.method.num_epochs, config.task, config.method.loss_at_epoch, config.break_early)
        elif config.approx_method == 'identity':
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            IF, loss_results_1run = identity(grad_loss_removed_pt, model_original, config.device, train_dataloader, config.method.regularization_param,
                                        config.method.lr, config.method.num_epochs, config.task, config.method.loss_at_epoch, config.break_early)
        else:
            logging.error("Approximation Method is Invalid")
            break

        logging.info(f"Iterations Completed: {count}")
        # Compute IF over test set
        param_influence = list(model_original.parameters())
        # Loss over test set
        model_original.zero_grad()
        test_grads = avg_grad(test_dataloader, model_original,
                              config.device, param_influence, config.task)
        if config.approx_method == 'arnoldi':
            loss_removed_pt = multi_loss_fn(model_original, rd, config.task)
            grad_loss_removed_pt = torch.autograd.grad(
                loss_removed_pt, model_original.parameters())
            influence = compute_influence_on_loss(
                grad_loss_removed_pt, test_grads, eigvals, eigvecs)
        else:
            influence = torch.dot(IF, gather_flat_grad(test_grads)).item()
        logs.append(influence)
        loss_results.append(loss_results_1run)
        count = count + 1
        print(count)
        results = {'logs': logs, 'loss': loss_results}
        if count >= config.iterations:
            break
        logging.info(f"Influence: {influence}")
    # Save Results
    results_dir = base_dir+config.results_dir
    results_path = f"{results_dir}/results_{config.task}_{config.n}_{config.approx_method}_{config.method.num_epochs}_{config.method.regularization_param}.pt"

    torch.save(results, results_path)
    print("Results outputed to: ", results_path)


if __name__ == "__main__":
    run()
