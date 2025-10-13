import sys
import os
import time
import math

from omegaconf import OmegaConf
import pandas as pd
import json

from tqdm import tqdm
import random

import numpy as np
import torch
import torch.nn as nn


# tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def prepare_batch(batch, device):
    if isinstance(batch, tuple) or isinstance(batch, list):
        X, Y = batch
    elif isinstance(batch, dict):
        X = batch['input_ids'][:, :-1]
        Y = batch['input_ids'][:, 1:]
    else:
        raise ValueError(f"Invalid batch type: {type(batch)}")
    X, Y = X.to(device), Y.to(device)
    return X, Y

            
@torch.no_grad()
def evaluation(model, dataloader, criterion, config, device):
    model.eval()
    total_loss = 0
    total_words = 0
    total_correct = 0

    for i, batch in enumerate(dataloader):
        X, Y = prepare_batch(batch, device)

        preds = model(X)
        loss = criterion(preds, Y)

        total_loss += loss.item()
        total_correct += (preds.argmax(dim=-1) == Y).sum().item()
        total_words += (Y != config.dataset.pad_token_id).sum().item()
        
    loss = total_loss / total_words
    perplexity = np.exp(loss)
    accuracy = total_correct / total_words
    return loss, perplexity, accuracy


def train_model(model, config, train_loader, val_loader, criterion):
    """
    Trains a model based on the provided configuration, logs results, and saves models.
    
    Args:
        model (Model): Model to train.
        config (Config): Experiment configuration.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader, optional): Validation data.
        criterion (Loss): Loss function.
    """
    # Create output directory based on state_transition attribute
    if not config.model.use_dfa:
        run_dir = f"./runs/{config.name}/BPTT/"
    else:
        run_dir = f"./runs/{config.name}/{config.model.state_transition}/"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_object(config), f, indent=4)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay) 
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay) 
    
    # create scheduler
    total_training_steps = int(len(train_loader) * config.training.epochs)
    if config.training.scheduler == 'step':
        assert config.training.warmup_ratio == 0, "StepLR scheduler does not support warmup"
        kwargs = {**config.training.scheduler_kwargs}
        kwargs['step_size'] *= len(train_loader)  # convert epochs to steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif config.training.scheduler == 'cosine':
        warmup_steps = round(config.training.warmup_ratio * total_training_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps - warmup_steps, eta_min=0.1*config.training.learning_rate)
        ], milestones=[warmup_steps])
    
    else:
        raise ValueError(f"Invalid scheduler type: {config.training.scheduler}")

    # Tracking metrics
    history = {"epoch": [], "lr": [], "train_loss": [], "valid_loss": [], "valid_perplexity": [], "valid_accuracy": []}
    best_perplexity = float("inf")
    best_model_path = os.path.join(run_dir, "best_model.pth")

    steps = 0
    total_loss = 0
    total_words = 0
    running_loss_avg = None
    start_time = time.time()
    
    eval_interval = total_training_steps // config.training.num_evals
    bar = tqdm(range(total_training_steps), desc="Training", leave=True, file=sys.stdout)
    while steps < total_training_steps:
        for batch in train_loader:
            
            model.train()
            X, Y = prepare_batch(batch, device)
            num_words = (Y != config.dataset.pad_token_id).sum().item()

            optimizer.zero_grad()
            preds = model(X)
            loss_sum = criterion(preds, Y)
            loss = loss_sum / num_words
            loss.backward()

            if hasattr(config.training, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()

            total_loss += loss_sum.item()
            total_words += num_words
            running_loss_avg = loss.item() if running_loss_avg is None else 0.99 * running_loss_avg + 0.01 * loss.item()
            steps += 1
            bar.update(1)
            epoch = steps / len(train_loader)

            # maybe log
            if steps % config.training.log_interval == 0 and steps > 0:
                tqdm.write(f'|epoch {epoch:.3f} | train loss {running_loss_avg:5.2f} | train ppl {np.exp(running_loss_avg):8.2f} |')
            
            # maybe eval
            if steps % eval_interval == 0 and steps > 0:
                avg_train_loss = total_loss / total_words
                total_loss = 0
                total_words = 0

                avg_valid_loss, valid_perplexity, valid_accuracy = evaluation(model, val_loader, criterion, config=config, device=device)
        
                # Save best model if perplexity improves
                if valid_perplexity < best_perplexity:
                    best_perplexity = valid_perplexity
                    torch.save(model.state_dict(), best_model_path)

                # Save metrics to history
                history["epoch"].append(epoch)
                history["lr"].append(optimizer.param_groups[0]["lr"])
                history["train_loss"].append(avg_train_loss)
                history["valid_loss"].append(avg_valid_loss)
                history["valid_perplexity"].append(valid_perplexity)
                history["valid_accuracy"].append(valid_accuracy)
                
                tqdm.write(f'|epoch {epoch:.3f} | train loss {avg_train_loss:5.2f} | train ppl {np.exp(running_loss_avg):8.2f} | eval loss {avg_valid_loss:5.2f} | eval ppl {valid_perplexity:8.2f} | eval acc {valid_accuracy:3.2%} |')

            scheduler.step()

    bar.close()

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    print(f"✅ Training complete for state_transition={config.model.state_transition}, measure={config.model.measure}). Best Perplexity: {best_perplexity:.2f}")
