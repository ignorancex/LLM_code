import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
import torch
import esm
import ESM3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import ray
import torch.nn.functional as F
from ray import air, tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
import tempfile
from .model import esm2_model_mapping

ray.init(ignore_reinit_error=True)

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask
        x_masked = x * mask
        sum_x = torch.sum(x_masked, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
        average = sum_x / sum_mask
        return average

class Model(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, output_dim, repr_layers, self_attention_layers=False, dropout=0.1):
        super(Model, self).__init__()
        self.self_attention_layers = self_attention_layers
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        self.repr_layers = repr_layers
        if self.self_attention_layers:
            self.self_attention = nn.MultiheadAttention(embedding_dim, 8)
        self.masked_avg_pool = MaskedAveragePooling()
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, tokens, mask):
        results = self.pretrained_model(tokens, repr_layers=self.repr_layers, return_contacts=True)
        token_representations = results["representations"][self.repr_layers]
        if self.self_attention_layers:
            x = token_representations.permute(1, 0, 2)
            x_skip, _ = self.self_attention(x, x, x)
            x = x + x_skip
            x = x.permute(1, 0, 2)
            x = F.dropout(x, p=self.dropout, training=self.training)
        avg_pooled = self.masked_avg_pool(token_representations, mask)
        output = self.fc(avg_pooled)
        return output, results

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def train_or_validate(config, is_train=True, checkpoint_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config['model_name']
    if 'esm2' in model_path:
        pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        embedding_dim, layers = esm2_model_mapping[model_path]['embedding_dim'],esm2_model_mapping[model_path]['layers']
    else:
        pretrained_model, alphabet = ESM3.from_pretrained("esm3_sm_open_v1")
    pretrained_model.eval()
    pretrained_model = pretrained_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model = Model(pretrained_model, embedding_dim=embedding_dim, output_dim=1, self_attention_layers=True, repr_layers = layers, dropout=config["dropout"])
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        checkpoint_dir = './'

    file_path = config['file_path']
    df = load_data(file_path)
    sequence = config['sequence']

    if is_train:
        train_df, val_df = train_test_split(df, test_size=0.3, random_state=45)
    else:
        val_df = df

    model.train() if is_train else model.eval()

    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    data_loader = train_df if is_train else val_df
    for _, row in data_loader.iterrows():
        mutant_sequence = row['Mutations sequence']
        true_value = row['value']

        data = [
            ("original", sequence),
            ("mutant", mutant_sequence),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            mask = torch.ones_like(batch_tokens).to(device)
            mlp_output, results_esm = model(batch_tokens, mask)
            mutation_positions = [pos for pos, (orig_res, mut_res) in enumerate(zip(sequence, mutant_sequence)) if orig_res != mut_res]
            original_reps = torch.stack([results_esm["representations"][layers][0, pos + 1] for pos in mutation_positions])
            mutant_reps = torch.stack([results_esm["representations"][layers][1, pos + 1] for pos in mutation_positions])
            mutation_score = torch.norm(original_reps - mutant_reps, dim=1).mean()

            loss = criterion(mutation_score.unsqueeze(0), torch.tensor([true_value]).to(device))
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_samples += 1

            all_preds.append(mutation_score.item())
            all_targets.append(true_value)

    avg_loss = total_loss / total_samples
    mse = mean_squared_error(all_targets, all_preds)
    print(f"Avg loss: {avg_loss}, MSE: {mse}")
    if not is_train:
        return {"loss": avg_loss, "mse": mse, "preds": all_preds, "targets": all_targets}
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{avg_loss}_{mse}_{config}.pt")
    torch.save(model, checkpoint_path)

    train.report(
                {"loss": avg_loss, "mse": mse },
                checkpoint=checkpoint,
            )

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
    # tune.report(loss=avg_loss, mse=mse)

def main(config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):

    if config['lr'] is None:
        config['lr'] = tune.loguniform(1e-6, 1e-3)
    if config['dropout'] is None:
        config['dropout'] = tune.uniform(0.001, 0.3)
    if config['num_epochs'] is None:
        config['num_epochs'] = 30

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    result = tune.run(
        train_or_validate,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler
    )
    # print(result)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation MSE: {best_trial.last_result['mse']}")

# if __name__ == "__main__":
# main(num_samples=5, max_num_epochs=10, gpus_per_trial=0.3)

