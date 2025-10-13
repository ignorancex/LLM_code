#!/usr/bin/env python3
# coding: utf-8

"""
Perform contrastive alignment between music and painting embeddings to generate joint 128D embeddings
and a 909 x 4105 similarity matrix. Saves outputs to mozart-crossmodal/data/joint_embeddings.csv
and mozart-crossmodal/data/similarity_matrix.csv. Also generates plots for grid search and training losses.

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
"""


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(
        True, warn_only=True
    )  # Warns about potential performance impact

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MUSIC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "music")
PAINTING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "paintings")
MUSIC_CSV = os.path.join(MUSIC_DATA_DIR, "music_embeddings_258D_normalized.csv")
PAINTING_CSV = os.path.join(
    PAINTING_DATA_DIR, "painting_embeddings_258D_normalized.csv"
)
JOINT_CSV = os.path.join(PROJECT_ROOT, "data", "joint_embeddings.csv")
SIMILARITY_CSV = os.path.join(PROJECT_ROOT, "data", "similarity_matrix.csv")


# Prepare Data
class EmbeddingDataset(Dataset):
    def __init__(self, music_csv, painting_csv, indices=None):
        # Load music data
        try:
            music_df = pd.read_csv(music_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Music CSV not found at {music_csv}")
        except Exception as e:
            raise Exception(f"Error loading music CSV: {str(e)}")

        self.music_ids = music_df["song_id"].values
        self.music_embeddings = torch.tensor(
            music_df.iloc[:, 1:].values, dtype=torch.float32
        )

        # Check for NaNs in music V-A
        va_columns = music_df.columns[-2:]  # Last 2 columns should be V-A
        nan_mask = music_df[va_columns].isna().any(axis=1)
        if nan_mask.any():
            print(
                f"Warning: Found {nan_mask.sum()} music samples with NaN in V-A columns:"
            )
            print(music_df[nan_mask][["song_id"] + list(va_columns)])

        # Load painting data
        try:
            painting_df = pd.read_csv(painting_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Painting CSV not found at {painting_csv}")
        except Exception as e:
            raise Exception(f"Error loading painting CSV: {str(e)}")

        self.painting_ids = painting_df["ID"].values
        self.painting_embeddings = torch.tensor(
            painting_df.iloc[:, 1:].values, dtype=torch.float32
        )

        # Check for NaNs in painting V-A
        va_columns_painting = painting_df.columns[-2:]
        nan_mask_painting = painting_df[va_columns_painting].isna().any(axis=1)
        if nan_mask_painting.any():
            print(
                f"Warning: Found {nan_mask_painting.sum()} painting samples with NaN in V-A columns:"
            )
            print(painting_df[nan_mask_painting][["ID"] + list(va_columns_painting)])

        # Combine data
        self.embeddings = torch.cat(
            [self.music_embeddings, self.painting_embeddings], dim=0
        )
        self.ids = np.concatenate([self.music_ids, self.painting_ids])
        self.modality = torch.cat(
            [torch.zeros(len(music_df)), torch.ones(len(painting_df))]
        )

        if indices is not None:
            self.embeddings = self.embeddings[indices]
            self.ids = self.ids[indices]
            self.modality = self.modality[indices]

        self.size = len(self.embeddings)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.embeddings[idx], self.modality[idx], self.ids[idx]


def custom_collate(batch):
    embeddings, modalities, ids = zip(*batch)
    return torch.stack(embeddings), torch.stack(modalities), list(ids)


# Create datasets
full_dataset = EmbeddingDataset(MUSIC_CSV, PAINTING_CSV)
train_idx, val_idx = train_test_split(
    range(len(full_dataset)), test_size=0.2, random_state=42
)
train_dataset = EmbeddingDataset(MUSIC_CSV, PAINTING_CSV, train_idx)
val_dataset = EmbeddingDataset(MUSIC_CSV, PAINTING_CSV, val_idx)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate
)

# Compute weights
N_m, N_p = len(train_dataset.music_ids), len(train_dataset.painting_ids)  # 909, 4105
lambda_m = N_p / (N_m + N_p)  # ≈ 0.818
lambda_p = N_m / (N_m + N_p)  # ≈ 0.182
weights = torch.tensor([lambda_m, lambda_p], dtype=torch.float32)


# Define Model
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=258, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"


# Weighted Contrastive Loss
def weighted_contrastive_loss(embeddings, modalities, va_values, sigma, margin):
    va_dist = torch.cdist(va_values, va_values)
    S_ij = torch.exp(-(va_dist**2) / (2 * sigma**2))
    emb_dist = torch.cdist(embeddings, embeddings)
    weights_batch = weights[modalities.long()]
    weight_pairs = weights_batch.unsqueeze(1) * weights_batch.unsqueeze(0)

    loss_sim = weight_pairs * S_ij * emb_dist**2
    loss_diff = weight_pairs * (1 - S_ij) * torch.clamp(margin - emb_dist, min=0) ** 2

    return (loss_sim + loss_diff).mean() / 2


# Grid Search for Hyperparameters
def train_model(train_loader, val_loader, sigma, margin, num_epochs=20):
    model = ProjectionHead().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for embeddings, modalities, _ in train_loader:
            embeddings = embeddings.to(device)
            modalities = modalities.to(device)
            va_values = embeddings[:, -2:]

            optimizer.zero_grad()
            z = model(embeddings)
            loss = weighted_contrastive_loss(z, modalities, va_values, sigma, margin)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for embeddings, modalities, _ in val_loader:
            embeddings = embeddings.to(device)
            modalities = modalities.to(device)
            va_values = embeddings[:, -2:]
            z = model(embeddings)
            val_loss += weighted_contrastive_loss(
                z, modalities, va_values, sigma, margin
            ).item()

    return val_loss / len(val_loader)


# Define hyperparameter grid
sigma_values = [0.5, 1.0, 1.5, 2.0]
margin_values = [0.5, 1.0, 1.5, 2.0]
param_grid = list(product(sigma_values, margin_values))

# Perform grid search and store losses
val_losses = []
best_sigma, best_margin = None, None
best_val_loss = float("inf")

for sigma, margin in param_grid:
    print(f"Testing sigma={sigma}, margin={margin}")
    val_loss = train_model(train_loader, val_loader, sigma, margin)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_sigma, best_margin = sigma, margin

print(
    f"Best hyperparameters: sigma={best_sigma}, margin={best_margin}, Val Loss={best_val_loss:.6f}"
)

# Plot Grid Search Validation Losses with Plotly
grid_labels = [f"σ={sigma}, m={margin}" for sigma, margin in param_grid]
fig = go.Figure(
    data=[
        go.Bar(
            x=grid_labels,
            y=val_losses,
            marker_color="skyblue",
            text=[f"{v:.6f}" for v in val_losses],
            textposition="auto",
        )
    ]
)
fig.update_layout(
    title="Validation Loss Across Hyperparameter Grid",
    xaxis_title="Hyperparameter Combination (σ, m)",
    yaxis_title="Validation Loss",
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#f0f0f0",
    font=dict(size=12),
    xaxis=dict(tickangle=45),
    bargap=0.2,
)
try:
    fig.write_html(os.path.join(PROJECT_ROOT, "grid_search_validation_losses.html"))
    fig.write_image(os.path.join(PROJECT_ROOT, "grid_search_validation_losses.png"))
    fig.write_image(os.path.join(PROJECT_ROOT, "grid_search_validation_losses.pdf"))
    print(
        "Saved grid search validation loss plot as 'grid_search_validation_losses.html', '.png', and '.pdf'"
    )
except Exception as e:
    print(f"Error saving grid search plots: {str(e)}")

# Final Training with Best Hyperparameters and Early Stopping
model = ProjectionHead().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs_final = 50
train_losses = []
best_loss = float("inf")
best_model_state = None
patience = 5  # Number of epochs to wait for improvement
trigger_times = 0  # Counter for patience

for epoch in range(num_epochs_final):
    model.train()
    epoch_loss = 0
    for embeddings, modalities, _ in train_loader:
        embeddings = embeddings.to(device)
        modalities = modalities.to(device)

        optimizer.zero_grad()
        z = model(embeddings)
        va_values = embeddings[:, -2:]
        loss = weighted_contrastive_loss(
            z, modalities, va_values, best_sigma, best_margin
        )
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss_avg = epoch_loss / len(train_loader)
    train_losses.append(epoch_loss_avg)
    print(
        f"Final Training Epoch {epoch+1}/{num_epochs_final}, Loss: {epoch_loss_avg:.6f}"
    )

    # Early stopping logic
    if epoch_loss_avg < best_loss:
        best_loss = epoch_loss_avg
        best_model_state = model.state_dict()
        trigger_times = 0
        print(f"New best loss: {best_loss:.6f}, saving model state")
    else:
        trigger_times += 1
        print(f"No improvement, trigger times: {trigger_times}")
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Plot Final Training Losses with Plotly
fig = go.Figure(
    data=[
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="blue"),
            marker=dict(size=8),
        )
    ]
)
fig.update_layout(
    title="Training Loss Over Epochs",
    xaxis_title="Epoch",
    yaxis_title="Average Loss per Batch",
    plot_bgcolor="#f0f0f0",
    paper_bgcolor="#f0f0f0",
    font=dict(size=12),
    legend=dict(x=0.8, y=0.9),
    xaxis=dict(gridwidth=1, gridcolor="lightgray"),
    yaxis=dict(gridwidth=1, gridcolor="lightgray"),
)
try:
    fig.write_html(os.path.join(PROJECT_ROOT, "final_training_losses.html"))
    fig.write_image(os.path.join(PROJECT_ROOT, "final_training_losses.png"))
    fig.write_image(os.path.join(PROJECT_ROOT, "final_training_losses.pdf"))
    print(
        "Saved final training loss plot as 'final_training_losses.html', '.png', and '.pdf'"
    )
except Exception as e:
    print(f"Error saving training loss plots: {str(e)}")

# Save Joint Embeddings (128D only, no V-A)
with torch.no_grad():
    model.eval()
    all_embeddings = full_dataset.embeddings.to(device)
    joint_embeddings = model(all_embeddings)  # 128D embeddings
    joint_embeddings = joint_embeddings.cpu().numpy()

    # Split into music and painting embeddings for similarity matrix
    music_embeddings = joint_embeddings[:N_m]  # First 909 samples
    painting_embeddings = joint_embeddings[N_m:]  # Next 4105 samples

    # Save joint embeddings (without V-A)
    joint_df = pd.DataFrame(joint_embeddings, columns=[f"dim_{i}" for i in range(128)])
    joint_df.insert(0, "ID", full_dataset.ids)
    joint_df.insert(1, "modality", full_dataset.modality.numpy())
    try:
        joint_df.to_csv(JOINT_CSV, index=False)
        print(f"Saved joint embeddings (128D) to {JOINT_CSV}")
    except Exception as e:
        print(f"Error saving joint embeddings: {str(e)}")

# Compute and Save Similarity Matrix (909 × 4105)
# Convert embeddings to tensors
music_embeddings_tensor = torch.tensor(music_embeddings, dtype=torch.float32).to(device)
painting_embeddings_tensor = torch.tensor(painting_embeddings, dtype=torch.float32).to(
    device
)

# Compute pairwise Euclidean distances
similarity_matrix = (
    torch.cdist(music_embeddings_tensor, painting_embeddings_tensor).cpu().numpy()
)

# Save similarity matrix as CSV
similarity_df = pd.DataFrame(
    similarity_matrix, index=full_dataset.ids[:N_m], columns=full_dataset.ids[N_m:]
)
try:
    similarity_df.to_csv(SIMILARITY_CSV)
    print(f"Saved similarity matrix (909 × 4105) to {SIMILARITY_CSV}")
except Exception as e:
    print(f"Error saving similarity matrix: {str(e)}")
