#!/usr/bin/env python3
# coding: utf-8

"""
Combines MERT and LLM features for music and reduces dimensionality to 256D using an autoencoder.
Saves reduced embeddings to mozart-crossmodal/data/music/music_multi_modal_256D.csv and loss plot to loss_plot.pdf.


Inputs:
- MERT features: data/music/music_embeddings_258D_normalized.csv (258D per track)
- LLM features: data/music/llm_music_embeddings.csv (768D per track)

Author:
- Bereket A. Yilma <name.surname@artaicare.com>
"""

import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import plotly.graph_objects as go

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA if available
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Limit PyTorch threads to avoid CPU overload
torch.set_num_threads(1)
logging.info(f"Set PyTorch threads to {torch.get_num_threads()}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /mozart-crossmodal
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "music")
MERT_CSV = os.path.join(DATA_DIR, "music_embeddings_258D_normalized.csv")
LLM_CSV = os.path.join(DATA_DIR, "llm_music_embeddings.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "music_multi_modal_256D.csv")
PLOT_PATH = os.path.join(DATA_DIR, "loss_plot.pdf")


# Load features
def load_features():
    try:
        mert_df = pd.read_csv(MERT_CSV)
        logging.info(f"Loaded MERT features: {mert_df.shape} (258D + song_id)")
        llm_df = pd.read_csv(LLM_CSV)
        logging.info(f"Loaded LLM features: {llm_df.shape} (768D + song_id)")

        # Ensure matching IDs
        mert_ids = set(mert_df["song_id"])
        llm_ids = set(llm_df["song_id"])
        common_ids = mert_ids.intersection(llm_ids)
        if not common_ids:
            raise ValueError("No common song_id between MERT and LLM features")
        logging.info(f"Found {len(common_ids)} common song_id")

        # Filter to common IDs
        mert_df = mert_df[mert_df["song_id"].isin(common_ids)].sort_values("song_id")
        llm_df = llm_df[llm_df["song_id"].isin(common_ids)].sort_values("song_id")

        # Extract feature columns
        mert_features = mert_df.drop(columns=["song_id"]).values  # 258D
        llm_features = llm_df.drop(columns=["song_id"]).values  # 768D

        # Verify dimensions
        if mert_features.shape[1] != 258 or llm_features.shape[1] != 768:
            raise ValueError(
                f"Unexpected feature dimensions: MERT={mert_features.shape[1]}, LLM={llm_features.shape[1]}"
            )

        # Concatenate features
        combined_features = np.concatenate(
            [mert_features, llm_features], axis=1
        )  # 258 + 768 = 1026D
        logging.info(f"Combined features shape: {combined_features.shape}")

        return list(mert_df["song_id"]), combined_features
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature file not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading features: {str(e)}")


# Define dataset for PyTorch
class MusicDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# Define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=1026, hidden_dim=256):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, hidden_dim), nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Train the autoencoder with validation and early stopping
def train_autoencoder(features, epochs=500, batch_size=32, patience=10, val_split=0.2):
    # Split into train and validation with fixed seed
    train_features, val_features = train_test_split(
        features, test_size=val_split, random_state=SEED
    )
    train_dataset = MusicDataset(train_features)
    val_dataset = MusicDataset(val_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Autoencoder(input_dim=features.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logging.info(
        f"Training autoencoder with {len(train_dataset)} train samples, {len(val_dataset)} val samples, input_dim={features.shape[1]}"
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            batch = batch.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                encoded, decoded = model(batch)
                loss = criterion(decoded, batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(
                    f"Early stopping triggered at epoch {epoch+1}, best val loss: {best_val_loss:.6f}"
                )
                model.load_state_dict(best_model_state)
                break

    # Extract reduced features using the full dataset with best model
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        reduced_features, _ = model(features_tensor)
        reduced_features = reduced_features.cpu().numpy()

    return reduced_features, train_loss_history, val_loss_history


# Plot and save loss curves
def plot_loss(train_loss_history, val_loss_history):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_loss_history) + 1)),
            y=train_loss_history,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_loss_history) + 1)),
            y=val_loss_history,
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="red", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title="Autoencoder Training and Validation Loss (Music)",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        plot_bgcolor="lightgray",
        paper_bgcolor="white",
        font=dict(size=12),
        width=800,
        height=600,
        legend=dict(x=0.8, y=0.9),
    )
    fig.write_image(PLOT_PATH)
    logging.info(f"Loss plot saved to {PLOT_PATH}")


# Main execution
if __name__ == "__main__":
    # Load and combine features
    song_ids, combined_features = load_features()

    # Train autoencoder and reduce to 256D
    reduced_features, train_loss_history, val_loss_history = train_autoencoder(
        combined_features, epochs=500
    )

    # Save reduced embeddings
    output_df = pd.DataFrame(reduced_features, columns=[f"z{i}" for i in range(256)])
    output_df.insert(0, "song_id", song_ids)
    try:
        output_df.to_csv(OUTPUT_CSV, index=False)
        logging.info(
            f"Reduced embeddings saved to {OUTPUT_CSV} with shape {output_df.shape}"
        )
    except Exception as e:
        raise Exception(f"Error saving output CSV: {str(e)}")

    # Plot loss
    plot_loss(train_loss_history, val_loss_history)
