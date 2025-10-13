#!/usr/bin/env python3
# coding: utf-8

"""
Reduce music features to 256D using an autoencoder, concatenate with normalized valence-arousal (V-A) labels,
Features are saved to mozart-crossmodal/data/music/music_embeddings_258D_normalized.csv.

Authors:
- Bereket A. Yilma <name.surname@artaicare.com>
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # /mozart-crossmodal
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "music")
INPUT_CSV = os.path.join(DATA_DIR, "music_features_with_embeddings.csv")
VA_CSV = os.path.join(DATA_DIR, "filtered_songs.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "music_embeddings_258D_normalized.csv")

# Load features
try:
    feature_df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
except Exception as e:
    raise Exception(f"Error loading input CSV: {str(e)}")

# Load V-A data
try:
    va_df = pd.read_csv(VA_CSV)
except FileNotFoundError:
    raise FileNotFoundError(f"V-A CSV not found at {VA_CSV}")
except Exception as e:
    raise Exception(f"Error loading V-A CSV: {str(e)}")

# Ensure song_id is an integer
va_df["song_id"] = va_df["song_id"].astype(int)
feature_df["song_id"] = feature_df["song_id"].astype(int)

# Extract feature columns (excluding song_id, valence, arousal)
feature_columns = [
    col for col in feature_df.columns if col not in ["song_id", "valence", "arousal"]
]
features = feature_df[feature_columns].values  # Feature matrix

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert to PyTorch tensor
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

# Check input dimension
input_dim = features_tensor.shape[1]  # Number of input features
latent_dim = 256  # Target compressed representation size


# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),  # 256D output
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),  # Reconstruct original input
        )

    def forward(self, x):
        z = self.encoder(x)  # Compressed representation
        x_recon = self.decoder(z)  # Reconstructed input
        return x_recon, z


# Initialize model, loss, and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder(input_dim, latent_dim).to(device)
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the autoencoder
num_epochs = 50
batch_size = 256
loss_values = []

for epoch in range(num_epochs):
    permutation = torch.randperm(features_tensor.size(0))
    epoch_loss = 0

    for i in range(0, features_tensor.size(0), batch_size):
        batch_indices = permutation[i : i + batch_size]
        batch_x = features_tensor[batch_indices].to(device)

        optimizer.zero_grad()
        x_recon, z = model(batch_x)
        loss = criterion(x_recon, batch_x)  # Minimize reconstruction error
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)  # Weighted by batch size

    avg_loss = epoch_loss / features_tensor.size(0)  # Average loss per sample
    loss_values.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

print("Training complete! Extracting embeddings...")

# Extract the compressed 256D embeddings
with torch.no_grad():
    music_embeddings = model.encoder(features_tensor.to(device)).cpu().numpy()

# Normalize embeddings (Min-Max scaling to [0, 1])
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
music_embeddings_norm = min_max_scaler.fit_transform(music_embeddings)

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(
    music_embeddings_norm, columns=[f"feat_{i}" for i in range(latent_dim)]
)
embeddings_df["song_id"] = feature_df["song_id"]

# Normalize V-A values
va_scaler = StandardScaler()
va_normalized = va_scaler.fit_transform(va_df[["valence_mean", "arousal_mean"]])
va_df[["valence_mean", "arousal_mean"]] = va_normalized

# Merge embeddings with V-A data
try:
    final_df = pd.merge(
        embeddings_df,
        va_df[["song_id", "valence_mean", "arousal_mean"]],
        on="song_id",
        how="inner",
        validate="one_to_one",  # Ensures no duplicate IDs
    )
    print(f"Embeddings before merge: {len(embeddings_df)}, after: {len(final_df)}")
except ValueError as e:
    raise ValueError(f"Error merging embeddings with V-A data: {str(e)}")
except Exception as e:
    raise Exception(f"Unexpected error during merge: {str(e)}")

# Reorder columns (song_id first)
columns = ["song_id"] + [col for col in final_df.columns if col != "song_id"]
final_df = final_df[columns]

# Save the final 258D embeddings
try:
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved 258D normalized embeddings to {OUTPUT_CSV}")
except Exception as e:
    raise Exception(f"Error saving output CSV: {str(e)}")
