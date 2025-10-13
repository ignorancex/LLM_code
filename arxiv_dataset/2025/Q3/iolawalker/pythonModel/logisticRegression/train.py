#!/usr/bin/env python3
"""
train.py

Train a logistic‐regression model on shoe‐mounted accelerometer time‐series data
using sliding windows (3 s @200 Hz, 75% overlap) just like train_and_test_refined_v2.py.
"""
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
TRAIN_CSV      = "/home/u/footfallTraining/train200hz.csv"
SCALER_OUT     = "logistic_scaler_windowed.pkl"
MODEL_OUT      = "best_footfall_logistic_windowed.pt"
WINDOW_SIZE    = 600    # 3s @200 Hz
STRIDE         = 150    # 75% overlap
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 100
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42
CLIP_NORM      = 5.0

# ─── REPRODUCIBILITY ───────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ─── 1) LOAD & LABEL ───────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV, sep=r"[,\t]+", engine="python")
# extract true footfall timestamps
event_ts = df.loc[df.eventType=="footfall", "timestamp"].values
# keep only samples
samples = df[df.eventType=="sample"].reset_index(drop=True)

# rename and convert accelerometer columns
samples = samples.rename(columns={
    "accelerationX": "ax",
    "accelerationY": "ay",
    "accelerationZ": "az"
})
for c in ("ax","ay","az"):
    samples[c] = samples[c].astype(float)

# label nearest sample to each footfall
samples["target"] = 0
for t in event_ts:
    idx = (samples["timestamp"] - t).abs().idxmin()
    samples.at[idx, "target"] = 1

# build features: X,Y,Z; magnitude; delta‐time
acc = samples[["ax","ay","az"]].values.astype(np.float32)
mag = np.linalg.norm(acc, axis=1).reshape(-1,1).astype(np.float32)
dt_last = samples["timestamp"].diff().fillna(0).values.reshape(-1,1).astype(np.float32)

feats = np.hstack([acc, mag, dt_last])  # shape (N_samples, 5)
labs  = samples["target"].values.astype(np.int64)

# ─── 2) SCALE & SAVE ───────────────────────────────────────────────────────
scaler = StandardScaler().fit(feats)
joblib.dump(scaler, SCALER_OUT)
print(f"Saved scaler → {SCALER_OUT}")

feats = scaler.transform(feats).astype(np.float32)

# ─── 3) SLIDING WINDOWS ────────────────────────────────────────────────────
X, y = [], []
N, D = feats.shape
for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
    window = feats[start:start+WINDOW_SIZE]         # (WINDOW_SIZE, 5)
    X.append(window.flatten())                      # (WINDOW_SIZE * 5,)
    y.append(int(labs[start:start+WINDOW_SIZE].max()))  # 1 if any footfall in window

X = np.stack(X)            # (num_windows, WINDOW_SIZE*5)
y = np.array(y, dtype=np.int64)

# ─── 4) SPLIT & SAMPLER ────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ─── 5) DATASET & DATALOADERS ──────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = WindowDataset(X_train, y_train)
val_ds   = WindowDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ─── 6) MODEL DEFINITION ───────────────────────────────────────────────────
class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)  # raw logits

input_dim = WINDOW_SIZE * D
model     = LogisticModel(input_dim).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

# ─── 7) TRAIN LOOP ────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS+1):
    # training
    model.train()
    total_loss = correct = total = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == yb.long()).sum().item()
        total   += xb.size(0)

    train_loss = total_loss / total
    train_acc  = correct / total

    # validation
    model.eval()
    val_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total   += xb.size(0)

    val_loss = val_loss / total
    val_acc  = correct / total

    print(f"Epoch {epoch:03d}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # checkpoint if improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f" → New best model saved (val_acc={best_val_acc:.4f})")

    scheduler.step(val_acc)

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
