#!/usr/bin/env python3
"""
train_spectrogram_cnn.py

1) Prepare & cache sliding‐window spectrogram dataset
2) Define 2D‐CNN model
3) Train & validate, saving best model
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
CSV_PATH     = "/home/u/footfallTraining/train200hz.csv"
CACHE_FILE   = "spec_cnn_data.pt"
MODEL_OUT    = "best_spec_cnn.pth"
WINDOW_SEC    = 1.0      # seconds per window
SR           = 200       # sampling rate (Hz)
WINDOW_SIZE  = int(WINDOW_SEC * SR)
STRIDE       = int(0.25 * WINDOW_SIZE)  # 75% overlap
N_FFT        = 64        # FFT size for spectrogram
HOP_LENGTH   = 32        # hop length for spectrogram
BATCH_SIZE   = 64
LR           = 1e-3
EPOCHS       = 50
VAL_RATIO    = 0.2
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1) DATA PREP ─────────────────────────────────────────────────────────
def build_and_cache():
    df = pd.read_csv(CSV_PATH, sep=r"[,\t]+", engine="python")
    # extract footfall timestamps
    event_ts = df.loc[df.eventType=="footfall", "timestamp"].values
    samples  = df[df.eventType=="sample"].reset_index(drop=True)

    # rename and convert accelerometer columns
    samples = samples.rename(columns={
        "accelerationX":"ax",
        "accelerationY":"ay",
        "accelerationZ":"az"
    })
    for c in ("ax","ay","az"):
        samples[c] = samples[c].astype(float)

    # label nearest sample to each footfall
    samples["target"] = 0
    for t in event_ts:
        idx = (samples["timestamp"] - t).abs().idxmin()
        samples.at[idx, "target"] = 1

    # build raw windows
    acc = samples[["ax","ay","az"]].values.astype(np.float32)  # (N,3)
    labels = samples["target"].values.astype(np.int64)
    X_wins, y_wins = [], []
    for start in range(0, len(acc) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        X_wins.append(acc[start:end])            # (WINDOW_SIZE,3)
        y_wins.append(int(labels[start:end].max()))

    # compute spectrogram per window & channel
    specs = []
    for win in X_wins:
        # win: (WINDOW_SIZE,3)
        # compute STFT along time for each of the 3 axes
        ch_specs = []
        for ch in range(3):
            sig = torch.from_numpy(win[:,ch]).float()
            S = torch.stft(sig,
                           n_fft=N_FFT,
                           hop_length=HOP_LENGTH,
                           win_length=N_FFT,
                           return_complex=True)
            # magnitude spectrogram: (freq_bins, time_frames)
            mag = torch.abs(S)
            ch_specs.append(mag)
        # stack channels → (3, freq_bins, time_frames)
        specs.append(torch.stack(ch_specs))
    X = torch.stack(specs)            # (N_windows,3,freq_bins,frames)
    y = torch.tensor(y_wins, dtype=torch.long)

    # cache
    torch.save({"X": X, "y": y}, CACHE_FILE)
    print(f"  ↳ built and cached spectrogram dataset → {CACHE_FILE}")
    return X, y

def load_or_build():
    if os.path.exists(CACHE_FILE):
        data = torch.load(CACHE_FILE)
        print(f"  ↳ loaded cached dataset ({CACHE_FILE})")
        return data["X"], data["y"]
    else:
        return build_and_cache()

# ─── 2) MODEL ─────────────────────────────────────────────────────────────
class SpecCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)           # output logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── 3) TRAIN/VALIDATE ────────────────────────────────────────────────────
def train():
    X, y = load_or_build()

    # split
    N = len(y)
    n_val = int(N * VAL_RATIO)
    n_trn = N - n_val
    full_ds = TensorDataset(X, y)
    trn_ds, val_ds = random_split(
        full_ds,
        [n_trn, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # model + criterion + optim
    model     = SpecCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        correct = total = 0
        for xb, yb in tqdm(trn_loader, desc=f"Epoch {epoch}/{EPOCHS} ▶ train"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds   = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total   += xb.size(0)

        train_acc = correct / total

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} ▷ val"):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).float()
                logits = model(xb)
                preds  = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total   += xb.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch:02d}: train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f" → Saved new best model (val_acc={best_acc:.4f})")

        scheduler.step(val_acc)

    print(f"Training complete. Best val_acc = {best_acc:.4f}")

if __name__ == "__main__":
    train()
