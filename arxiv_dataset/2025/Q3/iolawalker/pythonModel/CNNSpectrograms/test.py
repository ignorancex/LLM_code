#!/usr/bin/env python3
"""
test_spectrogram_cnn.py

Load test200hz.csv, compute sliding‐window spectrograms exactly as in train_spectrogram_cnn.py,
load the trained 2D‐CNN, and evaluate event‐level detection accuracy allowing
a ±TOL_MS‐ms slack around each true footfall.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_CSV    = "/home/u/footfallTraining/test200hz.csv"
WINDOW_SEC  = 1.0       # seconds per window
SR          = 200       # sampling rate (Hz)
WINDOW_SIZE = int(WINDOW_SEC * SR)
STRIDE      = int(0.25 * WINDOW_SIZE)
N_FFT       = 64
HOP_LENGTH  = 32
BATCH_SIZE  = 64
MODEL_PATH  = "best_spec_cnn.pth"
THRESHOLD   = 0.5       # probability cutoff
TOL_MS      = 100       # matching tolerance in ms
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────
class SpecCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── MATCHING ──────────────────────────────────────────────────────────────
def match_events(pred_times, true_times, tol):
    TP = FP = FN = 0
    used = set()
    errors = []
    for t in true_times:
        cands = [(abs(p - t), i, p) for i, p in enumerate(pred_times) if i not in used]
        if not cands:
            FN += 1; continue
        diff, i, p_closest = min(cands)
        if diff <= tol:
            TP += 1; used.add(i); errors.append(p_closest - t)
        else:
            FN += 1
    FP = len(pred_times) - len(used)
    return TP, FP, FN, errors

# ─── DATA PREP ─────────────────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV, sep=r"[,\t]+", engine="python")
true_times = df.loc[df.eventType=="footfall", "timestamp"].astype(int).tolist()
samples    = df[df.eventType=="sample"].reset_index(drop=True)
samples    = samples.rename(columns={
    "accelerationX":"ax",
    "accelerationY":"ay",
    "accelerationZ":"az"
})
for c in ("ax","ay","az"):
    samples[c] = samples[c].astype(np.float32)
timestamps = samples["timestamp"].astype(int).values

# build raw windows and record end‐of‐window timestamps
acc = samples[["ax","ay","az"]].values.astype(np.float32)
X_wins, Tw = [], []
N = len(acc)
for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    win = acc[start:end]  # (WINDOW_SIZE,3)

    # spectrogram per channel
    ch_specs = []
    for ch in range(3):
        sig = torch.from_numpy(win[:,ch])
        S   = torch.stft(sig,
                         n_fft=N_FFT,
                         hop_length=HOP_LENGTH,
                         win_length=N_FFT,
                         window=torch.hann_window(N_FFT),
                         return_complex=True)
        mag = torch.abs(S)  # (freq_bins, time_frames)
        ch_specs.append(mag)
    spec = torch.stack(ch_specs)  # (3, freq_bins, time_frames)

    X_wins.append(spec)
    Tw.append(timestamps[end-1])  # end‐of‐window timestamp

X = torch.stack(X_wins)           # (num_windows,3,freq_bins,frames)
Tw = np.array(Tw, dtype=int)

# ─── INFERENCE ─────────────────────────────────────────────────────────────
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
model = SpecCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

with torch.no_grad():
    logits = model(X.to(DEVICE))
probs = torch.sigmoid(logits).cpu().numpy()

# select predicted event times
pred_idxs  = np.where(probs > THRESHOLD)[0]
pred_times = Tw[pred_idxs].tolist()

# ─── MATCH & REPORT ────────────────────────────────────────────────────────
TP, FP, FN, errors = match_events(pred_times, true_times, TOL_MS)
print(f"Spectrogram CNN (±{TOL_MS} ms):")
print(f"  True events:      {len(true_times)}")
print(f"  Predicted events: {len(pred_times)}")
print(f"  TP = {TP}, FP = {FP}, FN = {FN}")
if errors:
    print(f"  Mean timing error: {np.mean(errors):.1f} ms ±{np.std(errors):.1f} ms")

# ─── OPTIONAL: BAR CHART ───────────────────────────────────────────────────
stats = {"TP": TP, "FP": FP, "FN": FN}
plt.figure(figsize=(6,4))
bars = plt.bar(stats.keys(), stats.values())
plt.title(f"Spectrogram CNN Predictions within {TOL_MS} ms")
plt.ylabel("Count")
for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{int(bar.get_height())}", ha="center")
plt.tight_layout()
plt.savefig("spec_cnn_event_stats.png")
print("Saved diagram → spec_cnn_event_stats.png")
