#!/usr/bin/env python3
"""
test.py

Event‐level evaluation of the windowed logistic‐regression model,
plus a bar chart of TP, FP, FN counts (±TOL_MS).
"""
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 600    # must match train.py
STRIDE      = 150    # must match train.py
SCALER_PATH = "logistic_scaler_windowed.pkl"
MODEL_PATH  = "best_footfall_logistic_windowed.pt"
TEST_CSV    = "/home/u/footfallTraining/test200hz.csv"
THRESHOLD   = 0.50      # probability cutoff
TOL_MS      = 100       # matching tolerance in ms

# ─── MODEL ────────────────────────────────────────────────────────────────
class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

# ─── MATCHING ──────────────────────────────────────────────────────────────
def match_events(pred_times, true_times, tol):
    TP = FP = FN = 0
    used = set()
    errors = []
    for t in true_times:
        # find closest unused prediction
        cands = [(abs(p - t), i, p) for i, p in enumerate(pred_times) if i not in used]
        if not cands:
            FN += 1
            continue
        diff, i, p_closest = min(cands)
        if diff <= tol:
            TP += 1
            used.add(i)
            errors.append(p_closest - t)
        else:
            FN += 1
    FP = len(pred_times) - len(used)
    return TP, FP, FN, errors

# ─── LOAD & FEATURE PREP ───────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV, sep=r"[,\t]+", engine="python")
# extract true footfall timestamps
true_times = df.loc[df.eventType=="footfall", "timestamp"].astype(int).tolist()
# keep only sample rows
samples = df[df.eventType=="sample"].reset_index(drop=True)

# rename & convert accelerometer columns
samples = samples.rename(columns={
    "accelerationX": "ax",
    "accelerationY": "ay",
    "accelerationZ": "az"
})
for c in ("ax", "ay", "az"):
    samples[c] = samples[c].astype(np.float32)

# compute additional features
acc      = samples[["ax","ay","az"]].values                       # (N,3)
mag      = np.linalg.norm(acc, axis=1).reshape(-1,1)             # (N,1)
timestamps = samples["timestamp"].astype(int).values             # (N,)
dt_last  = samples["timestamp"].diff().fillna(0).values.reshape(-1,1).astype(np.float32)  # (N,1)

feats = np.hstack([acc.astype(np.float32), mag.astype(np.float32), dt_last])  # (N,5)

# load scaler & transform
scaler = joblib.load(SCALER_PATH)
scaled = scaler.transform(feats).astype(np.float32)

# ─── SLIDING WINDOWS ───────────────────────────────────────────────────────
Xw, Tw = [], []
N = len(scaled)
for start in range(0, N - WINDOW_SIZE + 1, STRIDE):
    window = scaled[start:start+WINDOW_SIZE]       # (WINDOW_SIZE,5)
    Xw.append(window.flatten())                    # (WINDOW_SIZE*5,)
    Tw.append(timestamps[start+WINDOW_SIZE-1])     # end‐of‐window timestamp

Xw = np.stack(Xw)          # (num_windows, WINDOW_SIZE*5)
Tw = np.array(Tw, dtype=int)

# ─── INFERENCE ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model  = LogisticModel(WINDOW_SIZE * feats.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    logits = model(torch.from_numpy(Xw).to(device))
probs = torch.sigmoid(logits).cpu().numpy()

# ─── PREDICTED EVENT TIMES ─────────────────────────────────────────────────
pred_times = [int(Tw[i]) for i, p in enumerate(probs) if p > THRESHOLD]

# ─── MATCH & REPORT ────────────────────────────────────────────────────────
TP, FP, FN, errors = match_events(pred_times, true_times, TOL_MS)
print("Windowed Logistic Regression (±{} ms):".format(TOL_MS))
print(f"  True events:      {len(true_times)}")
print(f"  Predicted events: {len(pred_times)}")
print(f"  TP = {TP}, FP = {FP}, FN = {FN}")
if errors:
    print(f"  Mean timing error: {np.mean(errors):.1f} ms ±{np.std(errors):.1f} ms")

# ─── PLOT COUNTS ───────────────────────────────────────────────────────────
stats = {"TP": TP, "FP": FP, "FN": FN}
plt.figure(figsize=(6,4))
bars = plt.bar(stats.keys(), stats.values())
plt.title(f"Windowed Logistic Predictions within {TOL_MS} ms")
plt.ylabel("Count")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{int(h)}", ha="center")
if errors:
    me, sd = np.mean(errors), np.std(errors)
    plt.text(0.5, max(stats.values())*0.9,
             f"Mean error: {me:.1f} ms ±{sd:.1f} ms",
             ha="center", color="gray")
plt.tight_layout()
plt.savefig("logistic_windowed_event_stats.png")
print("Saved diagram → logistic_windowed_event_stats.png")
