#!/usr/bin/env python3
"""
train_and_test_refined_v2.py

1) Train a Conv‐LSTM with per‐sample and per‐window heads
   – Use class‐weighted BCE on both heads
   – Oversample positive windows via WeightedRandomSampler
2) Export best model to TorchScript & save scaler/threshold metadata
3) Run test200hz.csv with tolerance sweep, printing TP/FN/FP/P/R/F1
"""
import os
import random
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
TRAIN_CSV      = "/home/u/footfallTraining/train200hz.csv"
TEST_CSV       = "/home/u/footfallTraining/test200hz.csv"
SCALER_OUT     = "scaler_refined_v2.pkl"
MODEL_OUT      = "best_refined_v2.pth"
TS_MODEL_OUT   = "refined_v2_ts.pt"
METADATA_OUT   = "metadata_refined_v2.json"

WINDOW_SIZE    = 600    # 3s @200Hz
STRIDE         = 150    # 75% overlap
BATCH_SIZE     = 64
LR             = 1e-3
EPOCHS         = 50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42
CLIP_NORM      = 5.0
POS_WINDOW_OVERSAMPLE = 10.0  # weight for windows with events

# ─── REPRODUCIBILITY ────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ─── DATASET ────────────────────────────────────────────────────────────
class SeqFootfallDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep=r"[,\t]+", engine="python")
        # extract footfall timestamps
        event_ts = df.loc[df.eventType=="footfall", "timestamp"].values
        samples  = df[df.eventType=="sample"].reset_index(drop=True)

        # numeric + rename
        samples[["accelerationX","accelerationY","accelerationZ"]] = \
            samples[["accelerationX","accelerationY","accelerationZ"]].astype(float)
        samples = samples.rename(columns={
            "accelerationX":"ax", "accelerationY":"ay", "accelerationZ":"az"
        })

        # label nearest sample to each footfall
        samples["target"] = 0
        for t in event_ts:
            idx = (samples["timestamp"] - t).abs().idxmin()
            samples.at[idx, "target"] = 1

        # features: magnitude + delta‐t
        acc = samples[["ax","ay","az"]].values
        samples["mag"]     = np.linalg.norm(acc, axis=1)
        samples["dt_last"] = samples["timestamp"].diff().fillna(0).values

        feats = samples[["ax","ay","az","mag","dt_last"]].values.astype(np.float32)
        labs  = samples["target"].values.astype(np.float32)

        # scale & persist
        self.scaler = StandardScaler().fit(feats)
        joblib.dump(self.scaler, SCALER_OUT)
        feats = self.scaler.transform(feats).astype(np.float32)

        # sliding windows → per‐sample & per‐window labels
        Xs, Ys, Ws = [], [], []
        N = len(feats)
        for st in range(0, N - WINDOW_SIZE + 1, STRIDE):
            ed = st + WINDOW_SIZE
            Xs.append(feats[st:ed])           # (WINDOW,5)
            Ys.append(labs[st:ed])            # (WINDOW,)
            Ws.append(float(labs[st:ed].max()))  # 1 if any footfall in window

        self.X = torch.from_numpy(np.stack(Xs))   # [M,WINDOW,5], float32
        self.Y = torch.from_numpy(np.stack(Ys))   # [M,WINDOW], float32
        self.W = torch.from_numpy(np.array(Ws, dtype=np.float32))  # [M]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.W[i]

# ─── MODEL ──────────────────────────────────────────────────────────────
class ConvLSTMGlobal(nn.Module):
    def __init__(self, in_ch=5, hid=128, layers=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, in_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(in_ch, hid, layers,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.seq_head    = nn.Linear(hid*2, 1)  # per-timestep
        self.global_head = nn.Linear(hid*2, 1)  # window-level

    def forward(self, x):
        # x: [B,WINDOW,5]
        c = x.permute(0,2,1)             # [B,5,WINDOW]
        c = self.conv(c)
        seq = c.permute(0,2,1)           # [B,WINDOW,5]
        out, _ = self.lstm(seq)          # [B,WINDOW,2*hid]
        seq_logits  = self.seq_head(out).squeeze(-1)       # [B,WINDOW]
        global_feat = out[:, -1, :]                         # [B,2*hid]
        glob_logits = self.global_head(global_feat).squeeze(-1)  # [B]
        return seq_logits, glob_logits

# ─── TRAIN → EXPORT → TEST ─────────────────────────────────────────────
def train_export_and_test():
    # 1) load dataset
    ds = SeqFootfallDataset(TRAIN_CSV)

    # 2) train/val split
    M     = len(ds)
    n_val = int(0.2 * M)
    n_trn = M - n_val
    train_ds, val_ds = random_split(ds, [n_trn, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    # 3) prepare train DataLoader with oversampling of positive windows
    W_train = ds.W[train_ds.indices]
    weights = torch.where(W_train>0,
                          torch.tensor(POS_WINDOW_OVERSAMPLE),
                          torch.tensor(1.0))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=2)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

    # 4) model, losses, optimizer
    model   = ConvLSTMGlobal().to(DEVICE)
    # sample‐level pos_weight
    pos_samples = ds.Y.sum().item()
    total_samples = ds.Y.numel()
    neg_samples = total_samples - pos_samples
    pw_sample = torch.tensor(neg_samples/pos_samples, dtype=torch.float32, device=DEVICE)
    seq_loss  = nn.BCEWithLogitsLoss(pos_weight=pw_sample)
    # window‐level pos_weight
    pos_windows = ds.W.sum().item()
    neg_windows = len(ds) - pos_windows
    pw_window = torch.tensor(neg_windows/pos_windows,
                             dtype=torch.float32, device=DEVICE)
    glob_loss = nn.BCEWithLogitsLoss(pos_weight=pw_window)
    opt       = AdamW(model.parameters(), lr=LR)
    sched     = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best_f1, best_thr = 0.0, 0.5
    thr_cands = np.linspace(0.01, 0.9, 90)

    # 5) training loop
    for ep in range(1, EPOCHS+1):
        model.train()
        for Xb, Yb, Wb in tqdm(train_ld, desc=f"Epoch {ep}/{EPOCHS}"):
            Xb, Yb, Wb = Xb.to(DEVICE), Yb.to(DEVICE), Wb.to(DEVICE)
            opt.zero_grad()
            seq_log, glob_log = model(Xb)
            l_seq  = seq_loss(seq_log, Yb)
            l_glob = glob_loss(glob_log, Wb)
            (l_seq + l_glob).backward()
            clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

        # validation & F1‐based threshold tuning
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for Xv, Yv, _ in val_ld:
                probs, _ = model(Xv.to(DEVICE))
                p = torch.sigmoid(probs).cpu().numpy().ravel()
                t = Yv.numpy().ravel()
                all_p.append(p); all_t.append(t)
        probs = np.concatenate(all_p)
        true  = np.concatenate(all_t)

        for thr in thr_cands:
            pred = (probs > thr).astype(int)
            f1   = f1_score(true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        print(f"→ Epoch {ep:02d}: val F1={best_f1:.4f} @ thr={best_thr:.3f}")
        sched.step(best_f1)
        torch.save(model.state_dict(), MODEL_OUT)

    # 6) export best model & metadata
    cpu_m = ConvLSTMGlobal().cpu()
    cpu_m.load_state_dict(torch.load(MODEL_OUT, map_location="cpu"))
    cpu_m.eval()
    example = torch.randn(1, WINDOW_SIZE, 5)
    ts = torch.jit.trace(cpu_m, example)
    ts.save(TS_MODEL_OUT)

    metadata = {
        "window": WINDOW_SIZE,
        "stride": STRIDE,
        "threshold": best_thr,
        "scaler": {
            "mean": ds.scaler.mean_.tolist(),
            "scale": ds.scaler.scale_.tolist()
        }
    }
    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=== Training & export complete ===")
    print(f"Model    → {TS_MODEL_OUT}")
    print(f"Metadata → {METADATA_OUT}")

    # 7) test with tolerance sweep
    print("\n=== Running test tolerance sweep ===")
    df_test = pd.read_csv(TEST_CSV, sep=r"[,\t]+", engine="python")
    ev_ts   = df_test.loc[df_test.eventType=="footfall","timestamp"].values
    sam     = df_test[df_test.eventType=="sample"].reset_index(drop=True)
    for c in ("accelerationX","accelerationY","accelerationZ"):
        sam[c] = pd.to_numeric(sam[c], errors="raise")
    sam = sam.rename(columns={"accelerationX":"ax",
                              "accelerationY":"ay",
                              "accelerationZ":"az"})
    sam["target"] = 0
    for t in ev_ts:
        idx = (sam.timestamp - t).abs().idxmin()
        sam.at[idx,"target"] = 1

    acc = sam[["ax","ay","az"]].values.astype(np.float32)
    mag = np.linalg.norm(acc, axis=1, keepdims=True).astype(np.float32)
    dt  = sam["timestamp"].diff().fillna(0).values.reshape(-1,1).astype(np.float32)
    feats = np.hstack([acc, mag, dt]).astype(np.float32)
    # scale
    feats = (feats - metadata["scaler"]["mean"]) / metadata["scaler"]["scale"]

    # slide windows
    Xw, starts = [], []
    N = len(feats)
    for st in range(0, N - WINDOW_SIZE + 1, STRIDE):
        Xw.append(feats[st:st+WINDOW_SIZE])
        starts.append(st)
    Xw = np.stack(Xw, axis=0)
    true_idxs = set(np.where(sam["target"].values==1)[0])

    # inference on CPU
    pred_idxs = set()
    with torch.no_grad():
        for i in range(0, len(Xw), BATCH_SIZE):
            batch = torch.from_numpy(Xw[i:i+BATCH_SIZE]).float()
            seq_log, _ = cpu_m(batch)
            prob = torch.sigmoid(seq_log).numpy()
            preds = (prob > best_thr).astype(int)
            for b in range(preds.shape[0]):
                st = starts[i+b]
                hits = np.where(preds[b]==1)[0]
                for h in hits:
                    pred_idxs.add(st+h)

    # sweep tolerance
    results = {}
    for tol in range(0, 51, 5):
        tp = fn = fp = 0
        matched = set()
        for ti in true_idxs:
            window = range(ti-tol, ti+tol+1)
            hits = [pi for pi in window if pi in pred_idxs]
            if hits:
                tp += 1
                matched.add(min(hits, key=lambda pi: abs(pi-ti)))
        fn = len(true_idxs) - tp
        fp = len(pred_idxs - matched)
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
        results[tol] = {
            "tp":tp, "fn":fn, "fp":fp,
            "precision":round(prec,4),
            "recall":   round(rec,4),
            "f1":       round(f1,4)
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    train_export_and_test()
