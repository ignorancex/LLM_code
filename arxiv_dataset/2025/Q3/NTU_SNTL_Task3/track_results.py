import os
import torch
import pandas as pd
from rich.progress import track

# How many top experiments to display
TOP_N = 20

rows = []
for exp in track(os.listdir("checkpoints"), description="Scanning experiments"):
    ckpt_file = os.path.join("checkpoints", exp, "best_model.pth")
    if not os.path.isfile(ckpt_file):
        continue

    # Load metrics
    ckpt = torch.load(ckpt_file, map_location="cpu")
    f1  = ckpt.get("best_f_score",         float("nan"))
    ang = ckpt.get("best_ang_err",         float("nan"))
    rde = ckpt.get("best_rel_dist_err",    float("nan"))
    seld_err = ((1 - f1) + ang/180 + rde) / 3

    rows.append({
        "Experiment":      exp,
        "F1":          f1 * 100,
        "LE":    ang,
        "RDE":  rde,
        "SELD Error":      seld_err
    })

# Build and sort the table, then take the top N
df = pd.DataFrame(rows)
df = df.sort_values("SELD Error", ascending=True).head(TOP_N)

# Print a clean table
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))