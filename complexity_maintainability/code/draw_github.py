import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

input_csv = "Metrics/result/averaged_output.csv"
output_dir = "plots_avg"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)
df = df.sort_values("quarter")

quarters = sorted(df["quarter"].unique())

def get_xticks(quarters):
    ticks, labels = [], []
    for q in quarters:
        if q.endswith("Q1") or q == "2025Q1":
            ticks.append(q)
            labels.append(q[:4])
    return ticks, labels

xticks, xtick_labels = get_xticks(quarters)

colors = {"cs": "#1f77b4", "non_cs": "#ff7f0e"}

def get_best_legend_loc(x_vals, y_vals):
    quadrants = defaultdict(int)
    x_mid = len(x_vals) // 2
    y_all = [y for series in y_vals for y in series if not np.isnan(y)]
    y_mid = (max(y_all) + min(y_all)) / 2 if y_all else 0
    for ys in y_vals:
        for i, y in enumerate(ys):
            if np.isnan(y):
                continue
            if i < x_mid and y >= y_mid:
                quadrants["upper left"] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants["upper right"] += 1
            elif i < x_mid and y < y_mid:
                quadrants["lower left"] += 1
            else:
                quadrants["lower right"] += 1
    return min(quadrants, key=quadrants.get) if quadrants else "best"

def fit_and_plot(x_idx, y_vals, color):
    x = np.array(x_idx)
    y = np.array(y_vals)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    poly = np.poly1d(coef)
    plt.plot(x[mask], poly(x[mask]),
             linestyle="--", linewidth=1.5, color=color, alpha=0.9)

def plot_metric(metric, ylabel):
    plt.figure(figsize=(3.5, 2.5))
    legend_vals = []

    for cat, color in colors.items():
        subset = df[df["category"] == cat]
        y_vals = subset[metric].tolist()
        legend_vals.append(y_vals)
        plt.plot(subset["quarter"], y_vals,
                 label=cat,
                 color=color,
                 linewidth=1.8)

        stage1 = [q for q in quarters if "2020Q1" <= q <= "2023Q1"]
        stage2 = [q for q in quarters if "2023Q2" <= q <= "2025Q3"]
        x_idx1 = [i for i, q in enumerate(quarters) if q in stage1]
        fit_and_plot(x_idx1, [y_vals[i] for i in x_idx1], "blue" if cat == "cs" else "red")
        x_idx2 = [i for i, q in enumerate(quarters) if q in stage2]
        fit_and_plot(x_idx2, [y_vals[i] for i in x_idx2], "blue" if cat == "cs" else "red")

    all_y = [v for series in legend_vals for v in series if not np.isnan(v)]
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)

    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(xticks, xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    if metric == "difficult":
        plt.legend(fontsize=9, loc="upper left", frameon=True,
                   facecolor="white", framealpha=1, labelspacing=0.2)
    else:
        best_loc = get_best_legend_loc(quarters, legend_vals)
        anchor_map = {
            "upper left": (0.0, 1.0),
            "upper right": (1.0, 1.0),
            "lower left": (0.0, 0.0),
            "lower right": (1.0, 0.0),
        }
        plt.legend(fontsize=9,
                loc=best_loc,
                bbox_to_anchor=anchor_map.get(best_loc, (1.0, 1.0)),
                frameon=True, facecolor="white", framealpha=1, labelspacing=0.2)

    if "2023Q1" in quarters:
        idx = quarters.index("2023Q1")
        plt.axvline(x=idx, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}_trend.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

plot_metric("cyclomatic", "Cyclomatic Complexity")
plot_metric("mi_custom", "Custom Maintainability Index")
plot_metric("bugs", "Estimated Bugs")
plot_metric("difficulty", "Difficulty")