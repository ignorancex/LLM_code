import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
                quadrants['upper left'] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants['upper right'] += 1
            elif i < x_mid and y < y_mid:
                quadrants['lower left'] += 1
            else:
                quadrants['lower right'] += 1

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

def plot_comment_ratio_from_csv(comment_csv_file, output_dir, lang="python"):
    df = pd.read_csv(comment_csv_file)
    df = df.sort_values("Quarter")

    quarters = df["Quarter"].tolist()
    cs_ratios = df["CS_Comment_Ratio"].tolist()
    noncs_ratios = df["NonCS_Comment_Ratio"].tolist()

    custom_xticks, custom_xtick_labels = [], []
    for q in quarters:
        if q.endswith("Q1") or q == "2025Q1":
            custom_xticks.append(q)
            custom_xtick_labels.append(q[:4])

    colors = {"cs": "#1f77b4", "non_cs": "#ff7f0e"}
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(3.5, 2.5))

    plt.plot(quarters, cs_ratios, linestyle='-', linewidth=2,
             label="cs", color=colors["cs"])
    plt.plot(quarters, noncs_ratios, linestyle='-', linewidth=2,
             label="non_cs", color=colors["non_cs"])

    stage1 = [q for q in quarters if "2020Q1" <= q <= "2022Q4"]
    stage2 = [q for q in quarters if "2023Q1" <= q <= "2025Q1"]

    x_idx1 = [i for i, q in enumerate(quarters) if q in stage1]


    all_y = [v for v in cs_ratios + noncs_ratios if not np.isnan(v)]
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), min(1, y_max + margin))
    else:
        plt.ylim(0, 1)

    plt.ylabel("Comment Ratio", fontsize=10)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    best_loc = get_best_legend_loc(quarters, [cs_ratios, noncs_ratios])
    anchor_map = {
        "upper left": (0.0, 1.0),
        "upper right": (1.0, 1.0),
        "lower left": (0.0, 0.0),
        "lower right": (1.0, 0.0),
    }
    plt.legend(
        fontsize=10,
        loc=best_loc,
        bbox_to_anchor=anchor_map.get(best_loc, (1.0, 1.0)),
        frameon=True,
        facecolor='white',
        framealpha=1,
        labelspacing=0.2,
    )


    plt.tight_layout()

    save_path = os.path.join(output_dir, f"comment_ratio_{lang}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comment ratio plot saved to: {save_path}")

if __name__ == "__main__":
    lang = "cpp"
    comment_csv_file = f"naming_patterns/github_result/naming_patterns_{lang}/comment_ratio_{lang}.csv"
    output_dir = f"naming_patterns/github_result/naming_patterns_{lang}/plots_{lang}"
    plot_comment_ratio_from_csv(comment_csv_file, output_dir, lang)
