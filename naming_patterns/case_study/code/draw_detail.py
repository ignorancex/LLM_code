import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


input_json = "naming_new/naming_patterns_function.json"
output_dir = "naming_new/plots_function"
os.makedirs(output_dir, exist_ok=True)


color_map = {
    'cs.CV': '#1f77b4',
    'cs.CL': '#ff7f0e',
    'cs.LG': '#2ca02c'
}


with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

quarters = sorted(data.keys())
categories = ['cs.CV', 'cs.CL', 'cs.LG']


def get_xticks(quarters):
    ticks, labels = [], []
    for q in quarters:
        if q.endswith("Q1") or q == "2025Q1":
            ticks.append(q)
            labels.append(q[:4])
    return ticks, labels

def get_best_legend_loc(x_vals, y_vals):
    quadrants = defaultdict(int)
    x_mid = len(x_vals) // 2
    y_all = [y for series in y_vals for y in series]
    if not y_all:
        return "best"
    y_mid = (max(y_all) + min(y_all)) / 2
    for ys in y_vals:
        for i, y in enumerate(ys):
            if i < x_mid and y >= y_mid:
                quadrants["upper left"] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants["upper right"] += 1
            elif i < x_mid and y < y_mid:
                quadrants["lower left"] += 1
            else:
                quadrants["lower right"] += 1
    return min(quadrants, key=quadrants.get)

xticks, xtick_labels = get_xticks(quarters)


naming_patterns = list(next(iter(data.values()))[categories[0]].keys())
print(f"Found {len(naming_patterns)} naming patterns")


for pattern in naming_patterns:
    plt.figure(figsize=(3.5, 2.5))
    legend_y_vals = []

    for cat in categories:
        means = [data[q][cat][pattern]["mean"] for q in quarters]
        legend_y_vals.append(means)
        plt.plot(
            quarters, means,
            label=cat,
            color=color_map[cat],
            linestyle='-',
            linewidth=2
        )


    all_y = [v for series in legend_y_vals for v in series if not np.isnan(v)]
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), y_max + margin)

    plt.ylabel("Ratio", fontsize=10)
    plt.xticks(xticks, xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)


    best_loc = get_best_legend_loc(quarters, legend_y_vals)
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
        facecolor="white",
        framealpha=1,
        labelspacing=0.2,
    )


    if "2023Q1" in quarters:
        idx = quarters.index("2023Q1")
        plt.axvline(x=quarters[idx], color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()


    out_path = os.path.join(output_dir, f"{pattern}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ All plots saved to: {output_dir}")
