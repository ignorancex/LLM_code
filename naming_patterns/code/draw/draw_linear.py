import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    y_mid = (max(y_all) + min(y_all)) / 2

    for ys in y_vals:
        for i, y in enumerate(ys):
            quadrants['upper left'] += 1
            # if i < x_mid and y >= y_mid:
            #     quadrants['upper left'] += 1
            # elif i >= x_mid and y >= y_mid:
            #     quadrants['upper right'] += 1
            # elif i < x_mid and y < y_mid:
            #     quadrants['lower left'] += 1
            # else:
            #     quadrants['lower right'] += 1

    return min(quadrants, key=quadrants.get)

def fit_and_plot(x_idx, y_vals, category):
    x = np.array(x_idx)
    y = np.array(y_vals)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    poly = np.poly1d(coef)

    color = "blue" if category == "cs" else "red"
    plt.plot(x[mask], poly(x[mask]),
             linestyle="--", linewidth=1.5, color=color, alpha=0.9)

def plot_pattern_trend(data, quarters, patterns, output_dir, label_prefix, colors, xticks, xtick_labels):
    os.makedirs(output_dir, exist_ok=True)

    stage1 = [q for q in quarters if "2020Q1" <= q <= "2023Q1"]
    stage2 = [q for q in quarters if "2023Q2" <= q <= "2025Q3"]

    for pattern in tqdm(patterns, desc=f"Plotting {label_prefix.title()} Names"):
        plt.figure(figsize=(3.5, 2.5))
        legend_y_vals = []

        for category in colors:
            y = [data.get(q, {}).get(category, {}).get(pattern, np.nan) for q in quarters]
            legend_y_vals.append(y)
            plt.plot(quarters, y, linestyle='-', label=category,
                     linewidth=2, color=colors[category])

            x_idx1 = [i for i, q in enumerate(quarters) if q in stage1]
            y1 = [y[i] for i in x_idx1]
            fit_and_plot(x_idx1, y1, category)

            x_idx2 = [i for i, q in enumerate(quarters) if q in stage2]
            y2 = [y[i] for i in x_idx2]
            fit_and_plot(x_idx2, y2, category)

        all_y = [v for series in legend_y_vals for v in series if not np.isnan(v)]
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), min(1, y_max + margin) if y_max + margin > 0 else 0.05)

        plt.ylabel("Proportion", fontsize=10)
        plt.xticks(xticks, xtick_labels, fontsize=10) 
        plt.yticks(fontsize=10)
        plt.grid(False)

        best_loc = get_best_legend_loc(quarters, legend_y_vals)
        plt.legend(fontsize=9, loc=best_loc, frameon=True,
                   facecolor='white', framealpha=1, labelspacing=0.2,
                  )

        if "2023Q1" in quarters:
            idx = quarters.index("2023Q1")
            plt.axvline(x=quarters[idx], color="gray", linestyle="--", linewidth=1)


        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{label_prefix}_{pattern}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_naming_trends(function_file, variable_file, filename_file, output_dir):
    func_data = load_json(function_file)
    var_data = load_json(variable_file)
    file_data = load_json(filename_file)

    quarters = sorted(func_data.keys())
    colors = {"cs": "#1f77b4", "non_cs": "#ff7f0e"} 
    xticks, xtick_labels = get_xticks(quarters)

    example_pattern_set = list(next(iter(next(iter(func_data.values())).values())).keys())

    plot_pattern_trend(func_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "function"), "function", colors, xticks, xtick_labels)
    plot_pattern_trend(var_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "variable"), "variable", colors, xticks, xtick_labels)
    plot_pattern_trend(file_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "filename"), "filename", colors, xticks, xtick_labels)

    print(f"\n🎨 All plots saved in subdirectories of {output_dir}")

if __name__ == "__main__":
    lang = "cpp"
    base_dir = f"naming_patterns/github_result/naming_patterns_{lang}"
    plot_naming_trends(
        function_file=os.path.join(base_dir, "naming_patterns_function.json"),
        variable_file=os.path.join(base_dir, "naming_patterns_variable.json"),
        filename_file=os.path.join(base_dir, "naming_patterns_filename.json"),
        output_dir=os.path.join(base_dir, f"plots_{lang}_linear_1")
    )
