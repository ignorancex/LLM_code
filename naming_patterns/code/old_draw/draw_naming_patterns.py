import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

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
    """
    自动选择图例放置位置，避免遮挡线条。
    将图分成4象限，选择点最少的象限。
    """
    quadrants = defaultdict(int)
    x_mid = len(x_vals) // 2
    y_all = [y for series in y_vals for y in series]
    y_mid = (max(y_all) + min(y_all)) / 2

    for ys in y_vals:
        for i, y in enumerate(ys):
            if i < x_mid and y >= y_mid:
                quadrants['upper left'] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants['upper right'] += 1
            elif i < x_mid and y < y_mid:
                quadrants['lower left'] += 1
            else:
                quadrants['lower right'] += 1

    return min(quadrants, key=quadrants.get)

def plot_pattern_trend(data, quarters, patterns, output_dir, label_prefix, colors, xticks, xtick_labels):
    os.makedirs(output_dir, exist_ok=True)

    for pattern in tqdm(patterns, desc=f"Plotting {label_prefix.title()} Names"):
        plt.figure(figsize=(3.5, 2.5))

        legend_y_vals = []
        for category in colors:
            y = [data.get(q, {}).get(category, {}).get(pattern, 0) for q in quarters]
            legend_y_vals.append(y)
            plt.plot(quarters, y, marker='x', linestyle='--', label=category,
                     markersize=4, linewidth=2, color=colors[category])

        all_y = [v for series in legend_y_vals for v in series]
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(
            max(0, y_min - margin),
            min(1, y_max + margin) if y_max + margin > 0 else 0.05
        )

        plt.ylabel("Proportion", fontsize=10)
        plt.xticks(xticks, xtick_labels, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(False)

        best_loc = get_best_legend_loc(quarters, legend_y_vals)
        plt.legend(
            fontsize=10,
            loc=best_loc,
            frameon=True,
            facecolor='white',
            framealpha=1,
            labelspacing=0.2,
        )

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{label_prefix}_{pattern}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_naming_trends(function_file, variable_file, filename_file, output_dir):
    func_data = load_json(function_file)
    var_data = load_json(variable_file)
    file_data = load_json(filename_file)

    quarters = sorted(func_data.keys())
    categories = ["cs", "non_cs"]
    colors = {"cs": "#4589c8ff", "non_cs": "#ee7c7aff"}
    xticks, xtick_labels = get_xticks(quarters)

    example_pattern_set = list(next(iter(next(iter(func_data.values())).values())).keys())

    plot_pattern_trend(func_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "function"), "function", colors, xticks, xtick_labels)
    plot_pattern_trend(var_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "variable"), "variable", colors, xticks, xtick_labels)
    plot_pattern_trend(file_data, quarters, example_pattern_set,
                       os.path.join(output_dir, "filename"), "filename", colors, xticks, xtick_labels)

    print(f"\n🎨 All plots saved in subdirectories of {output_dir}")

# === 主程序 ===
if __name__ == "__main__":
    lang = "python"
    base_dir = f"naming_patterns/github_result/naming_patterns_{lang}"
    plot_naming_trends(
        function_file=os.path.join(base_dir, "naming_patterns_function.json"),
        variable_file=os.path.join(base_dir, "naming_patterns_variable.json"),
        filename_file=os.path.join(base_dir, "naming_patterns_filename.json"),
        output_dir=os.path.join(base_dir, f"plots_{lang}_linear")
    )
