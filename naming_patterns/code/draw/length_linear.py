import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_best_legend_corner(x_vals, y_vals_list):
    """自动选择数据点最少重叠的角落"""
    quadrants = defaultdict(float)
    n = len(x_vals)
    x_mid = n / 2
    all_y = [y for series in y_vals_list for y in series if not np.isnan(y)]
    y_mid = (max(all_y) + min(all_y)) / 2 if all_y else 0.5

    for series in y_vals_list:
        for i, y in enumerate(series):
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

    return min(quadrants, key=quadrants.get) if quadrants else 'upper right'

def fit_and_plot(x_idx, y_vals, color):
    """对一个阶段的数据做线性拟合并画虚线"""
    x = np.array(x_idx)
    y = np.array(y_vals)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    poly = np.poly1d(coef)
    plt.plot(x[mask], poly(x[mask]), linestyle="--", linewidth=1.5, color=color, alpha=0.9)

def plot_avg_lengths(json_path, output_dir, metric_key, ylabel):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    quarters = sorted(data.keys())
    cs_vals    = [data[q]['cs'][metric_key] if 'cs' in data[q] else np.nan for q in quarters]
    noncs_vals = [data[q]['non_cs'][metric_key] if 'non_cs' in data[q] else np.nan for q in quarters]

    # xticks: 每年第一个季度
    custom_xticks, custom_xtick_labels = [], []
    for q in quarters:
        if q.endswith('Q1') or q == '2025Q1':
            custom_xticks.append(q)
            custom_xtick_labels.append(q[:4])

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(3.5, 2.5))

    # 主曲线
    plt.plot(quarters, cs_vals, linestyle='-', linewidth=2, label='cs', color='#1f77b4')
    plt.plot(quarters, noncs_vals, linestyle='-', linewidth=2, label='non-cs', color='#ff7f0e')

    # === 拟合两个阶段 ===
    stage1 = [q for q in quarters if "2020Q1" <= q <= "2023Q1"]
    stage2 = [q for q in quarters if "2023Q2" <= q <= "2025Q3"]

    x_idx1 = [i for i, q in enumerate(quarters) if q in stage1]
    y1_cs = [cs_vals[i] for i in x_idx1]
    y1_noncs = [noncs_vals[i] for i in x_idx1]
    fit_and_plot(x_idx1, y1_cs, "blue")
    fit_and_plot(x_idx1, y1_noncs, "red")

    x_idx2 = [i for i, q in enumerate(quarters) if q in stage2]
    y2_cs = [cs_vals[i] for i in x_idx2]
    y2_noncs = [noncs_vals[i] for i in x_idx2]
    fit_and_plot(x_idx2, y2_cs, "blue")
    fit_and_plot(x_idx2, y2_noncs, "red")

    # y 轴范围
    all_y = [v for v in cs_vals + noncs_vals if not np.isnan(v)]
    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1)

    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    # 图例：自动贴边
    legend_loc = get_best_legend_corner(quarters, [cs_vals, noncs_vals])
    anchor_map = {
        "upper left": (0.0, 1.0),
        "upper right": (1.0, 1.0),
        "lower left": (0.0, 0.0),
        "lower right": (1.0, 0.0),
    }
    plt.legend(
        fontsize=10,
        loc=legend_loc,
        bbox_to_anchor=anchor_map[legend_loc],
        frameon=True,
        facecolor='white',
        framealpha=1,
        labelspacing=0.2
    )

    # === 在 2023Q1 加竖直黑虚线 ===
    if "2023Q1" in quarters:
        idx = quarters.index("2023Q1")
        plt.axvline(x=quarters[idx], color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{metric_key}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {metric_key} plot saved to: {save_path}")

if __name__ == '__main__':
    lang = "cpp"
    json_path = f'naming_patterns/github_result/naming_patterns_{lang}/average_lengths_cs_split.json'
    output_dir = f'naming_patterns/github_result/naming_patterns_{lang}/plots_{lang}_linear'
    metrics_info = {
        'avg_func_len': 'Function Name Length',
        'avg_var_len':  'Variable Name Length',
        'avg_file_len': 'File Name Length'
    }
    for metric_key, ylabel in metrics_info.items():
        plot_avg_lengths(json_path, output_dir, metric_key, ylabel)
