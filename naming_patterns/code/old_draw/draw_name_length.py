import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def get_best_legend_corner(x_vals, y_vals_list):

    quadrants = defaultdict(float)
    n = len(x_vals)
    x_mid = n / 2
    all_y = [y for series in y_vals_list for y in series]
    y_mid = (max(all_y) + min(all_y)) / 2

    for series in y_vals_list:
        for i, y in enumerate(series):
            if i < x_mid and y >= y_mid:
                quadrants['upper left'] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants['upper right'] += 1
            elif i < x_mid and y < y_mid:
                quadrants['lower left'] += 1
            else:
                quadrants['lower right'] += 1

    return min(quadrants, key=quadrants.get)

def plot_avg_lengths(json_path, output_dir, metric_key, ylabel):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    quarters = sorted(data.keys())
    cs_vals    = [data[q]['cs'][metric_key]    for q in quarters]
    noncs_vals = [data[q]['non_cs'][metric_key] for q in quarters]


    custom_xticks = []
    custom_xtick_labels = []
    for q in quarters:
        if q.endswith('Q1') or q == '2025Q1':
            custom_xticks.append(q)
            custom_xtick_labels.append(q[:4])

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(3.5, 2.5))
    plt.plot(quarters, cs_vals,
             marker='x', linestyle='--', linewidth=2, markersize=4,
             label='cs', color='#4589c8ff')
    plt.plot(quarters, noncs_vals,
             marker='x', linestyle='--', linewidth=2, markersize=4,
             label='non-cs', color='#ee7c7aff')

    all_y = cs_vals + noncs_vals
    if all_y and max(all_y) != min(all_y):
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1)

    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    legend_loc = get_best_legend_corner(quarters, [cs_vals, noncs_vals])
    plt.legend(
        fontsize=10,
        loc=legend_loc,
        frameon=True,
        facecolor='white',
        framealpha=1,
        labelspacing=0.2,
        handletextpad=0.6
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{metric_key}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {metric_key} plot saved to: {save_path}")

if __name__ == '__main__':
    lang = "python"
    json_path = f'LLM_code/arxiv_result/naming_patterns_{lang}/average_lengths_cs_split.json'
    output_dir = f'LLM_code/arxiv_result/naming_patterns_{lang}/plots_{lang}'
    metrics_info = {
        'avg_func_len': 'Function Name Length',
        'avg_var_len':  'Variable Name Length',
        'avg_file_len': 'File Name Length'
    }
    for metric_key, ylabel in metrics_info.items():
        plot_avg_lengths(json_path, output_dir, metric_key, ylabel)
