import os
import json
import matplotlib.pyplot as plt

def plot_avg_lengths(json_path, output_dir, metric_key, ylabel, title):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    quarters = sorted(data.keys())
    cs_vals = [data[q]['cs'][metric_key] for q in quarters]
    noncs_vals = [data[q]['non_cs'][metric_key] for q in quarters]
    custom_xticks = []
    custom_xtick_labels = []
    for q in quarters:
        if q.endswith('Q1') or q == '2025Q1':
            custom_xticks.append(q)
            custom_xtick_labels.append(q[:4] if q != '2025Q1' else '2025Q1')
    colors = {'cs': '#4589c8ff', 'non_cs': '#ee7c7aff'}
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(quarters, cs_vals, marker='x', linestyle='--', linewidth=2, markersize=4, label='cs', color=colors['cs'])
    plt.plot(quarters, noncs_vals, marker='x', linestyle='--', linewidth=2, markersize=4, label='non_cs', color=colors['non_cs'])
    all_y = cs_vals + noncs_vals
    if all_y and max(all_y) != min(all_y):
        (y_min, y_max) = (min(all_y), max(all_y))
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
    else:
        plt.ylim(0, 1)
    plt.title(title, fontsize=10)
    plt.ylabel(ylabel, fontsize=9)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(False)
    plt.subplots_adjust(bottom=0.28)
    plt.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, columnspacing=8, handletextpad=0.6)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{metric_key}.pdf')
    plt.savefig(save_path, dpi=300)
    plt.close()
if __name__ == '__main__':
    json_path = 'LLM_code/arxiv_result/naming_patterns_python/average_lengths_cs_split.json'
    output_dir = 'LLM_code/arxiv_result/naming_patterns_python/plots_python'
    metrics_info = {'avg_func_len': ('Function Name Length', 'Function Name Length'), 'avg_var_len': ('Variable Name Length', 'Variable Name Length'), 'avg_file_len': ('File Name Length', 'File Name Length')}
    for (metric_key, (ylabel, title)) in metrics_info.items():
        plot_avg_lengths(json_path, output_dir, metric_key, ylabel, title)