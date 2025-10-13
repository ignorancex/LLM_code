import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_from_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    df = df.sort_values('quarter')
    quarters = df['quarter'].unique().tolist()
    metrics = [col for col in df.columns if col not in ['quarter', 'category']]
    custom_xticks = []
    custom_xtick_labels = []
    for q in quarters:
        if q.endswith('Q1') or q == '2025Q1':
            custom_xticks.append(q)
            custom_xtick_labels.append('2025Q1' if q == '2025Q1' else q[:4])
    colors = {'cs': '#4589c8ff', 'non_cs': '#ee7c7aff'}
    os.makedirs(output_dir, exist_ok=True)
    for metric in metrics:
        cs_vals = df[df['category'] == 'cs'][metric].tolist()
        noncs_vals = df[df['category'] == 'non_cs'][metric].tolist()
        plt.figure(figsize=(3.5, 2.5))
        plt.plot(quarters, cs_vals, marker='x', linestyle='--', linewidth=2, markersize=4, label='cs', color=colors['cs'])
        plt.plot(quarters, noncs_vals, marker='x', linestyle='--', linewidth=2, markersize=4, label='non_cs', color=colors['non_cs'])
        all_y = cs_vals + noncs_vals
        (y_min, y_max) = (min(all_y), max(all_y))
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
        plt.title(metric.replace('_', ' ').title(), fontsize=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=9)
        plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(False)
        plt.subplots_adjust(bottom=0.28)
        plt.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, columnspacing=8, handletextpad=0.6)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{metric}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.close()
if __name__ == '__main__':
    csv_path = 'github_py_metrics_by_category.csv'
    output_dir = 'LLM_code/arxiv_result/plots_metrics'
    plot_metrics_from_csv(csv_path, output_dir)