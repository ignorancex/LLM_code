import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_naming_trends(function_file, variable_file, filename_file, output_dir):
    with open(function_file, 'r', encoding='utf-8') as f:
        func_data = json.load(f)
    with open(variable_file, 'r', encoding='utf-8') as f:
        var_data = json.load(f)
    with open(filename_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    quarters = sorted(func_data.keys())
    categories = ['cs', 'non_cs']
    func_plot_dir = os.path.join(output_dir, 'function')
    var_plot_dir = os.path.join(output_dir, 'variable')
    file_plot_dir = os.path.join(output_dir, 'filename')
    os.makedirs(func_plot_dir, exist_ok=True)
    os.makedirs(var_plot_dir, exist_ok=True)
    os.makedirs(file_plot_dir, exist_ok=True)
    example_quarter = next(iter(func_data.values()))
    example_category = next(iter(example_quarter.values()))
    patterns = list(example_category.keys())
    custom_xticks = []
    custom_xtick_labels = []
    for q in quarters:
        if q.endswith('Q1') or q == '2025Q1':
            if q == '2025Q1':
                custom_xtick_labels.append('2025Q1')
            else:
                custom_xtick_labels.append(q[:4])
            custom_xticks.append(q)
    var_colors = {'cs': '#4589c8ff', 'non_cs': '#ee7c7aff'}
    func_colors = file_colors = var_colors
    for pattern in tqdm(patterns, desc='Plotting Function Names'):
        plt.figure(figsize=(3.5, 2.5))
        for cat in categories:
            y = []
            for quarter in quarters:
                y.append(func_data[quarter].get(cat, {}).get(pattern, 0))
            plt.plot(quarters, y, marker='x', linestyle='--', label=f'{cat}', markersize=4, linewidth=2, color=func_colors[cat])
        all_y = []
        for quarter in quarters:
            for cat in categories:
                all_y.append(func_data[quarter].get(cat, {}).get(pattern, 0))
        y_min = min(all_y)
        y_max = max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), min(1, y_max + margin) if y_max + margin > 0 else 0.05)
        plt.title(f'{lang.capitalize()} Function - {pattern}', fontsize=10)
        plt.ylabel('Proportion', fontsize=9)
        plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(False)
        plt.subplots_adjust(bottom=0.5)
        plt.legend(fontsize=9, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, columnspacing=8)
        plt.tight_layout()
        save_path = os.path.join(func_plot_dir, f'function_{pattern}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.close()
    for pattern in tqdm(patterns, desc='Plotting Variable Names'):
        plt.figure(figsize=(3.5, 2.5))
        for cat in categories:
            y = []
            for quarter in quarters:
                y.append(var_data[quarter].get(cat, {}).get(pattern, 0))
            plt.plot(quarters, y, marker='x', linestyle='--', label=f'{cat}', markersize=4, linewidth=2, color=var_colors[cat])
        all_y = []
        for quarter in quarters:
            for cat in categories:
                all_y.append(var_data[quarter].get(cat, {}).get(pattern, 0))
        y_min = min(all_y)
        y_max = max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), min(1, y_max + margin) if y_max + margin > 0 else 0.05)
        plt.title(f'{lang.capitalize()} Variable - {pattern}', fontsize=10)
        plt.ylabel('Proportion', fontsize=9)
        plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(False)
        plt.subplots_adjust(bottom=0.5)
        plt.legend(fontsize=9, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, columnspacing=8)
        plt.tight_layout()
        save_path = os.path.join(var_plot_dir, f'variable_{pattern}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.close()
    for pattern in tqdm(patterns, desc='Plotting File Names'):
        plt.figure(figsize=(3.5, 2.5))
        for cat in categories:
            y = []
            for quarter in quarters:
                y.append(file_data[quarter].get(cat, {}).get(pattern, 0))
            plt.plot(quarters, y, marker='x', linestyle='--', label=f'{cat}', markersize=4, linewidth=2, color=file_colors[cat])
        all_y = []
        for quarter in quarters:
            for cat in categories:
                all_y.append(file_data[quarter].get(cat, {}).get(pattern, 0))
        y_min = min(all_y)
        y_max = max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), min(1, y_max + margin) if y_max + margin > 0 else 0.05)
        plt.title(f'{lang.capitalize()} File - {pattern}', fontsize=10)
        plt.ylabel('Proportion', fontsize=9)
        plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(False)
        plt.subplots_adjust(bottom=0.5)
        plt.legend(fontsize=9, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, columnspacing=8)
        plt.tight_layout()
        save_path = os.path.join(file_plot_dir, f'filename_{pattern}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.close()
lang = 'cpp'
if __name__ == '__main__':
    function_json = f'LLM_code/arxiv_result/naming_patterns_{lang}/naming_patterns_function.json'
    variable_json = f'LLM_code/arxiv_result/naming_patterns_{lang}/naming_patterns_variable.json'
    filename_json = f'LLM_code/arxiv_result/naming_patterns_{lang}/naming_patterns_filename.json'
    output_plot_dir = f'LLM_code/arxiv_result/naming_patterns_{lang}/plots_{lang}'
    plot_naming_trends(function_json, variable_json, filename_json, output_plot_dir)