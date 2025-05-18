import json
import matplotlib.pyplot as plt
import numpy as np
import os
models = ['deepseek_32b', 'gemma_27b', 'qwen_32b']
categories = ['ac', 'ref', 'ans']
legend_labels = {'ac': 'Human-Written', 'ref': 'LLM-Revised', 'ans': 'LLM-Generated'}
colors = {'ac': '#c8c8c8', 'ref': '#ffde7b', 'ans': '#6ad1a3'}

def plot_avg_name_length(json_path, output_dir_base, plot_type):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if plot_type == 'funcs':
        title_label = 'Function Name Length'
    elif plot_type == 'vars':
        title_label = 'Variable Name Length'
    else:
        raise ValueError("plot_type must be 'funcs' or 'vars'")
    for language in ['python', 'cpp']:
        output_dir = os.path.join(output_dir_base, f'plots_{language}')
        os.makedirs(output_dir, exist_ok=True)
        (fig, ax) = plt.subplots(figsize=(3.5, 2.5))
        bar_width = 0.2
        x = np.arange(len(models))
        max_value = 0
        for (idx, category) in enumerate(categories):
            values = []
            for model in models:
                try:
                    value = data[language][model][category]
                except KeyError:
                    value = 0
                values.append(value)
                max_value = max(max_value, value)
            ax.bar(x + idx * bar_width, values, width=bar_width, label=legend_labels[category], color=colors[category])
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel('Avg Length', fontsize=9)
        ax.set_title(f'{language.capitalize()} - {title_label}', fontsize=10)
        plt.yticks(fontsize=8)
        fig.subplots_adjust(bottom=0.22)
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, columnspacing=0.8)
        ax.set_ylim(0, max_value * 1.25)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{plot_type}_{language}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.close()
plot_avg_name_length(json_path='LLM_code/codeforces/name_length/avg_name_length_funcs.json', output_dir_base='LLM_code/codeforces/name_length', plot_type='funcs')
plot_avg_name_length(json_path='LLM_code/codeforces/name_length/avg_name_length_vars.json', output_dir_base='LLM_code/codeforces/name_length', plot_type='vars')