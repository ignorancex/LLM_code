import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_path = 'complexity_maintainability/result/subset/subset_metrics.csv'
df = pd.read_csv(csv_path)

df['language'] = df['model'].apply(lambda x: 'Python' if x.endswith('_py') else 'C/C++')
df['model_base'] = df['model'].str.replace('_py', '', regex=False).str.replace('_cpp', '', regex=False)

model_name_map = {
    'claude_standard': 'Claude-3.5-Sonnet',
    'deepseek_chat': 'DeepSeek-V3',
    'deepseek_reasoner': 'DeepSeek-R1',
    'gemma': 'Gemma-3-27B',
    'gpt': 'GPT-4o-mini',
    'llama': 'Llama 3.3',
    'qwen_14b': 'Qwen3-14B',
    'qwen_32b': 'Qwen3-32B',
    'qwen_4b': 'Qwen3-4B',
    'qwen_8b': 'Qwen3-8B',
    'qwen_coder': 'Qwen2.5-32B',
    'human': 'Human'
}
df['model_base'] = df['model_base'].map(model_name_map)

# === 绘制顺序 ===
ordered_models = [
    'Human',
    'Qwen2.5-32B',
    'Qwen3-4B',
    'Qwen3-8B',
    'Qwen3-14B',
    'Qwen3-32B',
    'DeepSeek-V3',
    'DeepSeek-R1',
    'Gemma-3-27B',
    'Llama 3.3',
    'GPT-4o-mini',
    'Claude-3.5-Sonnet'
]

metrics = ['mi_custom', 'cyclomatic', 'difficulty', 'bugs']
metric_names = {
    'mi_custom': 'Custom Maintainability Index',
    'cyclomatic': 'Cyclomatic Complexity',
    'difficulty': 'Halstead Difficulty',
    'bugs': 'Halstead Estimated Bugs'
}

output_dir = 'errorbar_charts'
os.makedirs(output_dir, exist_ok=True)

color_map = {'Python': '#1f77b4', 'C/C++': '#ff7f0e'}

for metric in metrics:
    mean_df = df[df['type'] == 'mean']
    std_df = df[df['type'] == 'var']  

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(ordered_models))
    width = 0.35


    bar_positions = []  
    for i, lang in enumerate(['Python', 'C/C++']):
        means, stds = [], []
        for model in ordered_models:
            mean_val = mean_df[(mean_df['model_base'] == model) & (mean_df['language'] == lang)][metric]
            std_val = std_df[(std_df['model_base'] == model) & (std_df['language'] == lang)][metric]
            means.append(mean_val.values[0] if not mean_val.empty else np.nan)
            stds.append(std_val.values[0] if not std_val.empty else np.nan)

        positions = x + (i - 0.5) * width
        bar_positions.append(positions)

        bars = ax.bar(
            positions,
            means,
            width=width,
            label=lang,
            color=color_map[lang],
            alpha=0.6,                
            edgecolor='black',
            linewidth=0.8
        )


        for (px, m, s) in zip(positions, means, stds):
            if np.isnan(m) or np.isnan(s):
                continue

            ax.plot([px, px], [m, m + s], color='black', linewidth=0.8)  

            cap_width = width * 0.25
            ax.plot([px - cap_width/2, px + cap_width/2], [m + s, m + s], color='black', linewidth=0.8)

    for lang, color in [('Python', 'blue'), ('C/C++', 'red')]:
        human_val = mean_df[(mean_df['model_base'] == 'Human') & (mean_df['language'] == lang)][metric]
        if not human_val.empty:
            y = human_val.values[0]
            ax.axhline(y, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_models, rotation=35, ha='right')
    ax.set_ylabel(metric_names.get(metric, metric))

    if metric == 'mi_custom':
        ax.legend(frameon=True, loc='upper left', fontsize=10)
    else:
        ax.legend(frameon=True, loc='upper right', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_bar.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

