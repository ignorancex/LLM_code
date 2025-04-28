import json
import matplotlib.pyplot as plt
import numpy as np
import os

# === 1. 配置参数 ===
patterns = [
    "single_letter", "lowercase", "UPPERCASE",
    "camelCase", "snake_case", "PascalCase",
    "UPPER_SNAKE_CASE", "endsWithDigits", "Other"
]
models = ["deepseek_32b", "gemma_27b", "qwen_32b"]

# 顺序调整：ac -> ref -> ans
categories = ["ac", "ref", "ans"]

legend_labels = {
    "ac": "Human-Written",
    "ref": "LLM-Revised",
    "ans": "LLM-Generated"
}

# === 2. 定义绘制函数 ===
def plot_naming_patterns(json_path, output_dir_base, plot_type):
    """plot_type = 'funcs' or 'vars'"""

    if plot_type == 'vars':
        colors = {
            "ac": "#A09FA4",
            "ref": "#91BFFA",
            "ans": "#FEA0A0"
        }
        type_label = "Variables"  # 给标题用
    elif plot_type == 'funcs':
        colors = {
            "ac": "#A09FA4",
            "ref": "#ffde7b",
            "ans": "#6ad1a3"
        }
        type_label = "Functions"  # 给标题用
    else:
        raise ValueError("plot_type must be 'funcs' or 'vars'")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for language in ['python', 'cpp']:
        output_dir = os.path.join(output_dir_base, f'plots_{language}')
        os.makedirs(output_dir, exist_ok=True)

        for pattern in patterns:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))

            bar_width = 0.18
            x = np.arange(len(models))
            max_value = 0

            for idx, category in enumerate(categories):
                values = []
                for model in models:
                    try:
                        value = data[language][model][category].get(pattern, 0)
                    except KeyError:
                        value = 0
                    values.append(value)
                    max_value = max(max_value, value)

                ax.bar(x + idx * bar_width, values, width=bar_width, label=legend_labels[category], color=colors[category])

            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(models, fontsize=8)
            ax.set_ylabel('Proportion', fontsize=9)
            plt.yticks(fontsize=8)

            # === 修改标题，加上语言+函数/变量信息 ===
            ax.set_title(f'{pattern} ({language.upper()} {type_label})', fontsize=10)

            # 图例
            ax.legend(fontsize=7, loc='upper right', frameon=True)

            # 动态设置y轴上限
            upper_ylim = min(1.0, (max_value * 1.25))
            ax.set_ylim(0, upper_ylim)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{pattern}.png"), dpi=300)
            plt.close()

        print(f"已完成 {plot_type} {language} 的绘图，保存到 {output_dir}/")

# === 3. 执行：分别绘制函数名和变量名
plot_naming_patterns(
    json_path='LLM_code/codeforces/naming_pattern/naming_pattern_distribution_funcs.json',
    output_dir_base='LLM_code/codeforces/naming_pattern/funcs',
    plot_type='funcs'
)

plot_naming_patterns(
    json_path='LLM_code/codeforces/naming_pattern/naming_pattern_distribution_vars.json',
    output_dir_base='LLM_code/codeforces/naming_pattern/vars',
    plot_type='vars'
)
