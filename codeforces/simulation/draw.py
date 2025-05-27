import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# === 配置参数 ===
patterns = [
    'single_letter', 'lowercase', 'UPPERCASE',
    'camelCase', 'snake_case', 'PascalCase',
    'endsWithDigits', 'Other', 'avg_length'
]
models = ['DeepSeek', 'GPT', 'Gemini', 'Qwen', 'Gemma', 'Llama']
categories = ['ac', 'ref', 'ans']
legend_labels = {
    'ac': 'Human-Written',
    'ref': 'LLM-Revised',
    'ans': 'LLM-Generated'
}
colors = {
    'ac': '#c8c8c8',
    'ref': '#ffde7b',
    'ans': '#6ad1a3'
}

def get_best_top_corner(bar_groups):
    """
    在左上或右上选择空白更多的一侧。
    """
    left_sum = right_sum = 0.0
    n = len(models)
    mid = n / 2

    all_y = [y for group in bar_groups for y in group]
    y_mid = (max(all_y) + min(all_y)) / 2

    for group in bar_groups:
        for i, y in enumerate(group):
            if y >= y_mid:
                if i < mid:
                    left_sum += y
                else:
                    right_sum += y

    return 'upper right' if left_sum > right_sum else 'upper left'


def plot_naming_patterns(json_paths: dict, output_dir_base: str, plot_type: str):
    """
    json_paths: {'cpp': path_to_cpp_json, 'python': path_to_py_json}
    plot_type: 'funcs' or 'vars'
    """
    if plot_type == 'vars':
        type_label = 'Variables'
    elif plot_type == 'funcs':
        type_label = 'Functions'
    else:
        raise ValueError("plot_type must be 'funcs' or 'vars'")

    # 读取所有语言的数据
    data = {}
    for lang, path in json_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            data[lang] = json.load(f)[lang]

    # 每个语言分别绘制
    for lang in json_paths:
        output_dir = os.path.join(output_dir_base, f'plots_{lang}')
        os.makedirs(output_dir, exist_ok=True)

        for pattern in patterns:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            bar_width = 0.22
            x = np.arange(len(models))
            max_value = 0
            bar_groups = []

            # 收集柱状值
            for idx, category in enumerate(categories):
                values = []
                for model in models:
                    v = data[lang].get(model, {}).get(category, {}).get(pattern, 0)
                    values.append(v)
                    max_value = max(max_value, v)
                bar_groups.append(values)
                ax.bar(x + idx * bar_width, values,
                       width=bar_width,
                       label=legend_labels[category],
                       color=colors[category])

            # X轴
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(models, fontsize=8)
            # Y轴标签
            if pattern == 'avg_length':
                ax.set_ylabel('Average Name Length', fontsize=9)
                ax.set_ylim(0, max_value * 1.25 if max_value > 0 else 1)
            else:
                ax.set_ylabel('Proportion', fontsize=9)
                ax.set_ylim(0, min(1.0, max_value * 1.25))

            plt.yticks(fontsize=8)
            # 移除标题
            # ax.set_title(f'{lang.capitalize()} {type_label} - {pattern}')

            # 图例位置
            legend_loc = get_best_top_corner(bar_groups)
            ax.legend(
                fontsize=7,
                loc=legend_loc,
                frameon=True,
                facecolor='white',
                framealpha=1,
                ncol=1
            )

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{pattern}.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Completed {plot_type} for {lang}, saved to {output_dir}/")


# === 主程序：同时画出 C++ 和 Python 的 函数名和变量名 ===
json_function = {
    'cpp': 'LLM_code/codeforces/simulation/result/function_naming_all_models_cpp.json',
    'python': 'LLM_code/codeforces/simulation/result/function_naming_all_models_python.json'
}
json_variable = {
    'cpp': 'LLM_code/codeforces/simulation/result/variable_naming_all_models_cpp.json',
    'python': 'LLM_code/codeforces/simulation/result/variable_naming_all_models_python.json'
}

plot_naming_patterns(json_function,
    output_dir_base='LLM_code/codeforces/simulation/result/funcs',
    plot_type='funcs'
)
plot_naming_patterns(json_variable,
    output_dir_base='LLM_code/codeforces/simulation/result/vars',
    plot_type='vars'
)
