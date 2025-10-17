import json
import matplotlib.pyplot as plt
import numpy as np
import os

patterns = [
    'single_letter', 'lowercase', 'UPPERCASE',
    'camelCase', 'snake_case', 'PascalCase',
    'endsWithDigits', 'Other', 'avg_length'
]
models = ['GPT', 'Gemini', 'DS', 'Llama', 'Qw', 'Gemma']
legend_labels = {
    'ref': 'LLM-Revised',
    'ans': 'LLM-Generated'
}
colors = {
    'ac':  '#c8c8c8',
    'ref': '#ffde7b',
    'ans': '#6ad1a3'
}

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

    data = {}
    for lang, path in json_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            data[lang] = json.load(f)[lang]

    for lang in json_paths:
        output_dir = os.path.join(output_dir_base, f'plots_{lang}')
        os.makedirs(output_dir, exist_ok=True)

        for pattern in patterns:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            bar_width = 0.5  # human bar width
            sub_width = 0.3  # each model bar width

            x = np.arange(len(models) + 1)
            human_pos = x[0]
            model_positions = x[1:]

            first_model = models[0]
            ac_val = data[lang][first_model]['ac'].get(pattern, 0)
            ax.bar(human_pos, ac_val,
                   width=bar_width,
                   color=colors['ac'])

            for i, model in enumerate(models):
                pos = model_positions[i]
                ref_val = data[lang][model]['ref'].get(pattern, 0)
                ans_val = data[lang][model]['ans'].get(pattern, 0)
                ax.bar(pos - sub_width/2, ref_val,
                       width=sub_width,
                       label=legend_labels['ref'] if i == 0 else "",
                       color=colors['ref'])
                ax.bar(pos + sub_width/2, ans_val,
                       width=sub_width,
                       label=legend_labels['ans'] if i == 0 else "",
                       color=colors['ans'])

            xticks = x
            xtick_labels = ['Human'] + models
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, fontsize=8)

            max_val = max(
                ac_val,
                *(data[lang][m]['ref'].get(pattern, 0) for m in models),
                *(data[lang][m]['ans'].get(pattern, 0) for m in models)
            )
            if pattern == 'avg_length':
                ax.set_ylabel('Average Name Length', fontsize=9)
                ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
            else:
                ax.set_ylabel('Proportion', fontsize=9)
                ax.set_ylim(0, min(1.0, max_val * 1.25))

            plt.yticks(fontsize=8)

            ax.legend(fontsize=8, loc='upper right', frameon=True, labelspacing=0.2)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{pattern}.pdf')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Completed {plot_type} for {lang}, saved to {output_dir}/")


json_function = {
    'cpp':    'LLM_code/codeforces/simulation/result/function_naming_all_models_cpp.json',
    'python': 'LLM_code/codeforces/simulation/result/function_naming_all_models_python.json'
}
json_variable = {
    'cpp':    'LLM_code/codeforces/simulation/result/variable_naming_all_models_cpp.json',
    'python': 'LLM_code/codeforces/simulation/result/variable_naming_all_models_python.json'
}

plot_naming_patterns(
    json_function,
    output_dir_base='LLM_code/codeforces/simulation/result/funcs',
    plot_type='funcs'
)
plot_naming_patterns(
    json_variable,
    output_dir_base='LLM_code/codeforces/simulation/result/vars',
    plot_type='vars'
)
