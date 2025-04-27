import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_naming_trends(function_file, variable_file, output_dir):
    with open(function_file, "r", encoding="utf-8") as f:
        func_data = json.load(f)
    with open(variable_file, "r", encoding="utf-8") as f:
        var_data = json.load(f)

    quarters = sorted(func_data.keys())
    categories = ["cs.LG", "cs.CV", "cs.CL", "other_cs", "non_cs"]

    func_plot_dir = os.path.join(output_dir, "function")
    var_plot_dir = os.path.join(output_dir, "variable")
    os.makedirs(func_plot_dir, exist_ok=True)
    os.makedirs(var_plot_dir, exist_ok=True)

    # 动态获取所有出现过的 pattern
    example_quarter = next(iter(func_data.values()))
    example_category = next(iter(example_quarter.values()))
    patterns = list(example_category.keys())

    for pattern in tqdm(patterns, desc="Plotting Function Names"):
        plt.figure(figsize=(10, 6))

        for cat in categories:
            y = []
            for quarter in quarters:
                y.append(func_data[quarter].get(cat, {}).get(pattern, 0))
            plt.plot(quarters, y, marker='o', linestyle='-', label=f"{cat}")

        # 动态设置纵轴
        all_y = []
        for quarter in quarters:
            for cat in categories:
                all_y.append(func_data[quarter].get(cat, {}).get(pattern, 0))
        y_min = min(all_y)
        y_max = max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(
            max(0, y_min - margin),
            min(1, y_max + margin) if y_max + margin > 0 else 0.05
        )

        plt.title(f"Function Names - {pattern}")
        plt.xlabel("Quarter")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        save_path = os.path.join(func_plot_dir, f"function_naming_trend_{pattern}.png")
        plt.savefig(save_path)
        plt.close()

    for pattern in tqdm(patterns, desc="Plotting Variable Names"):
        plt.figure(figsize=(10, 6))

        for cat in categories:
            y = []
            for quarter in quarters:
                y.append(var_data[quarter].get(cat, {}).get(pattern, 0))
            plt.plot(quarters, y, marker='s', linestyle='--', label=f"{cat}")

        # 动态设置纵轴
        all_y = []
        for quarter in quarters:
            for cat in categories:
                all_y.append(var_data[quarter].get(cat, {}).get(pattern, 0))
        y_min = min(all_y)
        y_max = max(all_y)
        margin = (y_max - y_min) * 0.1
        plt.ylim(
            max(0, y_min - margin),
            min(1, y_max + margin) if y_max + margin > 0 else 0.05
        )

        plt.title(f"Variable Names - {pattern}")
        plt.xlabel("Quarter")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        save_path = os.path.join(var_plot_dir, f"variable_naming_trend_{pattern}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"\n🎨 All plots saved in {func_plot_dir} and {var_plot_dir}")


# === 主程序 ===
if __name__ == "__main__":
    function_json = "LLM_code/naming_patterns_combined/naming_patterns_function_by_category.json"
    variable_json = "LLM_code/naming_patterns_combined/naming_patterns_variable_by_category.json"
    output_plot_dir = "LLM_code/naming_patterns_combined/plots"

    plot_naming_trends(function_json, variable_json, output_plot_dir)
