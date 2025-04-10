import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 全局配置
method = "by_pub"
targets = ["functions", "variables", "file_name"]
input_dir = f"LLM_code/output_by_quarter/{method}"
output_dir = f"LLM_code/output_by_quarter/{method}/figures"

# 需要绘制的命名风格列表
# naming_styles_to_plot = [
#     "camelCase", "snake_case", "PascalCase",
#     "UPPER_SNAKE_CASE", "lowercase", "UPPERCASE",
#     "single_letter", "endsWithDigits", "Other"
# ]
naming_styles_to_plot = [
    "avg_name_length"
]

# 读取各 CSV 文件并存入字典 dfs
dfs = {}
for target in targets:
    input_file = os.path.join(input_dir, f"naming_patterns_{target}.csv")
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        dfs[target] = df
    else:
        print(f"未找到文件: {input_file}")

# 1. 收集所有可能出现的季度
all_quarters = set()
for df in dfs.values():
    all_quarters.update(df["quarter"].unique())

# 2. 将季度进行排序
#    这里使用简单的字符串排序，如果跨度年份多，可按需求改为更精确的排序
sorted_quarters = sorted(all_quarters)

# 3. 创建季度 -> 整数下标的映射
quarter_to_idx = {q: i for i, q in enumerate(sorted_quarters)}

# 4. 提前找出只用于显示刻度的季度位置（每年第一个季度 Q1）
x_locs = []
x_labels = []
for q in sorted_quarters:
    if q.endswith("Q1"):
        x_locs.append(quarter_to_idx[q])
        x_labels.append(q)

# 5. 为每种命名风格依次绘图
for style in tqdm(naming_styles_to_plot, desc="生成命名风格趋势图"):
    # 5.1 设置图表尺寸，使之适合双列论文单列宽度
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # 5.2 在同一张图上绘制不同 target 的趋势
    for target, df in dfs.items():
        if style in df.columns:
            # 根据 sorted_quarters 的顺序构建 y 值
            y_values = []
            for q in sorted_quarters:
                row = df.loc[df["quarter"] == q]
                if len(row) == 1:
                    y_values.append(row[style].values[0])
                else:
                    # 缺失数据可填 None 或 0
                    y_values.append(None)

            x_indices = range(len(sorted_quarters))
            # 模仿示例的绘制风格
            ax.plot(
                x_indices,
                y_values,
                marker='o',
                linewidth=2,
                markersize=4,
                label=target
            )

    # 5.3 设置 x 轴：仅显示每年第一个季度
    ax.set_xticks(x_locs)
    ax.set_xticklabels(x_labels, fontsize=7.5)  # 与示例中相同的字体大小

    # 5.4 设置 y 轴刻度字体
    ax.tick_params(axis='y', labelsize=7.5)

    # 5.5 设置标题与 y 轴标签
    # 标题示例：可根据需要自定义
    ax.set_title(f"Trends of '{style}' Naming Patterns", fontsize=10, pad = 20)
    ax.set_title(f"Trends of Name Length", fontsize=10, pad = 20)
    ax.set_ylabel("Character Count", fontsize=9)

    # 5.6 去掉网格（根据需求是否保留）
    ax.grid(False)

# 图例在标题下方，带边框
    ax.legend(
        fontsize=7.5,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(dfs),
        frameon=True
    )

# 留出顶部空间，让标题 + 图例不被压缩
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    # 5.9 保存图像
    fig_path = os.path.join(output_dir, f"{style}_trend.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图像已保存: {fig_path}")
