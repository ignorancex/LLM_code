import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设定通用配置
method = "by_file"
targets = ["functions", "variables", "comments_words", "file_name"]
input_dir = f"LLM_code/output_by_quarter/{method}"
output_dir = input_dir

# 选择要绘制的命名方式
# naming_styles_to_plot = [
#     "camelCase", "snake_case", "PascalCase",
#     "UPPER_SNAKE_CASE", "lowercase", "UPPERCASE",
#     "single_letter", "endsWithDigits", "Other"
# ]

naming_styles_to_plot = [
    "lowercase"
]

# 逐个处理 target
for target in tqdm(targets, desc="绘图中"):
    input_file = os.path.join(input_dir, f"naming_patterns_{target}.csv")
    if not os.path.exists(input_file):
        print(f"文件不存在：{input_file}")
        continue

    df = pd.read_csv(input_file)

    # 设置 x 轴为季度
    x = df["quarter"]
    
    # 绘图
    plt.figure(figsize=(14, 7))
    for pattern in naming_styles_to_plot:
        if pattern in df.columns:
            plt.plot(x, df[pattern], label=pattern, linewidth=2)

    plt.xticks(rotation=45)
    plt.xlabel("Quarter")
    plt.ylabel("Count")
    plt.title(f"Naming Pattern Trends Over Quarters: {target}")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()

    # 保存图像
    fig_path = os.path.join(output_dir, f"{target}_naming_trend.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"图已保存: {fig_path}")
