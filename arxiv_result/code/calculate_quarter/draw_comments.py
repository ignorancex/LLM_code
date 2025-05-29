import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def get_best_legend_loc(x_vals, y_vals):
    """自动选择图例位置：分4个象限，选数据点最少的一个"""
    quadrants = defaultdict(int)
    x_mid = len(x_vals) // 2
    y_all = [y for series in y_vals for y in series]
    y_mid = (max(y_all) + min(y_all)) / 2

    for ys in y_vals:
        for i, y in enumerate(ys):
            if i < x_mid and y >= y_mid:
                quadrants['upper left'] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants['upper right'] += 1
            elif i < x_mid and y < y_mid:
                quadrants['lower left'] += 1
            else:
                quadrants['lower right'] += 1

    return min(quadrants, key=quadrants.get)

def plot_comment_ratio_from_csv(comment_csv_file, output_dir, lang="python"):
    # === 加载 CSV 数据 ===
    df = pd.read_csv(comment_csv_file)
    df = df.sort_values("Quarter")

    quarters = df["Quarter"].tolist()
    cs_ratios = df["CS_Comment_Ratio"].tolist()
    noncs_ratios = df["NonCS_Comment_Ratio"].tolist()

    # 自定义 xticks（显示每年）
    custom_xticks = []
    custom_xtick_labels = []
    for q in quarters:
        if q.endswith("Q1") or q == "2025_Q1":
            custom_xticks.append(q)
            custom_xtick_labels.append(q[:4])

    colors = {"cs": "#4589c8ff", "non_cs": "#ee7c7aff"}
    os.makedirs(output_dir, exist_ok=True)

    # === 绘图 ===
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(quarters, cs_ratios, marker='x', linestyle='--', linewidth=2, markersize=4,
             label="cs", color=colors["cs"])
    plt.plot(quarters, noncs_ratios, marker='x', linestyle='--', linewidth=2, markersize=4,
             label="non_cs", color=colors["non_cs"])

    all_y = cs_ratios + noncs_ratios
    y_min, y_max = min(all_y), max(all_y)
    margin = (y_max - y_min) * 0.1
    plt.ylim(
        max(0, y_min - margin),
        min(1, y_max + margin) if y_max + margin > 0 else 0.05
    )

    # === 移除标题 ===
    # plt.title(...)

    plt.ylabel("Comment Ratio", fontsize=10)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)

    # === 自动图例位置（不遮挡）===
    best_loc = get_best_legend_loc(quarters, [cs_ratios, noncs_ratios])
    plt.legend(
        fontsize=10,
        loc=best_loc,
        frameon=True,
        facecolor='white',
        framealpha=1,
        labelspacing=0.2
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, "comment_ratio.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comment ratio plot saved to: {save_path}")

# === 主程序调用 ===
if __name__ == "__main__":
    lang = "python"
    comment_csv_file = f"LLM_code/arxiv_result/comments/comment_ratio_{lang}_by_group.csv"
    output_dir = f"LLM_code/arxiv_result/naming_patterns_{lang}/plots_{lang}"
    plot_comment_ratio_from_csv(comment_csv_file, output_dir, lang)
