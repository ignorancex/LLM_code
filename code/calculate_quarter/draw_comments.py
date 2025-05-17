import os
import pandas as pd
import matplotlib.pyplot as plt

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
            if q == "2025_Q1":
                custom_xtick_labels.append("2025Q1")
            else:
                custom_xtick_labels.append(q[:4])
            custom_xticks.append(q)

    colors = {"cs": "#4589c8ff", "non_cs": "#ee7c7aff"}
    os.makedirs(output_dir, exist_ok=True)

    # === 绘图 ===
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(quarters, cs_ratios, marker='x', linestyle='--', linewidth=2, markersize=4, label="cs", color=colors["cs"])
    plt.plot(quarters, noncs_ratios, marker='x', linestyle='--', linewidth=2, markersize=4, label="non_cs", color=colors["non_cs"])

    all_y = cs_ratios + noncs_ratios
    y_min, y_max = min(all_y), max(all_y)
    margin = (y_max - y_min) * 0.1
    plt.ylim(
        max(0, y_min - margin),
        min(1, y_max + margin) if y_max + margin > 0 else 0.05
    )

    plt.title(f"{lang.capitalize()} - Comment Ratio", fontsize=10)
    plt.ylabel("Comment Ratio", fontsize=9)
    plt.xticks(custom_xticks, custom_xtick_labels, fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(False)

    # 图例在下方
    plt.subplots_adjust(bottom=0.28)
    plt.legend(
        fontsize=8,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
        columnspacing=8,
        handletextpad=0.6
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "comment_ratio.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Comment ratio plot saved to: {save_path}")

# === 主程序调用 ===
if __name__ == "__main__":
    lang = "python"  # 或 "python"
    comment_csv_file = f"LLM_code/arxiv_result/comments/comment_ratio_{lang}_by_group.csv"
    output_dir = f"LLM_code/arxiv_result/naming_patterns_{lang}/plots_{lang}"
    plot_comment_ratio_from_csv(comment_csv_file, output_dir, lang)
