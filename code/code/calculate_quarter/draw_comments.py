import json
import matplotlib.pyplot as plt

# 1. 读取 JSON 文件
json_file = "LLM_code/output_by_quarter/by_mod/comment_ratio.json"  # 请替换为实际的 JSON 文件名
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 将季度按照键名排序（如 "2020_Q1", "2020_Q2", 等）
quarters = sorted(data.keys())

# 3. 依次计算「平均每个文件的注释数」与「平均注释比例」
avg_comments_per_file = []
avg_comment_density = []

for q in quarters:
    total_files = data[q]["total_py_files"]
    total_comments = data[q]["total_comments"]
    comment_density = data[q]["avg_comment_density"]
    
    # 平均每个文件的注释数
    if total_files != 0:
        avg_comments = total_comments / total_files
    else:
        avg_comments = 0
    
    avg_comments_per_file.append(avg_comments)
    avg_comment_density.append(comment_density)

# 4. 仅标注每年第一个季度（Q1）
x_indices = range(len(quarters))
x_locs = []
x_labels = []
for i, q in enumerate(quarters):
    if q.endswith("Q1"):
        x_locs.append(i)
        x_labels.append(q.replace("_", ""))

# 5. 设置图表尺寸与字体（与之前示例一致），创建左轴
fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

# （左轴）绘制「平均每个文件的注释数」，使用蓝色
line1 = ax1.plot(
    x_indices,
    avg_comments_per_file,
    marker='o',      
    linewidth=2,
    markersize=4,
    color='tab:blue',  # 第一条曲线颜色
    label="Avg Comments per File"
)

# 设置左轴标签与字体
ax1.set_ylabel("Avg Comments per File", fontsize=9)
ax1.tick_params(axis='x', labelsize=7)
ax1.tick_params(axis='y', labelsize=7)

# 只显示 Q1 作为 x 轴刻度
ax1.set_xticks(x_locs)
ax1.set_xticklabels(x_labels, fontsize=7)

# 设置标题、去除网格
ax1.set_title("Comments Trend", fontsize=10, pad = 17)
ax1.grid(False)

# （右轴）绘制「平均注释比例」，使用橙色
ax2 = ax1.twinx()
line2 = ax2.plot(
    x_indices,
    avg_comment_density,
    marker='s',      
    linewidth=2,
    markersize=4,
    color='tab:orange',  # 第二条曲线颜色
    label="Avg Comment Density"
)
ax2.set_ylabel("Avg Comment Ratio", fontsize=9)
ax2.tick_params(axis='y', labelsize=7)

# 6. 合并两条曲线图例
lines = line1 + line2
labels = [l.get_label() for l in lines]

# 将图例放在标题下方、横向居中
plt.legend(
    lines, 
    labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.11),  # 可根据需要微调 y 值
    borderaxespad=0.,
    fontsize=7,
    ncol = 2
)

# 7. 紧凑布局并保存
plt.tight_layout()
plt.savefig("LLM_code/output_by_quarter/by_mod/figures/comments_trend_dual.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("绘图完成，图像已保存为 'comments_trend_dual.pdf'。")
