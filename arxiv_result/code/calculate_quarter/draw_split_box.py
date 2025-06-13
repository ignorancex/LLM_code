import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_split_naming_boxplots(split_file, output_dir):
    # 1. 读取原始跨仓库数据（每个项目的比例）
    with open(split_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # 2. 收集所有季度并排序
    quarters = sorted({entry['quarter'] for entry in entries})
    num_quarters = len(quarters)
    # 为每个季度分配一个索引
    quarter_to_idx = {q: i for i, q in enumerate(quarters)}

    # 命名模式列表（与代码中一致）
    naming_patterns = [
        'single_letter', 'lowercase', 'UPPERCASE',
        'camelCase', 'snake_case', 'PascalCase',
        'endsWithDigits', 'Other'
    ]
    groups = ['fewer', 'more']
    kinds  = ['func', 'var']

    # 3. 初始化数据结构：group → kind → pat → list of lists（每个季度一个列表）
    data = {
        grp: {
            kind: {pat: [[] for _ in quarters] for pat in naming_patterns}
            for kind in kinds
        }
        for grp in groups
    }

    # 4. 遍历每个项目条目，将值放入对应季度、组、种类、模式的位置
    for entry in entries:
        q_idx = quarter_to_idx[entry['quarter']]
        for grp in groups:
            for kind in kinds:
                for pat in naming_patterns:
                    val = entry[grp][kind].get(pat, 0.0)
                    data[grp][kind][pat][q_idx].append(val)

    # 5. 自定义 xticks：只在 Q1 或最后一个点显示年份标签
    custom_xticks = []
    custom_labels = []
    for idx, q in enumerate(quarters):
        if q.endswith('Q1') or q == quarters[-1]:
            custom_xticks.append(idx)
            custom_labels.append(q[:4])

    # 6. 为每个 kind 和 pat 画 box plot
    for kind in kinds:
        kind_dir = os.path.join(output_dir, kind)
        os.makedirs(kind_dir, exist_ok=True)

        for pat in tqdm(naming_patterns, desc=f'Plotting boxplots for {kind}'):
            plt.figure(figsize=(6, 4))

            # x 轴位置
            x = list(range(num_quarters))
            offset = 0.2
            pos_fewer = [i - offset for i in x]
            pos_more  = [i + offset for i in x]

            # 准备两组数据：每季度的样本列表
            data_fewer = data['fewer'][kind][pat]
            data_more  = data['more'][kind][pat]

            # 绘制 boxplot
            bp_fewer = plt.boxplot(
                data_fewer,
                positions=pos_fewer,
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', edgecolor='blue'),
                medianprops=dict(color='navy'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                showfliers=False
            )
            bp_more = plt.boxplot(
                data_more,
                positions=pos_more,
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor='lightgreen', edgecolor='green'),
                medianprops=dict(color='darkgreen'),
                whiskerprops=dict(color='green'),
                capprops=dict(color='green'),
                showfliers=False
            )

            # 添加图例：手动创建图例句柄
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor='lightblue', edgecolor='blue', label='fewer'),
                Patch(facecolor='lightgreen', edgecolor='green', label='more')
            ]
            plt.legend(handles=legend_handles, fontsize=8, loc='upper right', frameon=False)

            # 设置坐标轴和标题
            plt.title(f'{kind.capitalize()} – {pat}', fontsize=10)
            plt.ylabel('Proportion', fontsize=9)
            plt.xticks(custom_xticks, custom_labels, fontsize=8, rotation=45)
            plt.yticks(fontsize=8)

            # 动态调整 y 轴范围：以数据的最小值和最大值为中心，加上一定 margin，
            # 并确保不超出 [0, 1] 的上下限
            all_vals = []
            for idx in range(num_quarters):
                all_vals += data_fewer[idx] + data_more[idx]
            if all_vals:
                ymin, ymax = min(all_vals), max(all_vals)
                margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                lower = max(0, ymin - margin)
                upper = min(1, ymax + margin)
                plt.ylim(lower, upper)

            plt.grid(False)
            plt.tight_layout()

            # 保存
            fname = f'{kind}_{pat}_boxplot.pdf'
            plt.savefig(os.path.join(kind_dir, fname), dpi=300)
            plt.close()

if __name__ == '__main__':
    split_json = 'LLM_code/arxiv_result/naming_patterns_split/naming_patterns_split.json'
    out_dir    = 'LLM_code/arxiv_result/naming_patterns_split/boxplots'
    plot_split_naming_boxplots(split_json, out_dir)
