import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_split_naming_trends(agg_file, output_dir):
    # 1. 读取归一化后的跨仓库数据
    with open(agg_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 准备目录
    func_dir = os.path.join(output_dir, 'function')
    var_dir  = os.path.join(output_dir, 'variable')
    os.makedirs(func_dir, exist_ok=True)
    os.makedirs(var_dir,  exist_ok=True)

    # 3. 提取所有季度和命名模式
    quarters = sorted(data.keys())
    xs = list(range(len(quarters)))
    # 两个“类别”是按行数分的 fewer/more
    groups = ['fewer', 'more']
    # 取第一个季度、first group、first kind 下的模式列表
    example_pats = list(next(iter(data.values()))['fewer']['func'].keys())

    # 4. 自定义 x 轴显示：只在 Q1 或最后一个点显示年份
    custom_xticks = []
    custom_labels = []
    for idx, q in enumerate(quarters):
        if q.endswith('Q1') or q == quarters[-1]:
            custom_xticks.append(idx)
            # Q1 显示年份，最后一个点如果不是 Q1 也直接显示标签
            custom_labels.append(q[:4])

    # 5. 绘图：分别对 func 和 var
    for kind, plot_dir in [('func', func_dir), ('var', var_dir)]:
        for pat in tqdm(example_pats, desc=f'Plotting {kind} patterns'):
            plt.figure(figsize=(4, 3))
            all_means = []
            all_stds = []
            # 先收集所有值以便计算 y 轴范围
            for grp in groups:
                grp_means = []
                grp_stds  = []
                for q in quarters:
                    entry = data[q][grp][kind].get(pat, {"mean": 0.0, "std": 0.0})
                    grp_means.append(entry["mean"])
                    grp_stds.append(entry["std"])
                all_means.append(grp_means)
                all_stds.append(grp_stds)

            # 绘制误差棒
            for idx, grp in enumerate(groups):
                means = all_means[idx]
                stds  = all_stds[idx]
                plt.errorbar(
                    xs, means, yerr=stds,
                    marker='o', linestyle='-',
                    label=grp,
                    markersize=4, linewidth=1.5,
                    capsize=3, elinewidth=1
                )

            # 计算 y 轴范围并加上 margin
            flat_means = [m for grp_list in all_means for m in grp_list]
            flat_stds  = [s for grp_list in all_stds for s in grp_list]
            ymin = min([m - s for m, s in zip(flat_means, flat_stds)])
            ymax = max([m + s for m, s in zip(flat_means, flat_stds)])
            margin = (ymax - ymin) * 0.1
            plt.ylim(max(0, ymin - margin), min(1, ymax + margin))

            plt.title(f'{kind.capitalize()} – {pat}', fontsize=10)
            plt.ylabel('Proportion', fontsize=9)
            plt.xticks(custom_xticks, custom_labels, fontsize=8)
            plt.yticks(fontsize=8)
            plt.grid(False)
            plt.legend(fontsize=8, ncol=1, loc='upper left', frameon=True)
            plt.tight_layout()

            # 保存
            fname = f'{kind}_{pat}.pdf'
            plt.savefig(os.path.join(plot_dir, fname), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    agg_json = 'LLM_code/arxiv_result/naming_patterns_split/naming_patterns_avg_std.json'
    out_dir  = 'LLM_code/arxiv_result/naming_patterns_split/plots'
    plot_split_naming_trends(agg_json, out_dir)
