#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-plot quarterly frequency trends of target variables
  • 每个变量单独成图
  • cs / non-cs 两组仓库对比
  • 画图范式与 plot_pattern_trend 模板保持一致
  • 仅读取至 2025Q1（含）
"""
from typing import Dict
import os
import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ====================== 通用工具 ====================== #
def get_xticks(quarters):
    """只在 Q1（或 2025Q1）处标年份"""
    ticks, labels = [], []
    for q in quarters:
        if q.endswith("Q1") or q == "2025Q1":
            ticks.append(q)
            labels.append(q[:4])
    return ticks, labels


def get_best_legend_loc(x_vals, y_vals):
    """智能挑选 legend 落点最少的象限"""
    quadrants = defaultdict(int)
    x_mid = len(x_vals) // 2
    y_all = [y for series in y_vals for y in series if not np.isnan(y)]
    y_mid = (max(y_all) + min(y_all)) / 2 if y_all else 0

    for ys in y_vals:
        for i, y in enumerate(ys):
            if np.isnan(y):
                continue
            if i < x_mid and y >= y_mid:
                quadrants["upper left"] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants["upper right"] += 1
            elif i < x_mid and y < y_mid:
                quadrants["lower left"] += 1
            else:
                quadrants["lower right"] += 1

    return min(quadrants, key=quadrants.get) if quadrants else "best"


def fit_and_plot(x_idx, y_vals, color):
    """对一个阶段的数据做线性拟合并画虚线"""
    x = np.array(x_idx)
    y = np.array(y_vals)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    poly = np.poly1d(coef)
    plt.plot(x[mask], poly(x[mask]), linestyle="--", linewidth=1.5, color=color, alpha=0.9)


# ====================== 绘图核心 ====================== #
def plot_variable_trend(
    data,
    quarters,
    variables,
    output_dir="plots",
    colors=None,
    xticks=None,
    xtick_labels=None,
    plot_non_cs=True,
    legend_locations: Dict[str, str] = None,
):
    """
    为 variables 中的每个变量出一张 cs / non-cs 趋势图
    data 结构:
        data[q]["cs" / "non_cs"][var] = freq
    """
    os.makedirs(output_dir, exist_ok=True)
    if colors is None:
        colors = {"cs": "#1f77b4", "non_cs": "#ff7f0e"}

    if legend_locations is None:
        legend_locations = {}

    # 两个阶段
    stage1 = [q for q in quarters if "2020Q1" <= q <= "2023Q1"]
    stage2 = [q for q in quarters if "2023Q2" <= q <= "2025Q3"]

    for var in tqdm(variables, desc="Plotting variables"):
        cs_y = [data.get(q, {}).get("cs", {}).get(var, np.nan) for q in quarters]
        noncs_y = [data.get(q, {}).get("non_cs", {}).get(var, np.nan) for q in quarters]

        plt.figure(figsize=(3.5, 2.5))
        legend_vals = []

        # 主曲线
        plt.plot(
            quarters,
            cs_y,
            linestyle="-",
            linewidth=2,
            label="cs",
            color=colors["cs"],
        )
        legend_vals.append(cs_y)

        if plot_non_cs:
            plt.plot(
                quarters,
                noncs_y,
                linestyle="-",
                linewidth=2,
                label="non-cs",
                color=colors["non_cs"],
            )
            legend_vals.append(noncs_y)


        # === 分阶段拟合 ===
        x_idx1 = [i for i, q in enumerate(quarters) if q in stage1]
        fit_and_plot(x_idx1, [cs_y[i] for i in x_idx1], "blue")
        fit_and_plot(x_idx1, [noncs_y[i] for i in x_idx1], "red")

        x_idx2 = [i for i, q in enumerate(quarters) if q in stage2]
        fit_and_plot(x_idx2, [cs_y[i] for i in x_idx2], "blue")
        fit_and_plot(x_idx2, [noncs_y[i] for i in x_idx2], "red")


        # === y 轴范围 ===
        all_y = [v for series in legend_vals for v in series if not np.isnan(v)]
        if all_y:
            y_min, y_max = min(all_y), max(all_y)
            margin = (y_max - y_min) * 0.1
            plt.ylim(max(0, y_min - margin), y_max + margin)
        else:
            plt.ylim(0, 0.05)

        plt.ylabel("Frequency", fontsize=10)
        plt.xticks(xticks, xtick_labels, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(False)

        # 图例靠边
        chosen_loc = legend_locations.get(var, get_best_legend_loc(quarters, legend_vals))
        anchor_map = {
            "upper left": (0.0, 1.0),
            "upper right": (1.0, 1.0),
            "lower left": (0.0, 0.0),
            "lower right": (1.0, 0.0),
        }
        plt.legend(
            fontsize=10,
            loc=chosen_loc,
            bbox_to_anchor=anchor_map.get(chosen_loc, (1.0, 1.0)),
            frameon=True,
            facecolor="white",
            framealpha=1,
            labelspacing=0.2,
        )

        # === 2023Q1 竖线 ===
        if "2023Q1" in quarters:
            idx = quarters.index("2023Q1")
            plt.axvline(x=quarters[idx], color="gray", linestyle="--", linewidth=1)

        plt.tight_layout()

        safe_var = re.sub(r"[\\/:*?\"<>|]", "_", var)
        suffix = "" if plot_non_cs else "_cs_only"
        fpath = os.path.join(output_dir, f"freq_{safe_var}{suffix}.pdf")
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()


# ====================== 数据构造 ====================== #
def build_variable_data(cs_df, noncs_df, quarters):
    """将 CSV 转为 data[quarter]['cs' / 'non_cs'][variable] = freq"""
    data = {q: {"cs": {}, "non_cs": {}} for q in quarters}
    for _, row in cs_df.iterrows():
        var = row["variable"]
        for q in quarters:
            data[q]["cs"][var] = float(row.get(q, np.nan))

    for _, row in noncs_df.iterrows():
        var = row["variable"]
        for q in quarters:
            data[q]["non_cs"][var] = float(row.get(q, np.nan))

    return data


# ====================== 主程序 ====================== #
if __name__ == "__main__":
    cs_csv = "naming_patterns/case_study/github_result/cs_trend.csv"
    noncs_csv = "naming_patterns/case_study/github_result/ncs_trend.csv"

    cs_df = pd.read_csv(cs_csv)
    noncs_df = pd.read_csv(noncs_csv)

    targets = ["max_length"]

    SPECIFIC_LEGEND_LOCATIONS = {
        "max_length": "upper left",
    }

    def quarter_leq_2025Q1(q: str) -> bool:
        m = re.fullmatch(r"(\d{4})Q([1-4])", q)
        if not m:
            return False
        year, qtr = int(m.group(1)), int(m.group(2))
        return (year < 2025) or (year == 2025 and qtr <= 3)

    quarters = sorted(
        [c for c in cs_df.columns if c != "variable" and quarter_leq_2025Q1(c)]
    )

    xticks, xtick_labels = get_xticks(quarters)

    var_data = build_variable_data(cs_df, noncs_df, quarters)

    COLORS = {"cs": "#1f77b4", "non_cs": "#ff7f0e"}
    plot_variable_trend(
        data=var_data,
        quarters=quarters,
        variables=targets,
        output_dir="plots",
        colors=COLORS,
        xticks=xticks,
        xtick_labels=xtick_labels,
        plot_non_cs=True,
        legend_locations=SPECIFIC_LEGEND_LOCATIONS,
    )
