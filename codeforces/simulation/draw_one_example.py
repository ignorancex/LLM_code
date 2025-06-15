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
from tqdm import tqdm


# ====================== 通用工具 ====================== #
def get_xticks(quarters):
    """只在 Q1（或 2025Q1）处标年份，保持横坐标简洁"""
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
    y_all = [y for series in y_vals for y in series]
    y_mid = (max(y_all) + min(y_all)) / 2 if y_all else 0

    for ys in y_vals:
        for i, y in enumerate(ys):
            if i < x_mid and y >= y_mid:
                quadrants["upper left"] += 1
            elif i >= x_mid and y >= y_mid:
                quadrants["upper right"] += 1
            elif i < x_mid and y < y_mid:
                quadrants["lower left"] += 1
            else:
                quadrants["lower right"] += 1

    return min(quadrants, key=quadrants.get) if quadrants else "best"


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
        colors = {"cs": "#4589c8ff", "non_cs": "#ee7c7aff"}

    if legend_locations is None:
        legend_locations = {}

    for var in tqdm(variables, desc="Plotting variables"):
        # --------- 提取 y --------- #
        cs_y = [data.get(q, {}).get("cs", {}).get(var, 0) for q in quarters]
        noncs_y = [data.get(q, {}).get("non_cs", {}).get(var, 0) for q in quarters]

        # --------- 开始绘图 --------- #
        plt.figure(figsize=(3.5, 2.5))
        legend_vals = []

        plt.plot(
            quarters,
            cs_y,
            marker="x",
            linestyle="--",
            linewidth=2,
            markersize=4,
            label="cs",
            color=colors["cs"],
        )
        legend_vals.append(cs_y)

        if plot_non_cs:
            plt.plot(
                quarters,
                noncs_y,
                marker="x",
                linestyle="--",
                linewidth=2,
                markersize=4,
                label="non-cs",
                color=colors["non_cs"],
            )
            legend_vals.append(noncs_y)

        # --------- 轴与图例 --------- #
        all_y = [v for series in legend_vals for v in series]
        y_min, y_max = (min(all_y), max(all_y)) if all_y else (0, 0.05)
        margin = (y_max - y_min) * 0.1
        plt.ylim(max(0, y_min - margin), y_max + margin if y_max + margin else 0.05)

        plt.ylabel("Frequency", fontsize=10)
        plt.xticks(xticks, xtick_labels, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(False)

        chosen_loc = legend_locations.get(var, get_best_legend_loc(quarters, legend_vals))
        plt.legend(
            fontsize=10,
            loc=chosen_loc,
            frameon=True,
            facecolor="white",
            framealpha=1,
            labelspacing=0.2,
        )

        plt.tight_layout()

        # --------- 保存 --------- #
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
            data[q]["cs"][var] = float(row.get(q, 0))

    for _, row in noncs_df.iterrows():
        var = row["variable"]
        for q in quarters:
            data[q]["non_cs"][var] = float(row.get(q, 0))

    return data


# ====================== 主程序 ====================== #
if __name__ == "__main__":
    # ---------- 输入文件 ----------
    cs_csv = "cs_freq_all.csv"
    noncs_csv = "ncs_freq_all.csv"

    # ---------- 加载数据 ----------
    cs_df = pd.read_csv(cs_csv)
    noncs_df = pd.read_csv(noncs_csv)

    # ---------- 目标变量 ----------
    targets = ["max_length"]

    # ---------- 特定变量的图例位置 ----------
    SPECIFIC_LEGEND_LOCATIONS = {
        "max_length": "upper left",
    }

    # ---------- 仅保留不晚于 2025Q1 的季度列 ----------
    def quarter_leq_2025Q1(q: str) -> bool:
        """
        判断季度字符串是否不晚于 2025Q1  
        要求格式为 'YYYYQX'
        """
        m = re.fullmatch(r"(\d{4})Q([1-4])", q)
        if not m:
            return False
        year, qtr = int(m.group(1)), int(m.group(2))
        return (year < 2025) or (year == 2025 and qtr <= 1)

    quarters = sorted(
        [c for c in cs_df.columns if c != "variable" and quarter_leq_2025Q1(c)]
    )

    xticks, xtick_labels = get_xticks(quarters)

    # ---------- 构造绘图数据 ----------
    var_data = build_variable_data(cs_df, noncs_df, quarters)

    # ---------- 绘图 ----------
    COLORS = {"cs": "#4589c8ff", "non_cs": "#ee7c7aff"}
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
