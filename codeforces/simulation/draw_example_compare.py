#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-plot quarterly frequency trends of target variables
  • 每个变量单独成图
  • cs / non-cs 两组仓库对比
  • 画图范式与 plot_pattern_trend 模板保持一致
"""
from typing import Dict, List, Tuple
import os
import re
import numpy as np # 用于 np.arange
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # 用于显示进度条

# —————————————— 0. 全局常量与路径 ——————————————
# 定义唯一的输入 CSV 文件路径
INTERSECTION_VARS_CSV_PATH = "LLM_code/codeforces/simulation/result/case/unique/intersection_vars.csv"
OUT_DIR = "plots" # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 定义固定总数，用于频率归一化
TOTAL_PROBLEMS = 1215 # 这是从您之前代码的 ` / 1215` 推断出的

# —————————————— 1. 从 CSV 加载数据并获取模型列表 ——————————————
intersection_df: pd.DataFrame = pd.DataFrame()
MODEL_NAMES: List[str] = [] # 存储从CSV中识别到的模型名称

if not os.path.exists(INTERSECTION_VARS_CSV_PATH):
    print(f"错误：文件 '{INTERSECTION_VARS_CSV_PATH}' 不存在。请检查文件路径是否正确。")
    # 如果文件不存在，则无法获取模型名称，程序将退出或跳过绘图
    exit() # 如果文件是必须的，直接退出
else:
    try:
        intersection_df = pd.read_csv(INTERSECTION_VARS_CSV_PATH)

        if 'variable' in intersection_df.columns:
            # 将 'variable' 列设为索引，方便按变量名查找数据
            intersection_df = intersection_df.set_index('variable')
            # 此时不再自动从 CSV 读取所有变量到 TARGET_VARS
            
            # 动态识别模型名称：查找所有以 '_ac_frequency' 结尾的列，并提取前缀
            potential_model_cols = [col for col in intersection_df.columns if col.endswith('_ac_frequency')]
            
            # 保持您原始代码中固定的模型顺序（如果 CSV 中存在这些模型）
            fixed_model_order = ["GPT", "Gemini", "DeepSeek", "Llama", "Qwen", "Gemma"]
            MODEL_NAMES = [m for m in fixed_model_order if f"{m}_ac_frequency" in intersection_df.columns]
            
            if not MODEL_NAMES:
                print("警告：未能在 CSV 中找到任何模型数据列（如 'DeepSeek_ac_frequency'）。请检查 CSV 格式。")
                # 如果没有识别到模型，可能需要调整处理，例如退出或跳过绘图
                exit()
            
            print(f"识别到的模型: {MODEL_NAMES}")

        else:
            print(f"错误：CSV 文件 '{INTERSECTION_VARS_CSV_PATH}' 中未找到 'variable' 列。")
            exit() # 如果没有 'variable' 列，数据结构不对，退出

    except Exception as e:
        print(f"读取或处理 CSV 文件时发生错误：{e}")
        exit() # 发生任何读取错误，退出

# —————————————— 指定目标变量 ——————————————
# 在这里手动指定您想要绘制的变量列表
TARGET_VARS: List[str] = [
    "max_length",
    "response"
    # 添加您想绘制的其他变量名
]

# 确保所有指定的 TARGET_VARS 都在 DataFrame 的索引中
# 过滤掉不存在的变量，避免绘图时报错
TARGET_VARS = [var for var in TARGET_VARS if var in intersection_df.index]
if not TARGET_VARS:
    print("警告：指定的 TARGET_VARS 列表为空或所有指定变量均未在 CSV 中找到。将不生成图表。")


# —————————————— 2. 绘图配置与核心函数 ——————————————
colors = {                                   # 与前一致
    "ac":  "#c8c8c8",   # Human
    "ref": "#ffde7b",   # LLM-Revised
    "ans": "#6ad1a3",   # LLM-Generated
}
legend_labels = {"ref": "LLM-Revised", "ans": "LLM-Generated"}

def plot_single_variable_comparison(target_var: str):
    """
    为指定的 target_var 绘制 Human、LLM-Revised 和 LLM-Generated 频率对比条形图。
    数据从全局的 intersection_df 中提取。
    """
    # 再次检查目标变量是否存在于 DataFrame 的索引中（在 TARGET_VARS 过滤后理论上已存在）
    if target_var not in intersection_df.index:
        print(f"内部错误：变量 '{target_var}' 未在 DataFrame 中找到，跳过绘图。")
        return

    row_data = intersection_df.loc[target_var]

    # ———— 提取数据 ————
    # Human 频率：从第一个模型的 ac_frequency 列获取，假设所有模型的 ac_frequency 都相同
    # 或者如果知道一个特定的模型数据最可靠，也可以直接指定
    human_freq = 0.0
    if MODEL_NAMES:
        first_model_ac_col = f"{MODEL_NAMES[0]}_ac_frequency"
        if first_model_ac_col in row_data:
            human_freq = row_data[first_model_ac_col] / TOTAL_PROBLEMS
        else:
            # 这通常不应该发生，因为 MODEL_NAMES 已经过滤过
            print(f"警告：无法从 '{first_model_ac_col}' 获取 Human 频率，设为 0。")
    
    ref_vals, ans_vals = [], []
    for model in MODEL_NAMES:
        ref_col = f"{model}_ref_frequency"
        ans_col = f"{model}_ans_frequency"
        
        # 使用 .get() 方法安全地获取值，如果列不存在则默认为 0.0
        current_ref_freq = row_data.get(ref_col, 0.0) / TOTAL_PROBLEMS
        current_ans_freq = row_data.get(ans_col, 0.0) / TOTAL_PROBLEMS
        
        ref_vals.append(current_ref_freq)
        ans_vals.append(current_ans_freq)

    # —————————————— 3. 绘图 ——————————————
    bar_width   = 0.5     # Human 柱宽
    sub_width   = 0.3     # 模型子柱宽
    x           = np.arange(len(MODEL_NAMES) + 1) # 0 为 Human，其余 1-N 为模型
    human_pos   = x[0]
    model_pos   = x[1:]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # —— Human 柱 ——
    ax.bar(human_pos, human_freq,
           width=bar_width,
           color=colors["ac"],)
        #    label="Human") # 为 Human 添加图例

    # —— 模型 ref / ans 对照柱 ——
    for i, model_name in enumerate(MODEL_NAMES): # 遍历识别到的模型名称
        pos = model_pos[i]
        
        # 确保只给第一个 bar 添加 label，以避免图例重复
        ax.bar(pos - sub_width/2, ref_vals[i],
               width=sub_width,
               color=colors["ref"],
               label=legend_labels["ref"] if i == 0 else "")
        ax.bar(pos + sub_width/2, ans_vals[i],
               width=sub_width,
               color=colors["ans"],
               label=legend_labels["ans"] if i == 0 else "")

    # ——— 轴 & 样式 ———
    xticks       = x
    xtick_labels = ["Human", "GPT", "Gemini", "DS", "Llama", "Qw", "Gemma"] # 横坐标标签现在包括 Human 和所有模型名称
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)

    max_val = max(human_freq, *ref_vals, *ans_vals)
    ax.set_ylabel("Frequency", fontsize=9)
    # 确保 ylim 的上限合理，避免 max_val 为 0 导致错误
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 0.1) # 至少给一个很小的上限如果都是0
    plt.yticks(fontsize=8)

    ax.legend(fontsize=8, loc="upper right", frameon=True, labelspacing=0.2)
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"{target_var}_compare.pdf") # 以变量名命名文件
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存：{save_path}")


# —————————————— 4. 主程序 ——————————————
if __name__ == "__main__":
    if not TARGET_VARS:
        print("没有指定的目标变量可供绘制图表。请检查 TARGET_VARS 列表。")
    else:
        # 使用 tqdm 包装循环，显示进度条
        for var_to_plot in tqdm(TARGET_VARS, desc="Generating plots for specified variables"):
            plot_single_variable_comparison(var_to_plot)

    print("\n所有图表绘制完成！")