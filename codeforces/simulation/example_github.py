#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计指定变量在 CS / non-CS 仓库中的文件频次
1. 计算指定变量在 2020Q1–2025Q1 各季度的文件频次
   - 平均频率将计入所有在该季度/类别下的仓库，即使变量未出现（频率为0）。
2. 将所有变量的频次写入 CSV
"""

import os
import ast
import json
import csv
import warnings
import concurrent.futures
from collections import defaultdict
from typing import Set, Dict, Tuple, List
from tqdm import tqdm
import pandas as pd 

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- 0. 路径与常量 ----------
BASE_DIR = "LLM_code/arxiv_dataset"                      # 年/季度/仓库
CATEGORIES_JSON = "LLM_code/arxiv_result/github_links/python_dataset_links.json"
OUT_DIR = "LLM_code/arxiv_result/naming_patterns_python"
SKIPPED_LOG = os.path.join(OUT_DIR, "skipped_files_limited_vars.txt") # 修改日志文件名
os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(SKIPPED_LOG):
    os.remove(SKIPPED_LOG)

# 定义 CSV 文件的路径
csv_file_path = "LLM_code/codeforces/simulation/result/case/unique/intersection_vars.csv"

# 检查文件是否存在
if not os.path.exists(csv_file_path):
    print(f"错误：文件 '{csv_file_path}' 不存在。请检查文件路径是否正确。")
    # 如果文件不存在，TARGET_VARS 将为空，或你可以选择退出程序
    TARGET_VARS = []
else:
    try:
        # 读取 CSV 文件
        # 使用 header=0 来确保第一行被识别为列名
        df = pd.read_csv(csv_file_path, header=0)

        # 从 'variable' 列中提取变量并转换为列表
        if 'variable' in df.columns:
            TARGET_VARS = df['variable'].tolist()
            print(f"成功从 '{csv_file_path}' 中读取 {len(TARGET_VARS)} 个变量。")
        else:
            print(f"错误：CSV 文件 '{csv_file_path}' 中未找到 'variable' 列。")
            TARGET_VARS = []

    except Exception as e:
        print(f"读取或处理 CSV 文件时发生错误：{e}")
        TARGET_VARS = []

QUARTERS_2020 = [f"2020Q{q}" for q in range(1, 5)] # This variable is defined but not used in the final script.
ALL_QUARTERS = [f"{y}Q{q}" for y in range(2020, 2025) for q in range(1, 5)] + ["2025Q1"]

# ---------- 1. 工具 ----------
def classify_category(cat_str: str) -> str:
    """arXiv 主类 转 cs / non_cs"""
    return "cs" if cat_str.startswith("cs.") else "non_cs"

def extract_vars(py_file: str) -> Set[str]:
    """返回该文件出现过的所有变量集合（文件级去重）"""
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            code = f.read()
        if "\x00" in code:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "null byte")
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError, ValueError) as e:
        with open(SKIPPED_LOG, "a", encoding="utf-8") as lg:
            lg.write(f"Skipped {py_file}: {e}\n")
        return set()

    vars_in_file: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    vars_in_file.add(t.id)  # 添加所有变量名，不论命名规则
    return vars_in_file

def process_repo(repo_path: str) -> Tuple[str, int, Dict[str, float]]:
    """
    遍历仓库，返回
      repo_name: 仓库名
      total_py_files: 该仓库的Python文件总数
      var_frequencies_in_repo: {var: 该变量在该仓库的文件频率} (只包含 TARGET_VARS 中的变量)
    """
    repo_name = os.path.basename(repo_path)
    total_files_in_repo = 0
    var_file_counts_in_repo: Dict[str, int] = defaultdict(int) # 变量在该仓库出现的文件计数
    
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            total_files_in_repo += 1
            vars_here = extract_vars(os.path.join(root, fn))
            # 仅统计 TARGET_VARS 中的变量
            for v in vars_here:
                if v in TARGET_VARS: # 限制只统计目标变量
                    var_file_counts_in_repo[v] += 1
    
    var_frequencies_in_repo: Dict[str, float] = {}
    if total_files_in_repo > 0:
        for v, count in var_file_counts_in_repo.items():
            var_frequencies_in_repo[v] = count / total_files_in_repo
            
    return repo_name, total_files_in_repo, var_frequencies_in_repo

# ---------- 2. 仓库 → 类别 映射 ----------
quarter_repo_cat: Dict[str, Dict[str, str]] = defaultdict(dict)
with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
    for quarter, items in json.load(f).items():
        for item in items:
            repo = item["link"].rstrip("/").split("/")[-1]
            quarter_repo_cat[quarter][repo] = classify_category(item["categories"])

# ---------- 3. 统计器 ----------
total_files_qc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# 存储每个季度、每个类别下，每个仓库的变量频率字典
# per_repo_var_freq[quarter][cat][repo_name][var] = freq
per_repo_var_freq: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(dict))
)

# 核心改变：all_variables_overall 现在就是 TARGET_VARS
all_variables_overall: Set[str] = set(TARGET_VARS)

# ---------- 4. 扫描全部季度 ----------
print("Starting to scan repositories for specified variables:", len(TARGET_VARS))

for year in range(2020, 2026):
    max_q = 1 if year == 2025 else 4
    for q in range(1, max_q + 1):
        q_dir = os.path.join(BASE_DIR, str(year), f"Q{q}")
        quarter_key = f"{year}Q{q}"
        if not os.path.isdir(q_dir):
            continue
        repos_in_quarter_names = [d for d in os.listdir(q_dir) if os.path.isdir(os.path.join(q_dir, d))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(process_repo, os.path.join(q_dir, repo_name)): repo_name for repo_name in repos_in_quarter_names}
            
            for fut in tqdm(concurrent.futures.as_completed(futs),
                            total=len(futs), desc=f"Scanning {quarter_key}"):
                repo_name, repo_total_files, var_freq_in_repo = fut.result()
                cat = quarter_repo_cat.get(quarter_key, {}).get(repo_name, "non_cs")
                
                total_files_qc[quarter_key][cat] += repo_total_files
                
                # 存储该仓库的变量频率字典 (这里 var_freq_in_repo 已经只包含 TARGET_VARS)
                per_repo_var_freq[quarter_key][cat][repo_name] = var_freq_in_repo

# ---------- 5. 计算仓库间的平均频次（所有仓库计入分母） ----------
final_avg_freq_qc: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

print("\nStarting to calculate average frequencies for specified variables...")
# 遍历所有可能的季度
for quarter in tqdm(ALL_QUARTERS, desc="Calculating Quarterly Averages"):
    # 确保当前季度有数据或至少有仓库类别信息
    if quarter not in per_repo_var_freq and quarter not in quarter_repo_cat:
        continue # 该季度无数据，跳过

    for cat in ["cs", "non_cs"]: # 遍历 CS 和 non-CS 类别
        # 获取该季度、该类别下所有应被考虑的仓库名
        repos_to_consider_in_cat = [
            repo_name for repo_name, repo_cat in quarter_repo_cat[quarter].items()
            if repo_cat == cat
        ]
        
        if not repos_to_consider_in_cat:
            continue
        
        num_repos_in_avg = len(repos_to_consider_in_cat)

        # 遍历所有目标变量
        for var in tqdm(TARGET_VARS, desc=f"  -> {quarter} {cat} variables", leave=False):
            sum_of_frequencies = 0.0
            
            # 遍历该季度/类别下所有应被计入平均的仓库
            for repo_name in repos_to_consider_in_cat:
                repo_specific_var_freqs = per_repo_var_freq[quarter][cat].get(repo_name, {})
                
                # 获取该变量在该仓库的频率，如果不存在则为 0.0
                freq_in_this_repo = repo_specific_var_freqs.get(var, 0.0)
                sum_of_frequencies += freq_in_this_repo
            
            if num_repos_in_avg > 0:
                final_avg_freq_qc[quarter][cat][var] = round(sum_of_frequencies / num_repos_in_avg, 6)
            else:
                final_avg_freq_qc[quarter][cat][var] = 0.0

# ---------- 6. 输出所有变量的频次 ----------
print("\nSaving results to CSV files...")
def save_all_vars_csv(cat: str, outfile: str) -> None:
    # 直接使用 TARGET_VARS 作为要输出的变量列表
    all_vars_for_csv = sorted(list(TARGET_VARS))
    
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable"] + ALL_QUARTERS)  # 标题行

        for var in all_vars_for_csv:
            writer.writerow([var] + [final_avg_freq_qc[q][cat].get(var, 0.0) for q in ALL_QUARTERS])  # 变量及其平均频率

    print(f"✅ Saved average frequency data for '{cat}' category (limited vars) to {outfile}")

# 输出 CSV 文件
save_all_vars_csv("cs", os.path.join(OUT_DIR, "variable_cs.csv"))
save_all_vars_csv("non_cs", os.path.join(OUT_DIR, "variable_non_cs.csv"))

print("\nProcessing complete!")