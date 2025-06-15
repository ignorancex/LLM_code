# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 统计所有变量在 CS / non-CS 模型代码中的出现频次
# 步骤：
# 1. 收集 AC, REF, ANS 三种代码中出现过的所有变量。
# 2. 对每个变量，计算其在 AC, REF, ANS 中的频次。
# 3. 计算 REF 相对 AC 的增幅，并处理 AC 频次为 0 的情况。
# 4. 按增幅从大到小输出到 CSV。
# """

# import json, ast, csv, os
# from pathlib import Path
# from collections import Counter
# import math # 引入 math 模块用于处理无穷大

# # === 1. Python 代码分析 ===
# def analyze_code(code: str) -> Counter:
#     var_counter = Counter()
#     try:
#         tree = ast.parse(code)
#     except Exception:
#         # 捕获解析错误，返回空 Counter，不中断程序
#         return var_counter
#     for node in ast.walk(tree):
#         # 只关注变量名，不包括函数名、类名等，但这里 ast.Name 会包括所有这些
#         # 如果需要更精确的变量定义，需要更复杂的 AST 遍历逻辑
#         if isinstance(node, ast.Name):
#             var_counter[node.id] += 1
#     return var_counter

# # === 2. 预加载人类 (ac) 代码 ===
# unique_path = Path("LLM_code/codeforces/simulation/unique_problem_python.json")
# # 检查文件是否存在，防止 FileNotFoundError
# if not unique_path.exists():
#     print(f"Error: unique_problem_python.json not found at {unique_path}. Exiting.")
#     exit()

# with unique_path.open("r", encoding="utf-8") as f:
#     unique_items = json.load(f)
# ac_map = {item["submission_id"]: item["sourceCode"] for item in unique_items}

# # === 3. 模型输出文件 ===
# input_files = {
#     "DeepSeek": "LLM_code/codeforces/simulation/output/DeepSeek_python.json",
#     "Gemma":    "LLM_code/codeforces/simulation/output/Gemma_python.json",
#     "Qwen":     "LLM_code/codeforces/simulation/output/Qwen_python.json",
#     "Gemini":   "LLM_code/codeforces/simulation/output/Gemini_python.json",
#     "GPT":      "LLM_code/codeforces/simulation/output/GPT_python.json",
#     "Llama":    "LLM_code/codeforces/simulation/output/Llama4_python.json",
# }

# out_dir = "LLM_code/codeforces/simulation/result/case"
# os.makedirs(out_dir, exist_ok=True)

# # === 4. 处理每个模型 ===
# for model_name, file_path in input_files.items():
#     # 检查模型文件是否存在
#     if not Path(file_path).exists():
#         print(f"Warning: Model output file not found for {model_name} at {file_path}. Skipping.")
#         continue

#     with open(file_path, "r", encoding="utf-8") as f:
#         items = json.load(f)

#     var_counter_ac  = Counter()
#     var_counter_ans = Counter()
#     var_counter_ref = Counter()
    
#     # 新增：一个集合，用于存储所有出现过的变量，无论来源
#     all_seen_vars = set() 

#     # ---- 4.1 遍历每条记录，累计频次 ----
#     for it in items:
#         sid       = it.get("submission_id")
#         ac_code   = ac_map.get(sid, "")
#         ans_code  = it.get("generate_code_block", "")
#         ref_code  = it.get("generate_ref_code_block", "")

#         current_ac_vars = analyze_code(ac_code)
#         current_ans_vars = analyze_code(ans_code)
#         current_ref_vars = analyze_code(ref_code)

#         var_counter_ac.update(current_ac_vars)
#         var_counter_ans.update(current_ans_vars)
#         var_counter_ref.update(current_ref_vars)
        
#         # 将当前代码块中发现的所有变量添加到总集合
#         all_seen_vars.update(current_ac_vars.keys())
#         all_seen_vars.update(current_ans_vars.keys())
#         all_seen_vars.update(current_ref_vars.keys())


#     # ---- 4.2 遍历所有出现过的变量，计算增幅和频次 ----
#     # 不再根据 AC 频次取前多少个，而是遍历 all_seen_vars
#     records = []
#     # 确保排序是稳定的，可以先转为列表再排序
#     sorted_all_seen_vars = sorted(list(all_seen_vars)) 

#     for var in sorted_all_seen_vars:
#         ac_freq  = var_counter_ac.get(var, 0)
#         ans_freq = var_counter_ans.get(var, 0)
#         ref_freq = var_counter_ref.get(var, 0)

#         inc_ratio = 0.0 # 默认增幅为0

#         if ac_freq == 0:
#             if ref_freq > 0:
#                 # 如果 ac 频次为 0 但 ref 频次大于 0，视为无限增幅
#                 inc_ratio = float('inf') 
#             # 如果 ac_freq 和 ref_freq 都为 0，增幅仍为 0.0
#         else:
#             inc_ratio = (ref_freq - ac_freq) / ac_freq
        
#         # 记录增幅，变量名，以及三种频次
#         records.append((inc_ratio, var, ac_freq, ans_freq, ref_freq))

#     # ---- 4.3 排序 ----
#     # 排序时，默认 float('inf') 会被排在最前面
#     records.sort(key=lambda x: x[0], reverse=True)   # 按增幅降序

#     # ---- 4.4 写 CSV ----
#     out_csv = os.path.join(out_dir, f"{model_name}_all_variable_frequencies.csv") # 修改输出文件名
#     with open(out_csv, "w", encoding="utf-8", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["variable", "ac_frequency", "ans_frequency",
#                     "ref_frequency", "increase_ratio"])
#         for inc, var, ac_f, ans_f, ref_f in records:
#             # 对于无限大，输出 'inf' 或其他标记，或直接保留浮点数形式
#             # csv 默认可以处理 float('inf')，但为了可读性，可以转字符串
#             display_inc = "inf" if math.isinf(inc) else round(inc, 6)
#             w.writerow([var, ac_f, ans_f, ref_f, display_inc])

#     print(f"✅ Completed all variable frequency stats for {model_name}, saved to {out_csv}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计所有变量在 CS / non-CS 模型代码中的出现频次
这里的频次定义为：一个变量在多少个独立的“代码块”中作为“唯一变量”出现过。
（即，在一个代码块中无论出现多少次，只算1次该代码块的贡献）

步骤：
1. 收集 AC, REF, ANS 三种代码中出现过的所有变量。
2. 对每个变量，计算其在 AC, REF, ANS 中“作为唯一变量出现过的代码块数量”。
3. 计算 REF 相对 AC 的增幅，并处理 AC 频次为 0 的情况。
4. 按增幅从大到小输出到 CSV。
"""

import json, ast, csv, os
from pathlib import Path
from collections import Counter
import math
from typing import Set # 新增导入 Set，用于类型提示

# === 1. Python 代码分析 ===
# 修改：现在返回的是一个集合 (set)，包含该代码块中所有不重复的变量名
def analyze_code(code: str) -> Set[str]:
    unique_vars_in_block = set() # 使用集合来存储不重复的变量名
    try:
        tree = ast.parse(code)
    except Exception:
        return unique_vars_in_block # 解析错误时返回空集合
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            unique_vars_in_block.add(node.id) # 将变量名添加到集合中，集合会自动去重
    return unique_vars_in_block

# === 2. 预加载人类 (ac) 代码 ===
unique_path = Path("LLM_code/codeforces/simulation/unique_problem_python.json")
if not unique_path.exists():
    print(f"Error: unique_problem_python.json not found at {unique_path}. Exiting.")
    exit()

with unique_path.open("r", encoding="utf-8") as f:
    unique_items = json.load(f)
ac_map = {item["submission_id"]: item["sourceCode"] for item in unique_items}

# === 3. 模型输出文件 ===
input_files = {
    "DeepSeek": "LLM_code/codeforces/simulation/output/DeepSeek_python.json",
    "Gemma":    "LLM_code/codeforces/simulation/output/Gemma_python.json",
    "Qwen":     "LLM_code/codeforces/simulation/output/Qwen_python.json",
    "Gemini":   "LLM_code/codeforces/simulation/output/Gemini_python.json",
    "GPT":      "LLM_code/codeforces/simulation/output/GPT_python.json",
    "Llama":    "LLM_code/codeforces/simulation/output/Llama4_python.json",
}

out_dir = "LLM_code/codeforces/simulation/result/case"
os.makedirs(out_dir, exist_ok=True)

# === 4. 处理每个模型 ===
for model_name, file_path in input_files.items():
    if not Path(file_path).exists():
        print(f"Warning: Model output file not found for {model_name} at {file_path}. Skipping.")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 这些仍然是 Counter，因为它们将累加变量在不同代码块中作为唯一变量出现的次数
    var_counter_ac  = Counter()
    var_counter_ans = Counter()
    var_counter_ref = Counter()
    
    all_seen_vars = set() 

    # ---- 4.1 遍历每条记录，累计频次 ----
    for it in items:
        sid       = it.get("submission_id")
        ac_code   = ac_map.get(sid, "")
        ans_code  = it.get("generate_code_block", "")
        ref_code  = it.get("generate_ref_code_block", "")

        # analyze_code 现在返回的是集合 (Set[str])
        current_ac_unique_vars = analyze_code(ac_code)
        current_ans_unique_vars = analyze_code(ans_code)
        current_ref_unique_vars = analyze_code(ref_code)

        # Counter 的 update() 方法接受一个可迭代对象。
        # 如果传入一个集合，它会为集合中的每个元素计数加1。
        # 因此，这会统计该变量在多少个独立的“代码块”中出现过（作为唯一变量）。
        var_counter_ac.update(current_ac_unique_vars)
        var_counter_ans.update(current_ans_unique_vars)
        var_counter_ref.update(current_ref_unique_vars)
        
        # 将当前代码块中发现的所有变量添加到总集合
        all_seen_vars.update(current_ac_unique_vars)
        all_seen_vars.update(current_ans_unique_vars)
        all_seen_vars.update(current_ref_unique_vars)


    # ---- 4.2 遍历所有出现过的变量，计算增幅和频次 ----
    records = []
    sorted_all_seen_vars = sorted(list(all_seen_vars)) 

    for var in sorted_all_seen_vars:
        # 这里的 ac_freq, ans_freq, ref_freq 现在代表的是变量在多少个代码块中作为唯一变量出现过
        ac_freq  = var_counter_ac.get(var, 0)
        ans_freq = var_counter_ans.get(var, 0)
        ref_freq = var_counter_ref.get(var, 0)

        inc_ratio = 0.0

        if ac_freq == 0:
            if ref_freq > 0:
                inc_ratio = float('inf') 
        else:
            inc_ratio = (ref_freq - ac_freq) / ac_freq
        
        records.append((inc_ratio, var, ac_freq, ans_freq, ref_freq))

    # ---- 4.3 排序 ----
    records.sort(key=lambda x: x[0], reverse=True)

    # ---- 4.4 写 CSV ----
    out_csv = os.path.join(out_dir, f"unique/{model_name}_all_variable_frequencies_unique_per_block.csv") # 修改输出文件名以区分
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "ac_frequency", "ans_frequency",
                    "ref_frequency", "increase_ratio"])
        for inc, var, ac_f, ans_f, ref_f in records:
            display_inc = "inf" if math.isinf(inc) else round(inc, 6)
            w.writerow([var, ac_f, ans_f, ref_f, display_inc])

    print(f"✅ Completed all variable frequency stats for {model_name}, saved to {out_csv}")