import json, ast, csv, os
from pathlib import Path
from collections import Counter
import math
from typing import Set 


def analyze_code(code: str) -> Set[str]:
    unique_vars_in_block = set() 
    try:
        tree = ast.parse(code)
    except Exception:
        return unique_vars_in_block 
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            unique_vars_in_block.add(node.id) 
    return unique_vars_in_block


unique_path = Path("LLM_code/codeforces/simulation/unique_problem_python.json")
if not unique_path.exists():
    print(f"Error: unique_problem_python.json not found at {unique_path}. Exiting.")
    exit()

with unique_path.open("r", encoding="utf-8") as f:
    unique_items = json.load(f)
ac_map = {item["submission_id"]: item["sourceCode"] for item in unique_items}


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


for model_name, file_path in input_files.items():
    if not Path(file_path).exists():
        print(f"Warning: Model output file not found for {model_name} at {file_path}. Skipping.")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    var_counter_ac  = Counter()
    var_counter_ans = Counter()
    var_counter_ref = Counter()
    
    all_seen_vars = set() 

    for it in items:
        sid       = it.get("submission_id")
        ac_code   = ac_map.get(sid, "")
        ans_code  = it.get("generate_code_block", "")
        ref_code  = it.get("generate_ref_code_block", "")

        current_ac_unique_vars = analyze_code(ac_code)
        current_ans_unique_vars = analyze_code(ans_code)
        current_ref_unique_vars = analyze_code(ref_code)


        var_counter_ac.update(current_ac_unique_vars)
        var_counter_ans.update(current_ans_unique_vars)
        var_counter_ref.update(current_ref_unique_vars)
        

        all_seen_vars.update(current_ac_unique_vars)
        all_seen_vars.update(current_ans_unique_vars)
        all_seen_vars.update(current_ref_unique_vars)


    records = []
    sorted_all_seen_vars = sorted(list(all_seen_vars)) 

    for var in sorted_all_seen_vars:
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

    records.sort(key=lambda x: x[0], reverse=True)

    out_csv = os.path.join(out_dir, f"unique/{model_name}_all_variable_frequencies_unique_per_block.csv") 
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "ac_frequency", "ans_frequency",
                    "ref_frequency", "increase_ratio"])
        for inc, var, ac_f, ans_f, ref_f in records:
            display_inc = "inf" if math.isinf(inc) else round(inc, 6)
            w.writerow([var, ac_f, ans_f, ref_f, display_inc])

    print(f"✅ Completed all variable frequency stats for {model_name}, saved to {out_csv}")