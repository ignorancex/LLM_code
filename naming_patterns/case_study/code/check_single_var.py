import os
import ast
import csv
import json
import warnings
import concurrent.futures
from collections import defaultdict
from typing import Set, Dict, Tuple, List
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SyntaxWarning)


BASE_DIR = "LLM_code/arxiv_dataset"                      
CATEGORIES_JSON = "LLM_code/arxiv_result/github_links/python_dataset_links.json"
OUT_DIR = "LLM_code/arxiv_result/naming_patterns_python"
SKIPPED_LOG = os.path.join(OUT_DIR, "skipped_files_single_var_check_categorized.txt") 
os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(SKIPPED_LOG):
    os.remove(SKIPPED_LOG)

def classify_category(cat_str: str) -> str:

    return "cs" if cat_str.startswith("cs.") else "non_cs"

def extract_vars(py_file: str) -> Set[str]:

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
                    vars_in_file.add(t.id) 
    return vars_in_file

def process_repo_for_single_var(repo_path: str, target_var: str) -> Tuple[str, float]:

    repo_name = os.path.basename(repo_path)
    total_files_in_repo = 0
    target_var_file_count_in_repo = 0
    
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            total_files_in_repo += 1
            vars_here = extract_vars(os.path.join(root, fn))
            if target_var in vars_here:
                target_var_file_count_in_repo += 1
    
    frequency = 0.0
    if total_files_in_repo > 0:
        frequency = target_var_file_count_in_repo / total_files_in_repo
            
    return repo_name, frequency


TARGET_VAR = "max_length"
TARGET_QUARTER = "2025Q1"
TARGET_YEAR = TARGET_QUARTER[:4]
TARGET_Q_NUM = TARGET_QUARTER[5:] # 'Q1' -> '1'


quarter_repo_cat: Dict[str, Dict[str, str]] = defaultdict(dict)
try:
    with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
        all_categories = json.load(f)
        if TARGET_QUARTER in all_categories:
            for item in all_categories[TARGET_QUARTER]:
                repo = item["link"].rstrip("/").split("/")[-1]
                quarter_repo_cat[TARGET_QUARTER][repo] = classify_category(item["categories"])
        else:
            print(f"Warning: No category data found for {TARGET_QUARTER} in {CATEGORIES_JSON}. All repos will be treated as 'non_cs'.")
except FileNotFoundError:
    print(f"Error: Category JSON file not found at {CATEGORIES_JSON}. All repos will be treated as 'non_cs'.")
    
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {CATEGORIES_JSON}. Please check file format.")
    exit()


quarter_dir = os.path.join(BASE_DIR, TARGET_YEAR, f"Q{TARGET_Q_NUM}")
if not os.path.isdir(quarter_dir):
    print(f"Error: Directory for {TARGET_QUARTER} not found: {quarter_dir}")
    exit()

repos_in_quarter_names = [d for d in os.listdir(quarter_dir) if os.path.isdir(os.path.join(quarter_dir, d))]


categorized_repo_frequencies: Dict[str, Dict[str, float]] = {
    "cs": {},
    "non_cs": {}
}

print(f"Calculating frequency for '{TARGET_VAR}' in each repository in {TARGET_QUARTER} (categorized by CS/non-CS)...")


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

    future_to_repo = {executor.submit(process_repo_for_single_var, os.path.join(quarter_dir, repo_name), TARGET_VAR): repo_name 
                      for repo_name in repos_in_quarter_names}
    
    for future in tqdm(concurrent.futures.as_completed(future_to_repo), 
                       total=len(repos_in_quarter_names), 
                       desc=f"Processing repos in {TARGET_QUARTER}"):
        repo_name_processed, frequency = future.result()

        category = quarter_repo_cat[TARGET_QUARTER].get(repo_name_processed, "non_cs")
        
        categorized_repo_frequencies[category][repo_name_processed] = frequency




output_cs_filename = os.path.join(OUT_DIR, f"{TARGET_VAR}_{TARGET_QUARTER}_cs_repo_frequencies.csv")
with open(output_cs_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["repository", "frequency"]) 
    

    sorted_repos_cs = sorted(categorized_repo_frequencies["cs"].items(), 
                             key=lambda item: item[1], reverse=True)
    
    for repo_name, freq in sorted_repos_cs:
        writer.writerow([repo_name, round(freq, 6)])
print(f"\n✅ Frequencies for '{TARGET_VAR}' in CS repositories of {TARGET_QUARTER} saved to: {output_cs_filename}")


output_noncs_filename = os.path.join(OUT_DIR, f"{TARGET_VAR}_{TARGET_QUARTER}_non_cs_repo_frequencies.csv")
with open(output_noncs_filename, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["repository", "frequency"]) 
    

    sorted_repos_noncs = sorted(categorized_repo_frequencies["non_cs"].items(), 
                                key=lambda item: item[1], reverse=True)
    
    for repo_name, freq in sorted_repos_noncs:
        writer.writerow([repo_name, round(freq, 6)])
print(f"✅ Frequencies for '{TARGET_VAR}' in non-CS repositories of {TARGET_QUARTER} saved to: {output_noncs_filename}")

print(f"Skipped files (if any) logged in: {SKIPPED_LOG}")