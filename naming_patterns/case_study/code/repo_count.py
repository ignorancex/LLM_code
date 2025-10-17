import os
import ast
import csv
import json
import warnings
import concurrent.futures
from collections import Counter, defaultdict
from typing import Set, Dict, Tuple, List
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SyntaxWarning)

BASE_DATA_DIR = "arxiv_dataset" 
CATEGORIES_JSON = "dataset_collection/github/links/python_dataset_links.json"
OUT_DIR = "LLM_code/arxiv_result/vars"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_QUARTERS = [f"{y}Q{q}" for y in range(2025, 2026) for q in range(2, 4)]

MAX_WORKERS = os.cpu_count() or 4 


def classify_category(cat_str: str) -> str:
    return "cs" if cat_str.startswith("cs.") else "non_cs"

def extract_variables_from_file(file_path: str, skipped_log_path: str) -> List[str]:

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: 
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except Exception as e:
            with open(skipped_log_path, "a", encoding="utf-8") as lg:
                lg.write(f"Error parsing {file_path}: {e}\n")
            return []

    variables = []

    class VariableVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.AugStore)):
                variables.append(node.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            for arg in node.args.args:
                variables.append(arg.arg)
            self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name):
                variables.append(node.target.id)
            self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    variables.append(item.optional_vars.id)
            self.generic_visit(node)

    VariableVisitor().visit(tree)
    return variables

def process_repo(repo_info: Tuple[str, str, str]) -> Tuple[str, str, List[str]]:

    repo_name, repo_path, skipped_log_path, category = repo_info 
    
    repo_variables = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                variables = extract_variables_from_file(file_path, skipped_log_path)
                repo_variables.extend(variables)
    
    return repo_name, category, repo_variables 

def write_to_csv(variable_counter: Counter, repo_count_dict: Dict[str, int], output_file: str):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Variable', 'TotalFrequency', 'RepoCount'])
        for variable in sorted(variable_counter.keys()):
            writer.writerow([variable, variable_counter[variable], repo_count_dict[variable]])
    print(f"✅ Variable statistics written to {output_file}")


if __name__ == '__main__':

    full_categories_data: Dict[str, List[Dict[str, str]]] = {}
    if not os.path.exists(CATEGORIES_JSON):
        exit()
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            full_categories_data = json.load(f)
    except Exception as e:
        exit()


    print("Starting batch processing for all quarters...")

    for current_quarter_key in tqdm(ALL_QUARTERS, desc="Overall Progress"):
        print(f"\n--- Processing {current_quarter_key} ---")

        year = current_quarter_key[:4]
        quarter_num = current_quarter_key[4:] 
        target_directory = os.path.join(BASE_DATA_DIR, year, quarter_num)

        SKIPPED_LOG_CURRENT_QUARTER = os.path.join(OUT_DIR, f"skipped_files_{current_quarter_key}.txt")
        if os.path.exists(SKIPPED_LOG_CURRENT_QUARTER):
            os.remove(SKIPPED_LOG_CURRENT_QUARTER) 

        if not os.path.isdir(target_directory):
            continue

        quarter_repo_cat_mapping: Dict[str, str] = defaultdict(str) 
        if current_quarter_key in full_categories_data:
            for item in full_categories_data[current_quarter_key]:
                repo = item["link"].rstrip("/").split("/")[-1]
                quarter_repo_cat_mapping[repo] = classify_category(item["categories"])
        else:
            print(f"Error")

        cs_variable_counter = Counter()
        non_cs_variable_counter = Counter()

        cs_repo_counter = defaultdict(set)
        non_cs_repo_counter = defaultdict(set)

        all_repo_names = [repo for repo in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, repo))]
        total_repos_in_quarter = len(all_repo_names)

        if total_repos_in_quarter == 0:
            continue

        print(f"Found {total_repos_in_quarter} repositories in {current_quarter_key}. Starting parallel processing...")

        repo_tasks = []
        for repo_name in all_repo_names:
            repo_path = os.path.join(target_directory, repo_name)
            category = quarter_repo_cat_mapping.get(repo_name, "non_cs") 
            repo_tasks.append((repo_name, repo_path, SKIPPED_LOG_CURRENT_QUARTER, category))

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_repo = {executor.submit(process_repo, task_info): task_info[0] for task_info in repo_tasks}

            for future in tqdm(concurrent.futures.as_completed(future_to_repo),
                               total=len(future_to_repo),
                               desc=f"  Scanning repos in {current_quarter_key} (Parallel)",
                               leave=False):
                repo_name_completed = future_to_repo[future]
                try:
                    repo_name_result, category_result, repo_variables = future.result()
                    
                    if category_result == "cs":
                        cs_variable_counter.update(repo_variables)
                        for var in set(repo_variables): 
                            cs_repo_counter[var].add(repo_name_result)
                    else: # non_cs
                        non_cs_variable_counter.update(repo_variables)
                        for var in set(repo_variables):
                            non_cs_repo_counter[var].add(repo_name_result)
                except Exception as exc:
                    print(f'Repository {repo_name_completed} generated an exception: {exc}')

        cs_repo_count_result = {var: len(repos) for var, repos in cs_repo_counter.items()}
        non_cs_repo_count_result = {var: len(repos) for var, repos in non_cs_repo_counter.items()} # Fix: non_cs_repo_counter

        write_to_csv(cs_variable_counter, cs_repo_count_result, os.path.join(OUT_DIR, f'variable_{current_quarter_key}_cs.csv'))
        write_to_csv(non_cs_variable_counter, non_cs_repo_count_result, os.path.join(OUT_DIR, f'variable_{current_quarter_key}_non_cs.csv'))

    print("\n--- All quarterly variable statistics by category have been generated! ---")