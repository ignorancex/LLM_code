import os
import ast
import csv
import json
import warnings
import concurrent.futures # 导入并行处理模块
from collections import Counter, defaultdict
from typing import Set, Dict, Tuple, List
from tqdm import tqdm

# 忽略 SyntaxWarning，因为它可能由解析某些代码文件引起
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ---------- 0. 路径与常量 ----------
# 假设 arxiv_result 目录与 arxiv_dataset 目录在同一层级
BASE_DATA_DIR = "arxiv_dataset" # 基础数据目录
CATEGORIES_JSON = "dataset_collection/github/links/python_dataset_links.json"
OUT_DIR = "LLM_code/arxiv_result/vars" # 输出目录
os.makedirs(OUT_DIR, exist_ok=True) # 确保输出目录存在

# 定义所有需要处理的季度
ALL_QUARTERS = [f"{y}Q{q}" for y in range(2025, 2026) for q in range(2, 4)]

# 定义并行处理的最大工作线程数
MAX_WORKERS = os.cpu_count() or 4 # 默认为CPU核心数，至少为4

# ---------- 1. 工具函数 ----------

def classify_category(cat_str: str) -> str:
    """根据 arXiv 主分类字符串判断是 'cs' 还是 'non_cs'。"""
    return "cs" if cat_str.startswith("cs.") else "non_cs"

def extract_variables_from_file(file_path: str, skipped_log_path: str) -> List[str]:
    """
    从 Python 文件中提取变量名。
    包括赋值、函数参数和 for 循环目标。
    现在接受 skipped_log_path 参数。
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: # 增加 errors='ignore' 处理编码问题
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
    """
    遍历仓库，提取所有 Python 文件中的变量。
    接受一个包含 (repo_name, repo_path, skipped_log_path) 的元组作为输入。
    返回 (repo_name, category, 提取到的变量列表)。
    """
    repo_name, repo_path, skipped_log_path, category = repo_info # 解包传入的元组
    
    repo_variables = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                variables = extract_variables_from_file(file_path, skipped_log_path)
                repo_variables.extend(variables)
    
    return repo_name, category, repo_variables # 返回仓库名、类别和变量列表

def write_to_csv(variable_counter: Counter, repo_count_dict: Dict[str, int], output_file: str):
    """将变量统计结果写入 CSV 文件。"""
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Variable', 'TotalFrequency', 'RepoCount'])
        for variable in sorted(variable_counter.keys()):
            writer.writerow([variable, variable_counter[variable], repo_count_dict[variable]])
    print(f"✅ Variable statistics written to {output_file}")


# ---------- 2. 主程序 ----------
if __name__ == '__main__':
    # 加载完整的分类信息一次
    full_categories_data: Dict[str, List[Dict[str, str]]] = {}
    if not os.path.exists(CATEGORIES_JSON):
        print(f"错误：分类 JSON 文件 '{CATEGORIES_JSON}' 不存在。请检查路径。")
        exit()
    try:
        with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
            full_categories_data = json.load(f)
    except Exception as e:
        print(f"读取或解析分类 JSON 文件时发生错误：{e}")
        exit()


    print("Starting batch processing for all quarters...")

    # 使用外部 tqdm 包装整个季度循环，显示总进度
    for current_quarter_key in tqdm(ALL_QUARTERS, desc="Overall Progress"):
        print(f"\n--- Processing {current_quarter_key} ---")

        # 从 quarter_key 解析年份和季度数，构建 target_directory
        year = current_quarter_key[:4]
        quarter_num = current_quarter_key[4:] # 例如 'Q1'
        target_directory = os.path.join(BASE_DATA_DIR, year, quarter_num)

        # 为当前季度设置专属的 SKIPPED_LOG
        SKIPPED_LOG_CURRENT_QUARTER = os.path.join(OUT_DIR, f"skipped_files_{current_quarter_key}.txt")
        if os.path.exists(SKIPPED_LOG_CURRENT_QUARTER):
            os.remove(SKIPPED_LOG_CURRENT_QUARTER) # 每次处理前清空日志

        if not os.path.isdir(target_directory):
            print(f"警告：目录 '{target_directory}' 不存在，跳过此季度。")
            continue

        # 获取当前季度的仓库分类数据
        quarter_repo_cat_mapping: Dict[str, str] = defaultdict(str) # 仓库名 -> 类别字符串
        if current_quarter_key in full_categories_data:
            for item in full_categories_data[current_quarter_key]:
                repo = item["link"].rstrip("/").split("/")[-1]
                quarter_repo_cat_mapping[repo] = classify_category(item["categories"])
        else:
            print(f"警告：JSON 文件中未找到 '{current_quarter_key}' 的分类数据。此季度所有仓库将默认为 'non_cs'。")

        # 初始化分类计数器
        cs_variable_counter = Counter()
        non_cs_variable_counter = Counter()

        cs_repo_counter = defaultdict(set)
        non_cs_repo_counter = defaultdict(set)

        # 获取目标目录下的所有仓库名
        all_repo_names = [repo for repo in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, repo))]
        total_repos_in_quarter = len(all_repo_names)

        if total_repos_in_quarter == 0:
            print(f"信息：目录 '{target_directory}' 中没有找到任何仓库，跳过此季度。")
            continue

        print(f"Found {total_repos_in_quarter} repositories in {current_quarter_key}. Starting parallel processing...")

        # 准备提交给线程池的任务列表
        # 每个任务是一个元组 (repo_name, repo_path, skipped_log_path, category)
        repo_tasks = []
        for repo_name in all_repo_names:
            repo_path = os.path.join(target_directory, repo_name)
            category = quarter_repo_cat_mapping.get(repo_name, "non_cs") # 获取仓库类别
            repo_tasks.append((repo_name, repo_path, SKIPPED_LOG_CURRENT_QUARTER, category))

        # 使用 ThreadPoolExecutor 进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_repo = {executor.submit(process_repo, task_info): task_info[0] for task_info in repo_tasks}

            # 遍历已完成的任务，并更新计数器
            for future in tqdm(concurrent.futures.as_completed(future_to_repo),
                               total=len(future_to_repo),
                               desc=f"  Scanning repos in {current_quarter_key} (Parallel)",
                               leave=False):
                repo_name_completed = future_to_repo[future]
                try:
                    repo_name_result, category_result, repo_variables = future.result()
                    
                    if category_result == "cs":
                        cs_variable_counter.update(repo_variables)
                        for var in set(repo_variables): # 使用 set 去重，统计仓库中出现过即可
                            cs_repo_counter[var].add(repo_name_result)
                    else: # non_cs
                        non_cs_variable_counter.update(repo_variables)
                        for var in set(repo_variables):
                            non_cs_repo_counter[var].add(repo_name_result)
                except Exception as exc:
                    print(f'Repository {repo_name_completed} generated an exception: {exc}')
                    # 可以在这里记录到 SKIPPED_LOG_CURRENT_QUARTER，但 process_repo 内部已经记录了文件级别的错误

        # 转换 repo_counter 的集合为数量
        cs_repo_count_result = {var: len(repos) for var, repos in cs_repo_counter.items()}
        non_cs_repo_count_result = {var: len(repos) for var, repos in non_cs_repo_counter.items()} # Fix: non_cs_repo_counter

        # 输出结果到各自的 CSV 文件
        write_to_csv(cs_variable_counter, cs_repo_count_result, os.path.join(OUT_DIR, f'variable_{current_quarter_key}_cs.csv'))
        write_to_csv(non_cs_variable_counter, non_cs_repo_count_result, os.path.join(OUT_DIR, f'variable_{current_quarter_key}_non_cs.csv'))

    print("\n--- All quarterly variable statistics by category have been generated! ---")