import os
from tqdm import tqdm

base_dir = 'LLM_code/arxiv_dataset'
years = [str(y) for y in range(2020, 2026)]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

def has_file_with_extensions(root_path, extensions):
    """递归检查是否有指定后缀的文件"""
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if any(f.endswith(ext) for ext in extensions):
                return True
    return False

def delete_files_with_extensions(root_path, extensions):
    """递归删除所有指定后缀的文件"""
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if any(f.endswith(ext) for ext in extensions):
                try:
                    os.remove(os.path.join(dirpath, f))
                    print(f"🗑️ Deleted: {os.path.join(dirpath, f)}")
                except Exception as e:
                    print(f"❌ Error deleting {f}: {e}")

# 统计所有仓库路径
all_repos = []
for year in years:
    for quarter in quarters:
        quarter_path = os.path.join(base_dir, year, quarter)
        if not os.path.exists(quarter_path):
            continue
        for repo in os.listdir(quarter_path):
            repo_path = os.path.join(quarter_path, repo)
            if os.path.isdir(repo_path):
                all_repos.append(repo_path)

# 遍历所有仓库，显示 tqdm 进度
for repo_path in tqdm(all_repos, desc="Checking repositories"):
    time_info_py = os.path.join(repo_path, 'time_info.txt')
    time_info_cpp = os.path.join(repo_path, 'time_info_cpp.txt')

    # 检查 .py 文件
    if has_file_with_extensions(repo_path, ['.py']) and not os.path.exists(time_info_py):
        tqdm.write(f"⚠️ Deleting .py files in: {repo_path} (missing time_info.txt)")
        delete_files_with_extensions(repo_path, ['.py'])

    # 检查 .c / .cpp 文件
    if has_file_with_extensions(repo_path, ['.c', '.cpp']) and not os.path.exists(time_info_cpp):
        tqdm.write(f"⚠️ Deleting .c/.cpp files in: {repo_path} (missing time_info_cpp.txt)")
        delete_files_with_extensions(repo_path, ['.c', '.cpp'])
