import os
import json

year = "2025"
output_original_links = True  # 设为 True 输出原始链接，设为 False 输出 repo 名称

# 定义仓库存储的根目录
repo_base_path = f"LLM_code/code_analyze/dataset/github_code/{year}"

# 定义 JSON 文件路径
json_file_path = f"LLM_code/code_analyze/dataset/github_links/link_{year}.json"

# 读取 JSON 文件
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 提取 JSON 文件中的 repo 名称和链接
json_repo_map = {url.rstrip("/").split("/")[-1]: url for url in data["github_links"]}

# 选择是否使用原始链接
if output_original_links:
    json_repos = set(data["github_links"])  # 直接使用完整链接
    json_repo_names = set(json_repo_map.keys())  # 额外存储 repo 名称
else:
    json_repos = set(json_repo_map.keys())  # 仅使用 repo 名称

# 获取本地文件夹的 repo 名称
if os.path.exists(repo_base_path):
    local_repos = {repo for repo in os.listdir(repo_base_path) if os.path.isdir(os.path.join(repo_base_path, repo))}
else:
    local_repos = set()

# 计算多余的文件夹
extra_folders = sorted(local_repos - json_repo_map.keys())  # 仅比较 repo 名称

# 计算缺少的 repo
if output_original_links:
    missing_repos = sorted([url for name, url in json_repo_map.items() if name not in local_repos])
else:
    missing_repos = sorted(json_repos - local_repos)

# 生成结果 JSON
result = {
    "extra_folders": extra_folders,
    "missing_repos": missing_repos
}

# 保存到 JSON 文件
output_file = f"repo_check_{year}.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=4, ensure_ascii=False)

print(f"Check complete. Results saved in '{output_file}'.")
