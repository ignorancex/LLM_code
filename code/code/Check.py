import os
import json

year = "2022"
output_original_links = True  # 设为 True 输出原始链接，设为 False 输出 repo 名称

# 定义仓库存储的根目录
repo_base_path = f"github_code/{year}"

# 定义 JSON 文件路径
json_file_path = f"link_{year}_new.json"

# 读取 JSON 文件
with open(json_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 提取 JSON 文件中的 repo 名称和链接（保持顺序）
json_links = data["github_links"]
json_repo_map = [(url.rstrip("/").split("/")[-1], url) for url in json_links]

# 获取本地文件夹的 repo 名称
if os.path.exists(repo_base_path):
    local_repos = {repo for repo in os.listdir(repo_base_path) if os.path.isdir(os.path.join(repo_base_path, repo))}
else:
    local_repos = set()

# 计算多余的文件夹
json_repo_names_set = {name for name, _ in json_repo_map}
extra_folders = sorted(local_repos - json_repo_names_set)  # 仅比较 repo 名称

# 计算缺少的 repo，保持原始顺序
missing_repos = []
for name, url in json_repo_map:
    if name not in local_repos:
        if output_original_links:
            missing_repos.append(url)
        else:
            missing_repos.append(name)

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
