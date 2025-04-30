import os
import json
from collections import defaultdict

# 原始 JSON 文件路径
input_json_path = "LLM_code/code/github_links/cpp_dataset_links_new.json"
output_json_path = "LLM_code/code/github_links/cpp_dataset_links_1.json"

# 加载原始 JSON 数据
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 存储清理后的数据
new_data = {}
duplicate_repos = defaultdict(list)

print("📊 各季度清理后链接数量：")

for key, url_dicts in data.items():
    # 提取 year 和 season
    year = key[:4]
    season = key[4:]

    base_path = os.path.join("LLM_code/arxiv_dataset_cpp", year, season)
    seen = set()
    cleaned_url_dicts = []

    for url_dict in url_dicts:
        link = url_dict.get("link")
        if not link:
            continue  # 没有 link 字段，跳过

        # 提取仓库名
        parts = link.rstrip("/").split("/")
        if len(parts) < 2:
            continue  # 非法链接，跳过
        repo_name = parts[-1]

        repo_path = os.path.join(base_path, repo_name)

        # 检查路径是否存在
        if os.path.exists(repo_path):
            cleaned_url_dicts.append(url_dict)
            if repo_name in seen:
                duplicate_repos[key].append(repo_name)
            seen.add(repo_name)
        else:
            print(f"❌ 不存在的仓库：{repo_path}（已从输出中移除）")

    if cleaned_url_dicts:
        new_data[key] = cleaned_url_dicts
        print(f"  {key}: {len(cleaned_url_dicts)} 个链接")
    else:
        print(f"  {key}: 0 个链接（全部无效或缺失）")

# 将清理后的结果写入新文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 新文件已生成：{output_json_path}")

# 输出重名仓库信息
if duplicate_repos:
    print("\n⚠️ 以下季度中存在仓库重名：")
    for key, names in duplicate_repos.items():
        print(f"  {key}: {names}")
else:
    print("\n✅ 没有发现仓库重名。")
