import json
import re
from tqdm import tqdm

# 输入和输出文件
input_file = "Code/github_links.json"
output_file_filtered = "Code/filtered_github_links.json"
output_file_details = "Code/Only_links.json"

# 正则表达式匹配 GitHub 仓库链接（允许可选的 .git）
github_pattern = re.compile(
    r'^https://github\.com/'
    r'([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)/'  # GitHub 用户名或组织名
    r'([\w.-]+)(?:\.git)?$'  # 仓库名，允许 .git 结尾
)

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 存储两个不同的 JSON 结果
filtered_data = []  # 存储原始结构
details_data = []  # 存储拆分 user/repo/update_date 结构

# 处理数据
for obj in tqdm(data, desc="Processing records", ncols=80):
    github_link = obj.get("github_links", "").strip().rstrip('/;')  # 直接作为字符串处理

    # 进行正则匹配
    match = github_pattern.match(github_link)
    if match:
        user, repo = match.groups()  # 提取用户名和仓库名
        
        # 如果链接以 .git 结尾，去掉 .git
        if github_link.endswith('.git'):
            github_link = github_link[:-4]
        
        # 更新对象，并保留该项
        obj["github_links"] = github_link
        filtered_data.append(obj)

        # 生成包含 user/repo/update_date 结构的 JSON 数据
        details_data.append({
            "github_links": github_link,
            "user": user,
            "repo": repo,
            "update_date": obj.get("update_date", "N/A")  # 直接提取 update_date，若不存在则填 "N/A"
        })

# 保存 `filtered_github_links.json`
with open(output_file_filtered, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

# 保存 `github_links_details.json`
with open(output_file_details, "w", encoding="utf-8") as f:
    json.dump(details_data, f, ensure_ascii=False, indent=4)

# 输出统计信息
total_details = len(details_data)
print(f"Filtered data saved to {output_file_filtered}")
print(f"GitHub details saved to {output_file_details}")
print(f"Total valid GitHub records: {total_details}")
