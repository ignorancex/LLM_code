import json
import requests
from collections import defaultdict

def is_link_accessible(url):
    """检查链接是否可访问"""
    print(url)
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def load_existing_links(prefix):
    """加载已有的链接（来自旧文件）"""
    try:
        with open(f"LLM_code/dataset/github_links/link_20{prefix}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("github_links", []))  # 以集合形式返回，便于快速查找
    except FileNotFoundError:
        return set()  # 如果没有旧文件，返回空集合

def extract_github_links(input_file):
    """提取 GitHub 链接并写入新文件"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 目标前缀集合
    prefixes = [f"2{x:01d}" for x in range(5)]

    # 记录符合条件的链接
    links_by_prefix = defaultdict(list)

    for prefix in prefixes:
        count = 0  # 记录当前 prefix 下的链接数量
        existing_links = load_existing_links(prefix)  # 加载旧文件中已存在的链接

        for y in range(1, 13):  # y 取 01-12
            sub_prefix = f"{prefix}{y:02d}"
            y_count = 0  # 记录当前 y 下的链接数量
            for item in data:
                item_id = item.get("id", "")
                github_link = item.get("github_links", "").rstrip("\\")  # 处理单个字符串
                if item_id.startswith(sub_prefix) and github_link:
                    # 如果链接已存在于旧文件中，跳过该链接
                    if github_link in existing_links:
                        continue
                    
                    if y_count < 50 and count < 600:
                        if is_link_accessible(github_link):  # 判断链接是否可访问
                            links_by_prefix[prefix].append(github_link)
                            y_count += 1
                            count += 1
                        if y_count >= 50 or count >= 600:
                            break
                if count >= 600:
                    break
            if count >= 600:
                break

    # 写入不同文件
    for prefix, links in links_by_prefix.items():
        output_data = {"github_links": links}  # 合并所有链接成一个结构
        output_file = f"link_20{prefix}_new.json"  # 按 x 值分组写文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"提取完成：{prefix}，共 {len(output_data['github_links'])} 个 GitHub 链接，已保存至 {output_file}")

# 示例使用
extract_github_links("LLM_code/dataset/github_links/filtered_github_links.json")
