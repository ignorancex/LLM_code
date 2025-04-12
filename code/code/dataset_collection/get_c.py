import json
import requests
from urllib.parse import urljoin
from tqdm import tqdm
import time
import os

# ===== 填写你的 GitHub Token =====
GITHUB_TOKEN = ""
# =================================

HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

def has_cpp_files(repo_url):
    """判断一个GitHub仓库是否含有至少一个.c/.cpp文件"""
    try:
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        user_repo = '/'.join(repo_url.split('/')[-2:])
        api_url = f"https://api.github.com/repos/{user_repo}/git/trees/HEAD?recursive=1"

        response = requests.get(api_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"[跳过] 请求失败 {repo_url}, 状态码: {response.status_code}")
            return False

        tree = response.json().get("tree", [])
        for item in tree:
            if item["type"] == "blob" and (item["path"].endswith(".c") or item["path"].endswith(".cpp")):
                return True
        return False
    except Exception as e:
        print(f"[跳过] 检查错误 {repo_url}: {e}")
        return False

def filter_cpp_repos_incremental(input_path, output_path):
    # 加载原始输入数据
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # 如果已有部分结果，尝试加载
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            result_data = json.load(f)
    else:
        result_data = {}

    for quarter, urls in input_data.items():
        print(f"\n处理季度：{quarter}")
        done_urls = set(result_data.get(quarter, []))
        cpp_repos = result_data.get(quarter, [])

        for url in tqdm(urls, desc=f"{quarter}", unit="repo"):
            if url in done_urls:
                continue  # 已完成

            if has_cpp_files(url):
                cpp_repos.append(url)

            # 写入进度（写回整个 json）
            result_data[quarter] = cpp_repos
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            time.sleep(0.5)  # 控制请求频率

    print("\n✅ 所有季度处理完毕！结果已保存到", output_path)

# === 调用 ===
filter_cpp_repos_incremental("LLM_code/code/github_links/valid_links_by_quarter_new.json", 
                             "LLM_code/code/github_links/cpp_repos.json")
