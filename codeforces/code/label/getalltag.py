import urllib.request
from bs4 import BeautifulSoup
import json
import os


# 爬取单个题目的标签
def get_tags(fullname):
    problemSet = ''.join(filter(str.isdigit, fullname))  # 比赛编号：数字部分
    problemId = ''.join(filter(str.isalpha, fullname))  # 题号：字母部分

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        # 题目链接
        url = f"https://codeforces.com/problemset/problem/{problemSet}/{problemId}"
        # 获取网页内容
        req = urllib.request.Request(url=url, headers=headers)
        html = urllib.request.urlopen(req).read()
        # 用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(html, "html.parser")
        # 找出所有 class 为 tag-box 的 span 元素，并提取其文本（去除前后空白）
        tags = [span.get_text(strip=True) for span in soup.find_all("span", class_="tag-box")]
        titles = [span["title"] for span in soup.find_all("span", class_="tag-box")]
        return tags,titles
    except Exception as e:
        print(f"爬取 {fullname} 失败: {e}")
        return None,None  # 返回 None 表示爬取失败


# 读取已有jsonl中的已完成题目
def load_finished(filepath):
    finished = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    finished.add(data['problem'])
                except json.JSONDecodeError:
                    continue
    return finished


# 主函数
def process(input_json, output_jsonl):
    # 读原始数据
    with open(input_json, 'r', encoding='utf-8') as f:
        problems_data = json.load(f)

    # 读已完成数据
    finished = load_finished(output_jsonl)
    print(f"已有{len(finished)}道题完成，继续处理未完成的题目...")

    with open(output_jsonl, 'a', encoding='utf-8') as out_f:
        for item in problems_data:
            fullname = item.get('fullname')
            if not fullname:
                continue

            if fullname in finished:
                print(f"{fullname} 已存在，跳过。")
                continue

            tags,titles = get_tags(fullname)

            if tags is None:
                print(f"{fullname} 爬取失败，跳过写入。")
                continue
            if len(tags) == 0:
                print(f"{fullname} 没有标签，跳过写入。")
                continue

            result = {
                "problem": fullname,
                "tags": tags,
                "tags_titles": titles
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_f.flush()  # 每写一行就 flush
            print(f"{fullname} 处理完成，标签数: {len(tags)}")


if __name__ == "__main__":
    input_json = 'gemma_27b_python.json'  # 输入文件，包含多个{..., "fullname": "581B", ...}结构
    output_jsonl = 'tags_py.jsonl'  # 输出文件
    process(input_json, output_jsonl)
