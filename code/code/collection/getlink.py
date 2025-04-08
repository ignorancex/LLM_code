# import json
# from collections import defaultdict
#
#
# def extract_github_links(input_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     # 目标前缀集合
#     prefixes = [f"2{x:01d}" for x in range(5)]
#
#     # 记录符合条件的链接
#     links_by_prefix = defaultdict(list)
#
#     for prefix in prefixes:
#         count = 0  # 记录当前 prefix 下的链接数量
#         for y in range(1, 13):  # y 取 01-12
#             sub_prefix = f"{prefix}{y:02d}"
#             y_count = 0  # 记录当前 y 下的链接数量
#             for item in data:
#                 item_id = item.get("id", "")
#                 if item_id.startswith(sub_prefix) and "github_links" in item:
#                     cleaned_links = [link.rstrip("\\") for link in item["github_links"]]  # 去除末尾反斜杠
#                     for link in cleaned_links:
#                         if y_count < 100 and count < 1200:
#                             links_by_prefix[prefix].append(link)
#                             y_count += 1
#                             count += 1
#                         if y_count >= 100 or count >= 1200:
#                             break
#                 if count >= 1200:
#                     break
#             if count >= 1200:
#                 break
#
#     # 写入不同文件
#     for prefix, links in links_by_prefix.items():
#         output_data = {"github_links": links}  # 合并所有链接成一个结构
#         output_file = f"link_20{prefix}.json"  # 按 x 值分组写文件
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(output_data, f, indent=4)
#         print(f"提取完成：{prefix}，共 {len(output_data['github_links'])} 个 GitHub 链接，已保存至 {output_file}")
#
#
# # 示例使用
# extract_github_links("github_links.json")


import json
from collections import defaultdict


def extract_github_links(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 目标前缀集合
    prefixes = [f"2{x:01d}" for x in range(5)]

    # 记录符合条件的链接
    links_by_prefix = defaultdict(list)

    for prefix in prefixes:
        count = 0  # 记录当前 prefix 下的链接数量
        for y in range(1, 13):  # y 取 01-12
            sub_prefix = f"{prefix}{y:02d}"
            y_count = 0  # 记录当前 y 下的链接数量
            for item in data:
                item_id = item.get("id", "")
                if item_id.startswith(sub_prefix) and "github_links" in item:
                    cleaned_links = [link.rstrip("\\") for link in item["github_links"]]
                    if cleaned_links:  # 确保列表非空
                        first_link = cleaned_links[0]  # 只取第一个链接
                        if y_count < 100 and count < 1200:
                            links_by_prefix[prefix].append(first_link)
                            y_count += 1
                            count += 1
                        if y_count >= 100 or count >= 1200:
                            break
                if count >= 1200:
                    break
            if count >= 1200:
                break

    # 写入不同文件
    for prefix, links in links_by_prefix.items():
        output_data = {"github_links": links}  # 合并所有链接成一个结构
        output_file = f"link_20{prefix}.json"  # 按 x 值分组写文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"提取完成：{prefix}，共 {len(output_data['github_links'])} 个 GitHub 链接，已保存至 {output_file}")


# 示例使用
extract_github_links("github_links.json")
