import os
import json

# 定义文件夹路径和年份列表
output_dir = 'LLM_code/output_by_quarter/by_file'
years = [2020, 2021, 2022, 2023, 2024, 2025]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

# 初始化最终的整合数据字典
merged_data = {}

# 遍历每一年每一季度
for year in years:
    for quarter in quarters:
        folder_path = os.path.join(output_dir, str(year), quarter)
        package_file = os.path.join(folder_path, 'python_package.json')

        if not os.path.exists(package_file):
            continue

        with open(package_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        quarter_label = f"{year}{quarter}"

        # 遍历当前json文件中的每个key，如 "torch"
        for key, value in data.items():
            if key not in merged_data:
                merged_data[key] = {
                    "count": {},       # 每季度总数
                    "methods": {}      # 每季度 method 的总数
                }

            # 保存当前季度的 count
            merged_data[key]["count"][quarter_label] = value["count"]

            # 保存当前季度各 method 的 count
            for method, method_count in value["methods"].items():
                if method not in merged_data[key]["methods"]:
                    merged_data[key]["methods"][method] = {}
                merged_data[key]["methods"][method][quarter_label] = method_count

# 计算总数
for key in merged_data:
    # 统计所有季度的 count 总和
    merged_data[key]["count_total"] = sum(merged_data[key]["count"].values())

    for method in merged_data[key]["methods"]:
        # 统计该 method 的所有季度总和
        merged_data[key]["methods"][method]["total"] = sum(
            v for k, v in merged_data[key]["methods"][method].items() if k != "total"
        )

# 按照 count_total 排序
sorted_data = dict(
    sorted(merged_data.items(), key=lambda item: item[1]["count_total"], reverse=True)
)

# 输出合并后的结果
with open('package_by_file.json', 'w', encoding='utf-8') as out_file:
    json.dump(sorted_data, out_file, ensure_ascii=False, indent=4)

print("合并并排序完成")
