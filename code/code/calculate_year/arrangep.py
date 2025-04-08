import os
import json

# 定义文件夹路径和年份列表
output_dir = 'output_package_asone'
years = [2020, 2021, 2022, 2023, 2024, 2025]

# 初始化最终的整合数据字典
merged_data = {}

# 遍历每一年
for year in years:
    folder_path = os.path.join(output_dir, str(year))
    package_file = os.path.join(folder_path, 'package.json')

    # 读取对应年份的package.json
    with open(package_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历当前json文件中的每个key，如 "torch"
    for key, value in data.items():
        if key not in merged_data:
            merged_data[key] = {"count": [], "methods": {}}

        # 添加当前年份的count值到merged_data
        merged_data[key]["count"].append({year: value["count"]})

        # 遍历methods中的每个method
        for method, method_count in value["methods"].items():
            if method not in merged_data[key]["methods"]:
                merged_data[key]["methods"][method] = {}
            # 将每个method的count添加到merged_data
            merged_data[key]["methods"][method][year] = method_count

# 计算总数并排序
# 计算每个method的总数
for key in merged_data:
    merged_data[key]["count_total"] = sum(
        [x.get(year, 0) for year in range(2020, 2026) for x in merged_data[key]["count"]])

    for method in merged_data[key]["methods"]:
        merged_data[key]["methods"][method]["total"] = sum(merged_data[key]["methods"][method].values())

# 根据count_total排序
sorted_data = dict(sorted(merged_data.items(), key=lambda item: item[1]["count_total"], reverse=True))

# 输出合并后的结果
with open('package_byproject_asone.json', 'w', encoding='utf-8') as out_file:
    json.dump(sorted_data, out_file, ensure_ascii=False, indent=4)

