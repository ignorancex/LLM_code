import json
import os

def normalize_usage(data_file, stats_file, output_file):
    # 读取总统计数据（含 count 和 methods）
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取季度的 total_py_files 数量
    with open(stats_file, 'r', encoding='utf-8') as f:
        quarter_stats = json.load(f)

    def get_file_count(q):
        key = q[:4] + '_' + q[4:]  # "2020Q1" -> "2020_Q1"
        return quarter_stats.get(key, {}).get("total_py_files", 1)  # 避免除以0

    normalized = {}

    for pkg, content in data.items():
        norm_count = {}
        for q, v in content.get("count", {}).items():
            total_files = get_file_count(q)
            norm_count[q] = round(v / total_files, 6)

        norm_methods = {}
        for method, m_q_dict in content.get("methods", {}).items():
            norm_m_q = {}
            for q, v in m_q_dict.items():
                if q == "total":
                    continue
                total_files = get_file_count(q)
                norm_m_q[q] = round(v / total_files, 6)
            norm_methods[method] = norm_m_q

        normalized[pkg] = {
            "count": norm_count,
            "methods": norm_methods
        }

    # 保存归一化后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, indent=4, ensure_ascii=False)

# 示例使用方式
if __name__ == "__main__":
    usage_data_file = "LLM_code/output_by_quarter/by_file/package_by_file.json"
    quarter_stats_file = "LLM_code/output_by_quarter/by_file/stats_summary.json"
    output_file = "LLM_code/output_by_quarter/by_file/normalized_package_usage.json"

    normalize_usage(usage_data_file, quarter_stats_file, output_file)
