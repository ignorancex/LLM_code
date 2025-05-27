import os
import json
from collections import defaultdict

# 1. 设定路径
input_path  = 'LLM_code/arxiv_result/naming_patterns_split/naming_patterns_split.json'
output_dir  = 'LLM_code/arxiv_result/naming_patterns_split'
output_path = os.path.join(output_dir, 'naming_patterns_agg_across_repos.json')

# 2. 读取已经输出的列表
with open(input_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

# 3. 按季度、分组、对象收集所有仓库的比例值
agg = defaultdict(lambda: {
    'fewer': {'func': defaultdict(list), 'var': defaultdict(list)},
    'more':  {'func': defaultdict(list), 'var': defaultdict(list)},
})

for entry in results:
    q = entry['quarter']
    for grp in ('fewer', 'more'):
        for kind in ('func', 'var'):
            for pat, val in entry[grp][kind].items():
                agg[q][grp][kind][pat].append(val)

# 4. 对收集到的列表再求一次平均
final = {}
for q, groups in agg.items():
    final[q] = {}
    for grp, kinds in groups.items():
        final[q][grp] = {}
        for kind, pats in kinds.items():
            # pat: [v1, v2, ...] → 平均
            final[q][grp][kind] = {
                pat: round(sum(vs) / len(vs), 6) if vs else 0.0
                for pat, vs in pats.items()
            }

# 5. 写回 JSON
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print(f"Cross-repo aggregated results saved to {output_path}")
