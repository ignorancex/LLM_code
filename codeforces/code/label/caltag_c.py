import json
import csv
import re

# 目标词语/词组集合
target_terms = {
    'meet-in-the-middle', 'games', 'schedules', 'number theory', 'ternary search', '2-sat', 'greedy',
    'brute force', 'interactive', 'math', 'chinese remainder theorem', 'sort', 'two pointers',
    'flows', 'dfs', 'shortest paths', 'geometry', 'hashing', 'matrices', 'string', 'suffix',
    'dp', 'fft', 'probabilities', 'implementation', 'strings', 'graphs', 'data structures', 'combinatorics',
    'constructive algorithms', 'binary search', 'trees', 'expression parsing', 'dsu', 'divide and conquer',
    'graph matchings', 'bitmasks', 'bfs', 'similar'
}

file_names = [
    'qwen_32b_python_extract.json', 'qwen_32b_cpp_extract.json',
    'gemma_27b_python_extract.json', 'gemma_27b_cpp_extract.json',
    'deepseek_32b_python_extract.json', 'deepseek_32b_cpp_extract.json'
]

# 初始化结构
word_occurrence = {term: {} for term in target_terms}
columns = []

for file in file_names:
    model = file.split('_')[0]
    lang = 'py' if 'python' in file else 'cpp'
    col_reason = f'REASON_{model}_{lang}'
    col_ref = f'REASON_REF_{model}_{lang}'
    columns.extend([col_reason, col_ref])
    for term in target_terms:
        word_occurrence[term][col_reason] = 0
        word_occurrence[term][col_ref] = 0

# 检查是否出现（只计一次）
def term_in_text(term, text):
    return bool(re.search(r'\b' + re.escape(term) + r'\b', text.lower()))

# 遍历文件统计是否出现
for file in file_names:
    model = file.split('_')[0]
    lang = 'py' if 'python' in file else 'cpp'
    col_reason = f'REASON_{model}_{lang}'
    col_ref = f'REASON_REF_{model}_{lang}'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        reasoning = item.get('generate_reasoning', '')
        reasoning_ref = item.get('generate_ref_reasoning', '')
        for term in target_terms:
            if term_in_text(term, reasoning):
                word_occurrence[term][col_reason] += 1
            if term_in_text(term, reasoning_ref):
                word_occurrence[term][col_ref] += 1

# 写入 CSV
with open('tags_frequencies_count.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word'] + columns)
    for word in sorted(target_terms):
        row = [word] + [word_occurrence[word][col] for col in columns]
        writer.writerow(row)

print("✅ 完成写入 word_frequencies_occurrence.csv")
