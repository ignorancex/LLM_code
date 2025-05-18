import json
import csv
import re
target_terms = {'meet-in-the-middle', 'games', 'schedules', 'number theory', 'ternary search', '2-sat', 'greedy', 'brute force', 'interactive', 'math', 'chinese remainder theorem', 'sort', 'two pointers', 'flows', 'dfs', 'shortest paths', 'geometry', 'hashing', 'matrices', 'string', 'suffix', 'dp', 'fft', 'probabilities', 'implementation', 'strings', 'graphs', 'data structures', 'combinatorics', 'constructive algorithms', 'binary search', 'trees', 'expression parsing', 'dsu', 'divide and conquer', 'graph matchings', 'bitmasks', 'bfs', 'similar'}
file_names = ['qwen_32b_python_extract.json', 'qwen_32b_cpp_extract.json', 'gemma_27b_python_extract.json', 'gemma_27b_cpp_extract.json', 'deepseek_32b_python_extract.json', 'deepseek_32b_cpp_extract.json']
word_freq = {term: {} for term in target_terms}
columns = []
for file in file_names:
    model = file.split('_')[0]
    lang = 'py' if 'python' in file else 'cpp'
    col_reason = f'REASON_{model}_{lang}'
    col_ref = f'REASON_REF_{model}_{lang}'
    columns.extend([col_reason, col_ref])
    for term in target_terms:
        word_freq[term][col_reason] = 0
        word_freq[term][col_ref] = 0

def count_matches(text, term):
    return len(re.findall('\\b' + re.escape(term) + '\\b', text.lower()))
for file in file_names:
    model = file.split('_')[0]
    lang = 'py' if 'python' in file else 'cpp'
    col_reason = f'REASON_{model}_{lang}'
    col_ref = f'REASON_REF_{model}_{lang}'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        for term in target_terms:
            word_freq[term][col_reason] += count_matches(item.get('generate_reasoning', ''), term)
            word_freq[term][col_ref] += count_matches(item.get('generate_ref_reasoning', ''), term)
with open('tags_frequencies.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word'] + columns)
    for word in sorted(target_terms):
        row = [word] + [word_freq[word][col] for col in columns]
        writer.writerow(row)