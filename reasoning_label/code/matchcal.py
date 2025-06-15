import json
import jsonlines
import re
import csv
json_file = 'gemma_27b_cpp_extract.json'
jsonl_file = 'tags_cpp.jsonl'
csv_file = 'gemma27b_cpp_ans_match_report.csv'
chose = 'generate_reasoning'
with open(json_file, 'r', encoding='utf-8') as f:
    submissions = json.load(f)
problem_tags = {}
with jsonlines.open(jsonl_file, 'r') as reader:
    for obj in reader:
        problem = obj.get('problem')
        tags = [t for t in obj.get('tags', []) if not t.startswith('*')]
        problem_tags[problem] = tags
special_rules = {'dfs and similar': ['dfs', 'similar'], 'divide and conquer': ['divide'], 'expression parsing': ['parsing'], 'string suffix structures': ['suffix']}
all_tags = set((tag for tags in problem_tags.values() for tag in tags))
tag_regex = {}
for tag in all_tags:
    terms = special_rules.get(tag, [tag])
    tag_regex[tag] = [re.compile(f'\\b{re.escape(term)}\\b', re.IGNORECASE) for term in terms]
results = []
match_true = 0
match_false = 0
error_true = 0
error_false = 0
for s in submissions:
    sid = s.get('submission_id')
    prob = s.get('fullname')
    reasoning = s.get(chose, '')
    tags = problem_tags.get(prob)
    tag_str = '' if tags is None else ';'.join(tags)
    row = [sid, prob, tag_str]
    if tags is None:
        row.extend(['', '', '', ''])
    else:
        matched = [t for t in tags if any((rx.search(reasoning) for rx in tag_regex[t]))]
        success = bool(matched)
        if success:
            match_true += 1
        else:
            match_false += 1
        wrong = [t for t in all_tags - set(tags) if any((rx.search(reasoning) for rx in tag_regex[t]))]
        error = bool(wrong)
        if error:
            error_true += 1
        else:
            error_false += 1
        row.append(str(success))
        row.append(';'.join(matched))
        row.append(str(error))
        row.append(';'.join(wrong))
    results.append(row)
match_rate = match_true / (match_true + match_false) if match_true + match_false > 0 else 0
error_rate = error_true / (error_true + error_false) if error_true + error_false > 0 else 0
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['submission_id', 'problem', 'tags', 'match_success', 'matched_tags', 'error_match', 'error_tags'])
    writer.writerows(results)
    writer.writerow([])
    writer.writerow(['', '', 'match_rate', f'{match_true}/{match_true + match_false}', f'{match_rate:.2%}'])
    writer.writerow(['', '', 'error_rate', f'{error_true}/{error_true + error_false}', f'{error_rate:.2%}'])