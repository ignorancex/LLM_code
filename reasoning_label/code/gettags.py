import json
import jsonlines
import re
import csv
with open('qwen_32b_python_extract.json', 'r', encoding='utf-8') as f:
    submissions = json.load(f)
problem_tags = {}
with jsonlines.open('tags_py.jsonl', 'r') as reader:
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
summary_total = 0
summary_success = 0
summary_error_total = 0
summary_error = 0
for s in submissions:
    sid = s.get('submission_id')
    prob = s.get('fullname')
    reasoning = s.get('generate_reasoning', '')
    tags = problem_tags.get(prob)
    tag_str = ''
    if tags is not None:
        tag_str = ';'.join(tags)
    row = [sid, prob, tag_str]
    if tags is None:
        row.extend(['', '', '', ''])
    else:
        summary_total += 1
        matched = []
        for tag in tags:
            for rx in tag_regex[tag]:
                if rx.search(reasoning):
                    matched.append(tag)
                    break
        success = bool(matched) or len(tags) == 0
        if success:
            summary_success += 1
        wrong = []
        for tag in all_tags - set(tags):
            for rx in tag_regex[tag]:
                if rx.search(reasoning):
                    wrong.append(tag)
                    break
        error = bool(wrong)
        summary_error_total += 1
        if error:
            summary_error += 1
        row.append(str(success))
        row.append(';'.join(matched))
        row.append(str(error))
        row.append(';'.join(wrong))
    results.append(row)
with open('qwen32b_py_ans_match_report.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['submission_id', 'problem', 'tags', 'match_success', 'matched_tags', 'error_match', 'error_tags'])
    writer.writerows(results)
total = summary_total