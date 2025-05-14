import json
import jsonlines
import re
import csv

# Load JSON of submissions
with open('qwen_32b_python_extract.json', 'r', encoding='utf-8') as f:
    submissions = json.load(f)

# Load JSONL of problems->tags
problem_tags = {}
with jsonlines.open('tags_py.jsonl', 'r') as reader:
    for obj in reader:
        problem = obj.get('problem')
        # Filter out difficulty tags (start with *)
        tags = [t for t in obj.get('tags', []) if not t.startswith('*')]
        problem_tags[problem] = tags

# Special matching rules for multi-word tags
special_rules = {
    'dfs and similar': ['dfs', 'similar'],
    'divide and conquer': ['divide'],
    'expression parsing': ['parsing'],
    'string suffix structures': ['suffix']
}

# Collect all unique tags for error matching
all_tags = set(tag for tags in problem_tags.values() for tag in tags)

# Precompile regex patterns for each tag
tag_regex = {}
for tag in all_tags:
    terms = special_rules.get(tag, [tag])
    tag_regex[tag] = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms]

results = []
summary_total = 0
summary_success = 0
summary_error_total = 0
summary_error = 0

# Process each submission
for s in submissions:
    sid = s.get('submission_id')
    prob = s.get('fullname')
    reasoning = s.get('generate_reasoning', '')

    # Base row includes submission_id, problem, tags
    tags = problem_tags.get(prob)
    tag_str = ''
    if tags is not None:
        tag_str = ';'.join(tags)
    row = [sid, prob, tag_str]

    if tags is None:
        # No tag info for this problem; leave match columns blank
        row.extend(['', '', '', ''])
    else:
        summary_total += 1
        # Match correct tags
        matched = []
        for tag in tags:
            for rx in tag_regex[tag]:
                if rx.search(reasoning):
                    matched.append(tag)
                    break
        success = bool(matched) or len(tags) == 0
        if success:
            summary_success += 1

        # Match error tags (appear in reasoning but not in this problem's tags)
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

# Write to CSV
with open('qwen32b_py_ans_match_report.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['submission_id', 'problem', 'tags', 'match_success', 'matched_tags', 'error_match', 'error_tags'])
    writer.writerows(results)

# Summary output
total = summary_total
print(f"匹配成功占比: {summary_success}/{total} = {summary_success/total:.2%}")
print(f"错误匹配占比: {summary_error}/{total} = {summary_error/total:.2%}")
