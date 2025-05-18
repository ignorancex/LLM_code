import json
from collections import defaultdict
with open('LLM_code/code/github_links/categories.json', 'r') as f:
    data = json.load(f)
summary = defaultdict(lambda : {'cs.CV': 0, 'cs.CL': 0, 'cs.*': 0, 'non-cs': 0})
for (quarter, entries) in data.items():
    for entry in entries:
        cat = entry['categories']
        if cat == 'cs.CV':
            summary[quarter]['cs.CV'] += 1
        if cat == 'cs.CL':
            summary[quarter]['cs.CL'] += 1
        if cat.startswith('cs.'):
            summary[quarter]['cs.*'] += 1
        else:
            summary[quarter]['non-cs'] += 1
for quarter in sorted(summary.keys()):
    counts = summary[quarter]