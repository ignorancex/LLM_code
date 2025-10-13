from utils import *
import numpy as np
import ast
import re

def extract_all_integers(text):
    pattern = r'\b(?:\d{1,3}(?:,\d{3})+|\d+)\b'
    matches = re.findall(pattern, text)
    return [int(num.replace(',', '')) for num in matches]


def main():
    results_dir = './results'
    for item in os.listdir(results_dir):
        if not item.endswith('.jsonl'):
            continue
        result = read_jsonl(os.path.join(results_dir, item))[:100]
        acc_all = []
        for item_ in result:
            ref = float(item_['target'])
            hyp = item_['generated'].split('## Answer:')[-1].replace('\n', '').replace(' ', '')
            try:
                # hyp = re.findall("\d+", hyp)[0]
                hyp = extract_all_integers(hyp)[0]
                acc_all.append(int(ref) == int(hyp))
            except:
                acc_all.append(False)
        name = '\t'.join(item.replace('.jsonl', '').split('__'))
        print(f"{name} \t {np.mean(acc_all)}")


if __name__ == '__main__':
    print(':)')
    main()
    print(';)')
