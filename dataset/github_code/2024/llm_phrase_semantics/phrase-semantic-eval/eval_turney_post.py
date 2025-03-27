import json
import os, sys
import re
import string
import argparse
from tqdm import tqdm
import pandas as pd

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)



def main_turney(args):
    with open(args.result_path, 'r') as result_file:
        pred_details = json.load(result_file)
    punc_regex = re.compile('[%s]' % re.escape(string.punctuation))

    wrong_preds = []
    num_invalid, num_invalid_after_process, num_invalid_is_query, num_correct_before_process, num_correct_after_process = 0, 0, 0, 0, 0
    for pred_idx, pred_detail in enumerate(tqdm(pred_details)):
        pred_detail['id'] = pred_idx
        target = pred_detail['target'].lower().strip()
        candidates = [l.lower().strip() for l in pred_detail['candidates']]
        raw_pred = pred_detail['raw_pred']
        lines = [l for l in raw_pred[0].split('\n') if l.strip()]
        raw_last_line = lines[-1].lower().strip()
        last_line = punc_regex.sub('', lines[-1]).lower().strip()
        pred = last_line.lower().strip()
        is_valid = any([pred == l.lower().strip() for l in pred_detail['candidates']])
        if not is_valid:
            num_invalid += 1
        is_correct = (pred == target)
        if is_valid:
            if is_correct:
                num_correct_before_process += 1
                num_correct_after_process += 1
            else:
                wrong_preds.append(pred_detail)

            continue
        pred = [l for l in candidates if l in last_line]
        if len(pred) != 1:
            print(f"Query={pred_detail['query']}")
            print(f"Cands={pred_detail['candidates']}")
            print(raw_pred[0])
            # print(raw_last_line)
            # print(pred)
            matches = [re.search(rf"\b{l}\b", raw_last_line) for l in candidates]
            matches = [(c, m) for c, m in zip(candidates, matches) if m]
            if len(matches):
                # sort by their appearance, and take the last one as the prediction
                matches = sorted(matches, key=lambda t: t[1].regs[-1][0], reverse=True)
                pred = matches[0][0]
                print("corrected after post-process")
                print(pred)
            else:
                pred = "N/A"
                if raw_last_line == pred_detail['query']:
                    num_invalid_is_query += 1
                    print("is_query")
                else:
                    print("N/A")
                    print(pred_detail['raw_pred'])
                num_invalid_after_process += 1
            print('*' * 20)
        else:
            pred = pred[0]
        is_correct = (pred == target)
        pred_detail['pred'] = pred
        if is_correct:
            num_correct_after_process += 1
        else:
            wrong_preds.append(pred_detail)

    print(f'#invalid={num_invalid}')
    print(f'#invalid_after_process={num_invalid_after_process}, #num_invalid_is_query={num_invalid_is_query}')
    print(f'Before process: #correct = {num_correct_before_process}, acc={num_correct_before_process / len(pred_details)}')
    print(f'After process:  #correct = {num_correct_after_process}, acc={num_correct_after_process / len(pred_details)}')

    header = ['id', 'query', 'candidates', 'target', 'pred', 'raw_pred', 'prompt']
    preds = []
    for pred in wrong_preds:
        pred = {k: pred[k] for k in header}
        pred['raw_pred'] = pred['raw_pred'][0]
        preds.append(pred)

    df = pd.json_normalize(preds)
    csv_df = df.to_csv('~/Desktop/turney-badcase.csv')
    # print(csv_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str,  required=True, default='')
    args = parser.parse_args()
    main_turney(args)
