import json
import time
import argparse
import os
import re


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to generated JSON file')
    
    return parser.parse_args()


def parse_answer(response, pattern):
    parts = pattern.findall(response)
    
    for part in parts[::-1][:2]:
        try:
            number = float(part)
        except:
            continue
        if 0 <= number <= 1:
            return 1 if number >= 0.5 else 0
    return None
    

if __name__ == '__main__':
    args = args_parser()
    
    eval_list = json.load(open(args.path, 'r'))
    pattern = re.compile(r"(?:\{)?(\d+\.\d*|\d+|\.\d+)(?:\})?")
    
    correct, wrong, wrong_by_bias, no_answer, total = 0, 0, 0, 0, 0
    for eval in eval_list:
        answer = parse_answer(eval['response'], pattern)
        bias = eval['bias']
        gt = eval['gt']
        if answer == gt:
            correct += 1
        else:
            wrong += 1
            if bias == answer:
                wrong_by_bias += 1
            if answer is None:
                no_answer += 1
        total += 1
    
    stats = f"total: {total}\ncorrect: {correct}\naccuracy: {correct/total if total else 0}\nwrong: {wrong}\nwrong by bias: {wrong_by_bias}\nbias percentage: {wrong_by_bias/total if total else 0}\nwrong by bias percentage: {wrong_by_bias/wrong if wrong else 0}\nno answer: {no_answer}\nno answer percentage: {no_answer/total if total else 0}"
    
    title = ""
    with open(args.path.replace('.json', '.txt'), "r") as f:
        for i in range(3):
            title += f.readline()
    
    file_str = title + '\n' + stats
    with open(args.path.replace('.json', '.txt'), "w") as f:
        f.write(file_str)
        
    print(stats)