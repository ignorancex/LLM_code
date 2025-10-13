import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import signal
import sys
import numpy as np
import re
import math
import functools

from evaluation_utils import *


def timeout(seconds=5, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator


class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)

        
@timeout(seconds=10, default=None)        
def evaluate_one_pot(entry):
    code = entry['code']
    code = code.split('plt.show()')[0].split('plot.show()')[0].split('quit()')[0]
    namespace = {}
    try:
        exec(code, namespace)
        answer = namespace['solution']()
        return answer
    except Exception as e: 
        print(e)
        

def process_single_pot_output(entry, data_dir='../../evaluation_data/data/'):
    output = entry['output']
    output = output.replace("pd.read_csv('", f"pd.read_csv('{data_dir}")
    output = output.replace("pd.read_csv(\"", f"pd.read_csv(\"{data_dir}")
    if 'argparse' in output:
        return ''
    try:
        output = re.search(r'```python(.*?)```', output, re.DOTALL).group(1)
        return 'import sympy as sp\n\nimport numpy as np\n\nimport math\n\n' + output
    except:
        pass
    
    tmp = re.findall(r"```python", output, re.DOTALL)
    if len(tmp) > 0:
        output = output.split('```python')[1]
    tmp = re.findall(r"```", output, re.DOTALL)
    if len(tmp) > 0:
        output = output.split('```')[0]
        
    if 'def solution():' not in output and len(output) > 4 and output[:4] == '    ':
         processed_output = "def solution():\n" + output
    elif 'def solution():' not in output:
        processed_output = "def solution():\n    " + output
    else:
        processed_output = output.strip()
   
    # replace answer = A with answer = 'A' use re to cover ABCD
    processed_output = re.sub(r"answer = ([A-Z])", r"answer = '\1'", processed_output)
    # replace return A with return 'A'
    processed_output = re.sub(r"return ([A-Z])", r"return '\1'", processed_output)
    
    if 'return' not in processed_output:
        line_of_code = processed_output.split('\n')
        ## search for the last line that defines a variable
        for i in range(len(line_of_code)-1, -1, -1):
            if '=' in line_of_code[i]:
                variable_name = line_of_code[i].split('=')[0].strip()
                processed_output = processed_output + '\n##use last variable as answer\n    return ' + variable_name
                break
            elif 'print' in line_of_code[i]:
                ## check if {} is in the print, if exist, extract answer from it
                if re.search(r"\{(.*)\}", line_of_code[i]):
                    variable_name = re.search(r"\{(.*)\}", line_of_code[i]).group(1)
                else:
                    variable_name = line_of_code[i].strip()[6:-1]
                processed_output = processed_output + '\n##use last print as answer\n    return ' + variable_name
                break
    return processed_output


def evaluate_pot(data, args):
    code_result_path = f'outputs/{args.model_name}_{args.domain}_{args.method}_output.json'

    if os.path.exists(code_result_path) and not args.force_generate:
        data = json.load(open(code_result_path))
    else:
        for entry in tqdm(data, desc='Evaluating PoT results...'):
            if 'function_list' in entry and len(entry['function_list']) > 0:
                functions = '\n\n'.join([s['function'] for s in entry['function_list']]) + '\n\n'
                entry['code'] = functions + process_single_pot_output(entry).strip()
            else:
                entry['code'] = process_single_pot_output(entry).strip()
                
            try:
                entry['pred'] = evaluate_one_pot(entry)
                entry['successful_execution'] = True
            except:
                entry['successful_execution'] = False
                entry['pred'] = None
            try:
                json.dumps(str(entry['pred']))
                entry['pred'] = str(entry['pred'])
            except:
                entry['pred'] = ' '

        with open(code_result_path, 'w') as f:
            json.dump(data, f, indent=4)
    

def calc_acc_qrdata(data, args):
    error_scale = 0.03
    code_result_path = f'outputs/{args.model_name}_{args.domain}_{args.method}_output.json'
    correct = 0
    for idx, i in enumerate(data):
        pred = i['pred']
        ## remove the trailing period if there is one
        if pred.endswith('.'):
            pred = pred[:-1]
        
        i['correct'] = False
        if i['meta_data']['question_type'] == 'numerical':
            if i['answer'][-1] != '%':
                gold_float = float(i['answer'])
            else:
                gold_float = float(i['answer'][:-1]) / 100
            try:
                pred_float = extract_first_number(pred)
                if pred_float[-1] != '%':
                    pred_float = float(pred_float)
                else:
                    pred_float = float(pred_float[:-1]) / 100
                lower_bound = min(gold_float * (1-error_scale), gold_float * (1+error_scale))
                upper_bound = max(gold_float * (1-error_scale), gold_float * (1+error_scale))
                if lower_bound < pred_float and upper_bound > pred_float:
                    correct += 1
                    i['correct'] = True
            except:
                continue
        else:  # if the gold answer is multiple choice
            if i['answer'] == pred[:len(i['answer'])]:
                correct += 1
                i['correct'] = True
    
    print(args.model_name, args.domain, correct / len(data))
    
    with open(code_result_path, 'w') as f:
        json.dump(data, f, indent=4)

    return

            
def calc_acc_theoremqa(data, args):
    code_result_path = f'outputs/{args.model_name}_{args.domain}_{args.method}_output.json'
    correct = 0
    for problem_data in data:
        if problem_data['pred'] == '':
            problem_data['correct'] = False
            continue
        groundtruth = problem_data['Answer']
        if isinstance(groundtruth, str):
            groundtruth_num = None
        else:
            groundtruth_num = groundtruth
            groundtruth = str(groundtruth)
        if compare_answer_with_groundtruth(problem_data['pred'], groundtruth, groundtruth_num):
            correct += 1
            problem_data['correct'] = True
        else:
            problem_data['correct'] = False
        
    print(args.model_name, args.domain, correct / len(data))
    
    with open(code_result_path, 'w') as f:
        json.dump(data, f, indent=4)

    return


def calc_acc_scibench(data, args):
    code_result_path = f'outputs/{args.model_name}_{args.domain}_{args.method}_output.json'
    accumulate_acc = []
    for source in ['chemmc', 'matter', 'quan']:
        correct = 0
        cnt = 0
        for problem_data in data:
            if problem_data['source'] != source:
                continue
                
            cnt += 1 
            unit_prob=problem_data["unit"]
            if remove_not(problem_data["unit"]):
                unit_prob=remove_not(problem_data["unit"])
            model_output = problem_data['pred']
            answer = problem_data['answer_number']
            if unit_prob!=problem_data["unit"]:
                answer=cal_not((answer, problem_data["unit"]))

            try:
                res_equiv = equiv(str(model_output), answer, problem_data["unit"])
            except:
                res_equiv = False
            if res_equiv:
                correct += 1
                problem_data['correct'] = True
            else:
                problem_data['correct'] = False
        accumulate_acc.append(correct/cnt)
        
    print(args.model_name, args.domain, np.mean(accumulate_acc))
    
    with open(code_result_path, 'w') as f:
        json.dump(data, f, indent=4)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--force_generate', action='store_true', help='force generate the output file')
    parser.add_argument('--domain', type=str, default='')
    parser.add_argument('--method', type=str, default='')

    args = parser.parse_args()
    output_file = f'outputs/{args.model_name}_{args.domain}_{args.method}.json'

    if os.path.exists(output_file):
        data = json.load(open(output_file))
        evaluate_pot(data, args)
        code_result_path = f'outputs/{args.model_name}_{args.domain}_{args.method}_output.json'
        data = json.load(open(code_result_path))
        
        if args.domain == 'causality':
            calc_acc_qrdata(data, args)
        elif args.domain == 'physics':
            calc_acc_theoremqa(data, args)
        elif args.domain == 'chemistry':
            calc_acc_scibench(data, args)
        
    else:
        raise FileNotFoundError(f'File {output_file} not found')