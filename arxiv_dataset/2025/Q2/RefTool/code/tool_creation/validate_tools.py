import json
import argparse
import os
import sys
import openai
from tqdm import tqdm
from io import StringIO
import signal
import functools

from utils import *


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

@timeout(seconds=10, default=None)    
def exec_code(code):
    namespace = {}
    try:
        exec(code, namespace)
        answer = namespace['solution']()
        return answer, ''
    except Exception as e:
        return '', e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--stage', type=str, required=True, help="stage: unfiltered or refined")
    args = parser.parse_args()

    verify_model = 'gpt3.5'  # model to check if the execution output meets the example answer

    output_folder = f'outputs/tmp_response/'
    tools = json.load(open(f'outputs/{args.model_name}_{args.domain}_tools_{args.stage}.json', 'r'))
    
    feedback = []
    prompts = []
    for i in tools:
        if 'feedback' in i and i['feedback'] != 'Refined.':
            feedback.append('Unchange')
            continue
        try:
            code = i['function'] + '\n' + i['example']['solution']
        except:
            feedback.append('Wrong format!')
            continue
        if 'plt.show()' in code or 'plot.show()' in code:
            feedback.append('Do not make plots!')
        elif 'quit()' in code:
            feedback.append('Do not generate quit()!')
        elif not 'answer' in i['example']:
            feedback.append('No example answer!')
        else:
            exec_result, exec_error = exec_code(code)
            if str(exec_result) != '':
                i['exec_result'] = str(exec_result)
                feedback.append("")
                prompts.append(f"Is `{exec_result}` roughly equivalent with `{i['example']['answer']}` (5% deviation is allowed)? Please just answer one word, Yes or No.")
            else:
                feedback.append(f'No execution output. Execution error: {exec_error}')

    all_responses = run_inference(prompts, output_folder, args)
    compare_idx = 0
        
    tools_succeed = []
    for idx in range(len(tools)):
        if feedback[idx] == 'Unchange':
            if tools[idx]['feedback'] == 'Succeed!':
                tools_succeed.append(tools[idx])
            continue
        if feedback[idx] != "":
            tools[idx]['feedback'] = feedback[idx]
        else:
            if "yes" in all_responses[compare_idx].lower():
                tools[idx]['feedback'] = 'Succeed!'
                tools_succeed.append(tools[idx])
            else:
                tools[idx]['feedback'] = f"Execution output `{tools[idx]['exec_result']}` does not match the answer `{tools[idx]['example']['answer']}`."
            compare_idx += 1
    assert compare_idx == len(all_responses)

    print(len(tools), len(tools_succeed))

    json.dump(tools, open(f'outputs/{args.model_name}_{args.domain}_tools_{args.stage}_feedback.json', 'w'), indent=4)
    if args.stage == 'refined':
        for i in tools_succeed:
            i.pop('exec_result', None)
            i.pop('feedback', None)
        json.dump(tools_succeed, open(f'../../generated_tools/{args.model_name}_{args.domain}_tools.json', 'w'), indent=4)

    remove_tmp_files(output_folder)