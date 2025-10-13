import json
import argparse
from itertools import chain
from copy import deepcopy
import os
from tqdm import tqdm
import os
from openai import AzureOpenAI

import os
import time
import re
import json
from openai import OpenAIError, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureChatOpenAI
from langchain.evaluation import load_evaluator


class LangChainEvaluator:
    def __init__(self):
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4.1",  # or your deployment
            api_version="2025-01-01-preview",  # or your api version
            temperature=0)

        self.evaluator = load_evaluator("cot_qa",llm=llm)

    def evaluate_item(self, item, max_retries=10):
        if item['eval']['gold'] is None:
            return None
        for attempt in range(max_retries):
            try:
                return self.evaluator.evaluate_strings(
                    prediction=item['eval']['pred'],
                    input=item['task'],
                    reference=item['eval']['gold'],
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    print(e)
                    return None


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data

def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

from copy import deepcopy

def build_messages(step, key):
    """通用构建消息的函数，key为'action'或'plan'"""
    d = deepcopy(step[key])
    msg = d['llm_input']
    msg.append({"role": "assistant", "content": d['llm_output']})
    return msg

def build_end_messages(step):
    """构建end消息"""
    msg = deepcopy(step['end']['llm_input'])
    msg.append({"role": "assistant", "content": step['end']['llm_output']})
    return msg

def is_valid_msg(msg):
    """判断消息内容是否全为字符串"""
    return all(isinstance(p["content"], str) for p in msg)

def process_sub_steps(sub_steps, key):
    """处理子步骤，key为'action'或'plan'"""
    messages = []
    for sub_step in sub_steps:
        d = deepcopy(sub_step[key])
        msg = d['llm_input']
        if not is_valid_msg(msg):
            continue
        msg.append({"role": "assistant", "content": d['llm_output']})
        messages.append(msg)
    return messages

def process_sub_end_messages(sub_steps):
    """处理子步骤的end消息"""
    messages = []
    for sub_step in sub_steps:
        if 'end' in sub_step:
            msg = deepcopy(sub_step['end']['llm_input'])
            msg.append({"role": "assistant", "content": sub_step['end']['llm_output']})
            messages.append(msg)
    return messages

def get_text_sft_data(item):
    action_messages = []
    planning_messages = []
    end_messages = []
    sub_action_messages = []
    sub_planning_messages = []
    sub_end_messages = []

    for step in item['session']['steps']:
        # action
        action_messages.append(build_messages(step, 'action'))
        # plan
        planning_messages.append(build_messages(step, 'plan'))
        # end
        if 'end' in step:
            end_messages.append(build_end_messages(step))
        # subagent
        action_dict = step['action']
        ob = action_dict.get("observation", [])
        if isinstance(ob, dict) and "session" in ob and 'steps' in ob['session']:
            sub_steps = ob['session']['steps']
            sub_action_messages.append(process_sub_steps(sub_steps, 'action'))
            sub_planning_messages.append(process_sub_steps(sub_steps, 'plan'))
            sub_end_messages.append(process_sub_end_messages(sub_steps))

    return (action_messages, planning_messages, end_messages, 
            sub_action_messages, sub_planning_messages, sub_end_messages)

import re
# filter out final action message and end message if they are None.
def rule_filter_final_action_message(final_message):
    if not 'stop' in final_message:
        return True
    patterns = [r'stop.*not found', r'stop.*none', r'stop.*\'\'', r'stop.*""']
    return any(bool(re.search(p, final_message, re.IGNORECASE | re.DOTALL)) for p in patterns)

def rule_filter_end_message(end_message):
    patterns = [r'output.*not found', r'output.*none', r'output.*\'\'', r'output.*""']
    return any(bool(re.search(p, end_message, re.IGNORECASE | re.DOTALL)) for p in patterns)


system_prompt = """
You will be presented with a chain-of-thought indicating the reason of calling ask_llm function.

If the reason is because the previous failures, or the previuos web/file agent were not able to retrieve useful information, return 1, otherwise return 0.

For example, 

Thought: Previous searches for the number of new recipes in the "Innovative French Cuisine" section of the 2018 "Culinary Arts Review" have failed, even after progressively broadening the queries. Since no direct data is available, I will now use ask_llm to estimate or infer the likely number of new recipes in that section, as this is the only remaining viable approach.
All attempts to locate the official IEA Germany 2021 review or any reputable summary via web search have failed, even with the broadest queries. As suggested in the progress state, the next best step is to use ask_llm to estimate or summarize the answer based on general knowledge, clearly noting the lack of direct source and focusing on the integer-rounded percentage as requested.
Previous web searches failed to return any results, indicating that the web search tool is currently unable to retrieve the required information. Therefore, I will use ask_llm to directly obtain the highest number of islands mentioned on the Wikipedia page for the Philippines.

Output: 1

Thought: No previous steps have been taken yet. The task is to extract the sentence by reading all letters in the provided 6x6 block from left to right, row by row. I will use ask_llm to process the block and output the sentence.

Output: 0
"""

def ask_llm(input_messages):
    while True:
        try:
            response = client.chat.completions.create(
                    model="gpt-4.1", # model = "deployment_name".
                    messages=input_messages
                )
            return response.to_dict()['choices'][0]["message"]['content']
        except Exception as e:
            print(e)
            try:
                # if e.body['innererror']['content_filter_result']['jailbreak']['filtered']:
                if any(e.body['innererror']['content_filter_result'][r]['filtered'] for r in e.body['innererror']['content_filter_result']):
                    return ''
            except:
                pass
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                return ''
            else:
                time.sleep(10)


def determine_force_ask_llm(thought):
    input_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Thought: " + thought},
                    ]
    return ask_llm(input_messages)
    

# re-generate the response for queries in all_ask_llm_arguments
def sample_output(query):
    input_messages = [
                        {"role": "system", "content": "You are a helpful assistant. Answer the user's query with your internal knowledge. Ensure to follow the required output format if specified."}, 
                        {"role": "user", "content": query}
                    ]
    
    return ask_llm(input_messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, nargs='+', required=True, help='Input JSONL file path(s)')
    parser.add_argument('--output_file', type=str, required=True, help='Output SFT JSONL file path')

    parser.add_argument('--remove_ask_llm', action='store_true', help='Whether to remove the stopped trajectories where ask_llm is performed to acquire the final answer due to previous failures.')
    parser.add_argument('--save_ask_llm', action='store_true', help='Whether to save the ask_llm results.')
    parser.add_argument('--rejection_sampling_type', choices=['none', 'llm_judge', 'em'], default='none')

    parser.add_argument('--include_screenshot', action='store_true', help='Include screenshots in the data')
    parser.add_argument('--input_screenshot_path', type=str, required=False, help='Path to screenshot saved by the web agent')
    parser.add_argument('--screenshot_destination_path', type=str, required=False, help='Destination path for screenshots')
    

    args = parser.parse_args()

    if args.include_screenshot:
        screenshot_destination_path = args.screenshot_destination_path
        os.makedirs(screenshot_destination_path, exist_ok=True)

    trajectories = []
    for input_file in args.input_file:
        trajectories += read_jsonl(input_file)

    if args.rejection_sampling_type == 'em':
        # exact match
        accuracy = sum([item['eval']['corr'] == 1 for item in trajectories]) / len(trajectories)
        print("exact match accuracy:", format(accuracy, '.4g'))
        trajectories = [item for item in trajectories if item['eval']['corr'] == 1]
    elif args.rejection_sampling_type == 'llm_judge':
        evaluator = LangChainEvaluator()
        scores = []
        with ThreadPoolExecutor(10) as executor:
            for res in tqdm(executor.map(evaluator.evaluate_item, trajectories), total=len(trajectories)):
                scores.append(res)
        trajectories = [item for item, score in zip(trajectories, scores) if score['score'] == 1]
        print("number of none in evaluation:", sum([item['score'] is None for item in scores]))
        accuracy = sum([item['score'] if item['score'] is not None else 0 for item in scores]) / len(scores)
        print("llm evaluator accuracy", format(accuracy, '.4g'))

    client = AzureOpenAI(
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'], 
        api_key=os.environ['AZURE_OPENAI_API_KEY'],  
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    )

    if args.remove_ask_llm:

        def worker_detect_ask_llm(t):
            for step in t['session']['steps']:
                if bool(re.search(r'ask_llm\((.*?)\)', step['action']['code'])):
                    if '1' in determine_force_ask_llm(step['action']['thought']):
                        return True
            return False

        filter_label = []
        with ThreadPoolExecutor(5) as executor:
            for res in tqdm(executor.map(worker_detect_ask_llm, trajectories), total=len(trajectories)):
                filter_label.append(res)
        print(f"Filtering trajectories... number of original trajectories: {len(trajectories)}, number after filter {len(trajectories)-sum(filter_label)}",)
        trajectories = [t for t, label in zip(trajectories, filter_label) if not label]

    #print(len(trajectories))
    if not args.include_screenshot:
        sft_data = []
        all_ask_llm_arguments = []
        all_ask_llm_observations = []
    
        success_traj_cnt = 0
        for i in tqdm(range(len(trajectories))):

            action_messages, \
            planning_messages,\
            end_messages, \
            sub_action_messages, \
            sub_planning_messages,\
            sub_end_messages = get_text_sft_data(trajectories[i])

            stopped_messages = []
            # if this task has not finished
            final_message = action_messages[-1][-1]['content']
            if len(end_messages) == 0:
                continue
            end_message = end_messages[-1][-1]['content']
            if rule_filter_final_action_message(final_message) or rule_filter_end_message(end_message):
                # failed
                continue
            else:
                stopped_messages = action_messages + planning_messages + end_messages
                success_traj_cnt += 1


            for subagent_i in range(len(sub_action_messages)):
                final_content = sub_action_messages[subagent_i][-1][-1]['content']
                # check if this subagent has finished
                if len(sub_end_messages[subagent_i]) > 0:
                    end_message = sub_end_messages[subagent_i][-1][-1]['content']
                else:
                    continue
                if rule_filter_final_action_message(final_message) or rule_filter_end_message(end_message): 
                    stopped_messages = stopped_messages + sub_action_messages[subagent_i] + sub_planning_messages[subagent_i] + sub_end_messages[subagent_i]

            sft_data.extend(stopped_messages)

            if args.remove_ask_llm or args.save_ask_llm:
                # acquire the ask_llm keys and values as they are not saved.
                
                for step in trajectories[i]['session']['steps']:
                    if bool(re.search(r'ask_llm\((.*?)\)', step['action']['code'])):

                        argument = re.search(r'ask_llm\((.*?)\)', step['action']['code']).group(1)
                        observation = step['action']['observation']
                        try:
                            if isinstance(eval(argument), str) :
                                input_query = eval(argument)
                                all_ask_llm_arguments.append(input_query)
                                all_ask_llm_observations.append(observation)
                                break
                        except:
                            pass

                        code = step['action']['code'].split('ask_llm')[0]

                        code = "\n".join(code.split('\n')[:-1])
                        try:
                            # input_query = exec(code+f"\nreturn {argument}")
                            local_vars = {}
                            exec(code, {}, local_vars)
                            input_query =local_vars[argument] 
                            all_ask_llm_arguments.append(input_query)
                            all_ask_llm_observations.append(observation)

                        except Exception as e:
                            print("error", step['action']['code'])

        # ask_llm_outputs = []
        # with ThreadPoolExecutor(5) as executor:
        #     for res in tqdm(executor.map(sample_output, all_ask_llm_arguments), total=len(all_ask_llm_arguments)):
        #         ask_llm_outputs.append(res)
        ask_llm_sft = [
            [{"role": "system", "content": "You are a helpful assistant. Answer the user's query with your internal knowledge. Ensure to follow the required output format if specified."}, 
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
            ]
            for query, answer in zip(all_ask_llm_arguments, all_ask_llm_observations)
        ]

    else:
        raise NotImplementedError
    print("number of sft_data", len(sft_data), "number of ask_llm data", len(ask_llm_sft))
    save_jsonl(sft_data, args.output_file)
    save_jsonl(ask_llm_sft, args.output_file.replace(".jsonl", ".ask_llm.jsonl"))
