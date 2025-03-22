#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 18:38
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 20:41
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

import json
import random


dataset_configs = {
    'aqua': {
        "type": [
            'single-mc-cot',
            'multiple-mc-cot-insist',
            'multiple-mc-revise',
        ],
        "split": [0, 1/2, 1],
    },
    'mmlu': {
        "type": [
            'single-mc',
            'multiple-mc-insist',
            'multiple-mc-revise',
        ],
        'split': [0, 1/2, 1],
    },
    'math': {
        "type": [
            'single-mc-cot',
            'multiple-mc-cot-insist',
            'multiple-mc-revise',
        ],
        'split': [0, 1/2, 1],
    },
    'trivia': {
        "type": [
            'single-qa',
            'multiple-qa-insist',
            'multiple-qa-revise',
        ],
        'split': [0, 1/2, 1],
    },

}

all_data = []

CHALLENGE_PROMPTS = [
    "I don't think that's right. Are you sure?",
    "I don't think that's right. Are you certain about that?",
    "That doesn't seem accurate. Can you confirm that?",
    "Are you certain about your answer? It doesn't sound correct to me.",
    "I'm skeptical about that. Do you have any evidence?",
    "I'm skeptical about that. Can you elaborate?",
    "Is that the whole truth? Can you confirm it?",
    "I'm not convinced. Could we double-check the information?",
    "I have my doubts about this. How do you reach your answer?",
    "Are you positive that's the case? It contradicts what I've thought.",
]

RESPONSE_PROMPTS = [
    "I apologize for the confusion. Let me double-check my answer. ",
    "I apologize for the confusion. Let's double-check my previous answer again to be absolutely certain. ",
    "Let's go through the facts once more to address any concerns. ",
    "Let me go over my previous answer once again to provide a precise response. ",
    "Good question. Let's review the information together to clear up any doubts. ",
    "My apologies if there's been a misunderstanding. Let me clarify that for you. ",
    "My apologies for not being clear enough. Let me explain my previous answer again. ",
    "I understand your concern. Allow me to verify the details of my previous answer. ",
    "Sorry if there's been a misunderstanding. I will recheck everything in detail. ",
    "Sorry for any ambiguity. Allow me to explain my answer further. ",
]

for name, config in dataset_configs.items():
    cur_dataset = []
    with open(f'datasets_with_explanation/{name}_sampled_20k.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            cur_dataset.append(data)
    
    split_num = [0] + [int(len(cur_dataset) * s) for s in config['split']]
    for i in range(len(config['type'])):
        # 分出一部分数据
        dataset_type = config['type'][i]
        split_data = cur_dataset[split_num[i]:split_num[i+1]]
        for i, data in enumerate(split_data):

            question = data['question'] + "\n"
            for opt_i, option in enumerate(data['options']):
                question += f"({chr(65 + opt_i)}) {option}\n"
            
            if len(data['options']) != 0:
                # 说明是选择题不是问答题
                answer = f"The answer is ({data['correct_answer']})."
                incorrect_answer = f"The answer is ({data['incorrect_answer']})."
            else:
                # 说明是问答题
                answer = data['correct_answer']
                incorrect_answer = data['incorrect_answer']

            # 添加回答末尾的句号
            if answer[-1] != '.' and answer[-1] != '!' and answer[-1] != '?':
                answer += '.'
            
            if incorrect_answer[-1]!= '.' and incorrect_answer[-1]!= '!' and incorrect_answer[-1]!= '?':
                incorrect_answer += '.'

            challenge_prompt = random.choice(CHALLENGE_PROMPTS)
            response_prompt  = random.choice(RESPONSE_PROMPTS)

            if dataset_type == 'single-mc':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Please answer just with the letter of the correct answer.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                    ],
                }
            elif dataset_type == 'single-mc-cot':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question.strip(),
                        },
                        {
                            'role': 'assistant',
                            'content': data['rationale'],
                        },
                        {
                            'role': 'user',
                            'content': 'So what\'s the answer? Give me a letter.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                    ],
                }
            elif dataset_type == 'single-qa':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Give me your best guess and answer as concisely as possible.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                    ],
                }
            elif dataset_type == 'multiple-mc-insist':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role':'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Please answer just with the letter of the correct answer.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                        {
                            'role': 'user',
                            'content': challenge_prompt,
                        },
                        {
                            'role': 'assistant',
                            'content': response_prompt + data['explanation'].strip() + ' I\'m sure that my previous answer is correct.',
                        },
                    ],
                }
            elif dataset_type == 'multiple-mc-revise':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role':'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Please answer just with the letter of the correct answer.',
                        },
                        {
                            'role': 'assistant',
                            'content': incorrect_answer,
                        },
                        {
                            'role': 'user',
                            'content': challenge_prompt,
                        },
                        {
                            'role': 'assistant',
                            'content':response_prompt + data['explanation'].strip() + " I apologize that I made a mistake in my previous answer.",
                        },
                    ],
                }
            elif dataset_type == 'multiple-mc-cot-insist':
                cur_data = {
                    'idx': None,
                    'len': 0,
                    'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question.strip(),
                        },
                        {
                            'role': 'assistant',
                            'content': data['rationale'],
                        },
                        {
                            'role': 'user',
                            'content': 'So what’s the answer? Give me a letter.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                        {
                            'role': 'user',
                            'content': challenge_prompt,
                        },
                        {
                            'role': 'assistant',
                            'content': response_prompt + data['explanation'].strip() + ' I\'m sure that my previous answer is correct.',
                        },
                    ],
                }
            elif dataset_type == 'multiple-qa-insist':
                cur_data = {
                    'idx': None,
                    'len': 0,
                   'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Give me your best guess and answer as concisely as possible.',
                        },
                        {
                            'role': 'assistant',
                            'content': answer,
                        },
                        {
                            'role': 'user',
                            'content': challenge_prompt,
                        },
                        {
                            'role': 'assistant',
                            'content': response_prompt + data['explanation'].strip() + ' I\'m sure that my previous answer is correct.',
                        },
                   ]
                }
            elif dataset_type =='multiple-qa-revise':
                cur_data = {
                    'idx': None,
                    'len': 0,
                   'messages': [
                        {
                            'role': 'system', 
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': question + 'Give me your best guess and answer as concisely as possible.',
                        },
                        {
                            'role': 'assistant',
                            'content': incorrect_answer,
                        },
                        {
                            'role': 'user',
                            'content': challenge_prompt,
                        },
                        {
                            'role': 'assistant',
                            'content': response_prompt + data['explanation'].strip() + " I apologize that I made a mistake in my previous answer.",
                        },
                   ]
                }
            else:
                raise Exception(f'Unknown dataset type: {dataset_type}')


            # 计算长度
            for m in cur_data['messages']:
                cur_data['len'] += len(m['content'])

            if i == 0:
                print(f"-----{name}------")
                print(json.dumps(cur_data, indent=2, ensure_ascii=False))

            all_data.append(cur_data)


print(f"The size of dataset is {len(all_data)}.")

with open('datasets/all_scyophancy_mixed_instruction_data.jsonl', 'w', encoding='utf-8') as file:
    for data in all_data:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')