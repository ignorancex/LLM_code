import argparse
import json

from helper import sys

def remove_braces(input_string):
    # 去掉左右花括号
    result_string = input_string.replace('{', '').replace('}', '')
    return result_string

def convert_empty_string_to_none(s):
    if s == "":
        return None
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert json to NLFT-style, adding instruction')
    parser.add_argument('--input_json', type=str, default='./gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct.json', help='path to input json')
    parser.add_argument('--output_json', type=str, required=True, help='path to output json') # default: reformatted_gsm8k_train_cot_gpt4_judgment_llama3_8b_instruct.json

    args = parser.parse_args()
    file_path = args.input_json
    save_file_path = args.output_json

    with open(file_path) as f:
        data = json.load(f)

    new_data = []

    for _, item in enumerate(data):
        # question, answer, ans, judgment
        new_item = {
            "id": item['id'], # 打乱过了
            "instruction": sys,
            "input": item["question"],
            "output": item["ans"],
            "judgment": item["judgment"],
            "answer": item["answer"]
        }
        new_data.append(new_item)

    with open(save_file_path, 'w') as f:
        json.dump(new_data, f, indent=2)
