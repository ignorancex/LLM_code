import json
from datasets import load_from_disk, load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import random
import re
from tqdm import tqdm
import argparse
import os

import sys

sys.path.append('.')
from executor import *

PROMPT = 'Answer the following question step by step. During the thinking process, you can create Python code snippets to help with your reasoning. The Python code should be presented as a format of python code block within the markers "```python" and "```". After running your code, share the results by placing them between "```output" and "```". All of these python codes and corresponding outputs should be embed within the "<think>" and "</think>" markers.\n'


def process_input(question, tokenizer):
    chat_prob = tokenizer.apply_chat_template(
        [
            # {
            #     "role": "system",
            #     "content": "You are a helpful and harmless assistant. You should think step-by-step.",
            # },
            {"role": "user", "content": PROMPT + question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prob


def load_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(_) for _ in f.readlines()]

    return data


def add_code_prefix(sub_response: str):
    code_prefix = [
        "Hmm, I think it's time to write some Python code to help me sort out my thoughts.",
        "Wait a minute, maybe I could use some code to double-check my reasoning.",
        "Alright, let me just whip up a quick Python script to clarify my thinking.",
        "Hmm, perhaps coding could assist in verifying my thought process.",
        "Hold on, I might be able to use some code to validate my conclusions.",
        "Okay, I should probably create a few Python snippets to help organize my ideas.",
        "Wait, it just hit me—using code might help confirm my thought patterns.",
        "Hmm, I wonder if writing some Python could shed light on my reasoning.",
        "Okay, it's time to leverage Python code to make my thoughts clearer:",
        "Hmm, perhaps I could utilize code to check the validity of my reasoning:",
        "Let me see if simulating this problem with Python aligns with my expectations:",
        "I believe that coding will enhance my understanding of this concept:",
        "Maybe by crafting some straightforward Python scripts, I can analyze this issue more efficiently:",
        "Wait a second, I might be able to use code to validate my thought process:",
    ]

    selected_prefix = random.sample(code_prefix, 1)[0]

    added_sub_response = sub_response + "\n\n" + selected_prefix + "\n```python"

    return added_sub_response


def split_string_by_keywords(input_string, keywords):
    pattern = "|".join(re.escape(keyword) for keyword in keywords)

    parts = re.split(f"({pattern})", input_string)

    result = []
    temp_piece = ""
    i = 0
    while i < len(parts) and parts[i] not in keywords:
        temp_piece += parts[i]
        i += 1
    if temp_piece:
        result.append(temp_piece)

    while i < len(parts):
        # print(i)
        part = parts[i]
        # print(part)
        # break
        if part in keywords:
            temp_piece = part
            i += 1
            while i < len(parts) and parts[i] not in keywords:
                temp_piece += parts[i]
                i += 1
            result.append(temp_piece)

    result = [_.strip() for _ in result]
    return result


def random_combine_pieces(response_split: list):
    piece_num = len(response_split)
    if piece_num < 5:
        return None
    random_integer = random.randint(int(piece_num / 3), piece_num)
    sub_response = "\n\n".join(response_split[:random_integer])
    return sub_response


def generation_loop(sub_thinking_data, model, bz, code_sampling_params, rollout_sampling_params, output_path):
    executor = PythonExecutor()
    max_iter = 5
    iter_num = 0

    while len(sub_thinking_data) > 0:
        if iter_num > max_iter:
            print(f'Force quit! Rest data: {len(sub_thinking_data)}')
            break
        processed_prompts = []
        processed_data = []

        complete_data = []
        uncompleted_data = []

        for each_data in sub_thinking_data:
            if each_data['cur_code_num'] >= each_data['max_code_num']:
                continue
            response = each_data["pred"]
            thought = response.split("</think>")[0]
            original_prompt = each_data["prompt"]

            pieces = split_string_by_keywords(thought, key_words)

            sub_response = random_combine_pieces(pieces)
            if not sub_response:
                continue

            added_code_sub_response = add_code_prefix(sub_response)

            processed_prompts.append(original_prompt + added_code_sub_response)
            processed_data.append(each_data)

        for i in range(0, len(processed_prompts), bz):
            bz_prompts = processed_prompts[i: i + bz]
            bz_data = processed_data[i: i + bz]
            responses = model.generate(bz_prompts, code_sampling_params)

            preds = []
            tmp_data = []
            for response, prompt_data, question_data in zip(
                    responses, bz_prompts, bz_data
            ):
                question = question_data["question"]
                for output in response.outputs:
                    pred = output.text
                    stop_reason = output.stop_reason
                    if stop_reason != '```':
                        pred = ''
                    preds.append(pred)
                    res = {
                        "question": question,
                        "combined_text": question_data["combined_text"],
                        "prompt": prompt_data,
                        "pred": pred,
                        "max_code_num": question_data["max_code_num"],
                        "cur_code_num": question_data["cur_code_num"],
                    }
                    tmp_data.append(res)
            batch_results, no_code_idx = excute_code(preds, executor=executor)
            roll_candidates_data = []
            roll_candidates_prompt = []
            for i, (excu_result, each_data) in enumerate(zip(batch_results, tmp_data)):
                output, report = excu_result
                if preds[i] == '':
                    continue
                if report == 'Done':
                    excu_content = output
                else:
                    excu_content = report
                prompt_with_code = each_data['prompt'] + preds[i] + '\n```\n\n```Output\n' + excu_content + '\n```\n\n'
                roll_candidates_prompt.append(prompt_with_code)
                roll_candidates_data.append(each_data)

            comp_responses = model.generate(roll_candidates_prompt, rollout_sampling_params)

            for response, prompt_data, question_data in zip(
                    comp_responses, roll_candidates_prompt, roll_candidates_data
            ):
                question = question_data["question"]
                for output in response.outputs:
                    pred = output.text
                    preds.append(pred)
                    res = {
                        "question": question,
                        "combined_text": question_data["combined_text"],
                        "prompt": prompt_data,
                        "pred": pred,
                        "max_code_num": question_data["max_code_num"],
                        "cur_code_num": question_data["cur_code_num"] + 1,
                    }
                    if res['cur_code_num'] < res['max_code_num']:
                        uncompleted_data.append(res)
                    else:
                        complete_data.append(res)

        sub_thinking_data = uncompleted_data
        if len(complete_data) > 0:
            with open(output_path, 'a+') as f:
                for _ in complete_data:
                    f.write(json.dumps(_) + '\n')
        iter_num += 1


def extract_code(preds):
    codes = []
    for pred in preds:
        if "```" not in pred:
            code = ''
        else:
            code = pred.split("```")[0]
        codes.append(code)
    return codes


def excute_code(preds, executor: PythonExecutor):
    codes = preds
    no_code_idx = []
    for i, code in enumerate(codes):
        if code == '':
            no_code_idx.append(i)
    batch_results = executor.batch_apply(codes)
    return batch_results, no_code_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthesizing Data for thinking with coding.!')
    parser.add_argument('--model_path', default='/capacity/userdata/models/DeepSeek-R1-Distill-Qwen-32', type=str,
                        help='Model path')
    parser.add_argument('--data_path',
                        default='/opt/aps/workdir/input/data/OpenThoughts-dpsk_format-114k-math-acc_0_0.1.jsonl',
                        type=str,
                        help='Original queries with responses data path')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--l_split', default=0, type=int,
                        help='Left boundary for dataset selection (inclusive)')
    parser.add_argument('--r_split', default=10000000, type=int,
                        help='Right boundary for dataset selection (exclusive)')
    parser.add_argument('--tp', default=4, type=int,
                        help='Tensor parallel')
    parser.add_argument('--bz', default=32, type=int,
                        help='Batch size of each iteration.')
    # sub_thinking_path = "/opt/aps/workdir/32b-example.jsonl"
    args = parser.parse_args()
    sub_thinking_path = (
        args.data_path
    )
    model_path = args.model_path
    # model_path = "/capacity/userdata/models/DeepSeek-R1-Distill-Qwen-1.5B"
    # output_path = f"/opt/aps/workdir/multi6/myq/test_output/OpenThoughts_code-dpsk_format-acc_0_0.1-l{args.l_split}_r{args.r_split}.jsonl"
    output_path = os.path.join(args.output_dir, f"l{args.l_split}_r{args.r_split}.jsonl")

    sub_thinking_data = load_jsonl(sub_thinking_path)[args.l_split:args.r_split]

    # model = LLM(model=model_path, tensor_parallel_size=8, enforce_eager=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    code_stop_words = [
        tokenizer.eos_token,
        "<|endoftext|>",
        "<|im_end|>",
        "```"
    ]
    rollout_stop_words = [
        tokenizer.eos_token,
        "<|endoftext|>",
        "<|im_end|>",
    ]

    # prepare data
    print("-" * 10, "Preparing Data", "-" * 10)
    for each_data in tqdm(sub_thinking_data):
        max_code_num = random.randint(0, 3)
        each_data["max_code_num"] = max_code_num
        each_data["cur_code_num"] = 0
        each_data["pred"] = each_data["combined_text"].replace("<think>", "", 1)
        each_data["prompt"] = process_input(each_data["question"], tokenizer)

    # print(sub_thinking_data[0])
    # exit(0)

    code_sampling_params = SamplingParams(
        top_p=1,
        temperature=1,
        max_tokens=2000,
        stop=code_stop_words,
        n=1,
    )
    rollout_sampling_params = SamplingParams(
        top_p=0.95,
        temperature=0.6,
        max_tokens=20000,
        stop=rollout_stop_words,
        n=1,
    )

    key_words = [
        "Wait",
        "But",
        "Alternatively",
        "Hmm",
        "Moreover",
        "Furthermore",
    ]

    model = LLM(model=model_path, tensor_parallel_size=args.tp, enforce_eager=True, enable_prefix_caching=True)

    processed_prompts = []

    for i in range(0, len(sub_thinking_data), args.bz):
        loop_data = sub_thinking_data[i:i + args.bz]
        generation_loop(sub_thinking_data=loop_data, model=model, bz=32,
                        code_sampling_params=code_sampling_params, rollout_sampling_params=rollout_sampling_params,
                        output_path=output_path)
