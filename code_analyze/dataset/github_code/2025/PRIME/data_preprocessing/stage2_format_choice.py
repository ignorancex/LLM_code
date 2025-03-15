import os
import json
import torch
import argparse
from IPython import embed
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def create_question_prompt(item):
    system_prompt = """You are a precise mathematical question reformatter that converts multiple choice questions into standard format and ONLY outputs the rephrased question.

CRITICAL REFORMATTING RULES:
1. Remove all letter labels (A), (B), etc. from the question, make it not like a multi-choice question, and always phrase the question to ask for a calculation or solution
2. NEVER include the answer in the rephrased question
3. Keep all mathematical notation and rigor intact

Example 1:

Input:
ORIGINAL QUESTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, then the number of sides of this regular polygon is ( )\n\nA: $12$\n\nB: $10$\n\nC: $8$\n\nD: $6$

Output: Given that an interior angle of a regular polygon is $144^{\\circ}$, calculate the number of sides of this regular polygon.

Example 2:

Input:
ORIGINAL QUESTION: Given the function $f(x)=-x^{5}-3x^{3}-5x+3$, if $f(a)+f(a-2) > 6$, then the range of the real number $a$ is $(\\quad)$  \nA: $(-\\infty,3)$  \nB: $(3,+\\infty)$  \nC: $(1,+\\infty)$  \nD: $(-\\infty,1)$

Output: Given the function $f(x)=-x^{5}-3x^{3}-5x+3$, if $f(a)+f(a-2) > 6$, determine the range of the real number $a$.

Example 3:

Input:
ORIGINAL QUESTION: Given the sequence ${a_n}$, the sum of the first n terms, S<sub>n</sub>, is equal to $n^2$. Find the value of $a_3^2 - a_2^2$.\nA: 9\nB: 16\nC: 21\nD: 11

Output: Given the sequence ${a_n}$, the sum of the first n terms, S<sub>n</sub>, is equal to $n^2$. Find the value of $a_3^2 - a_2^2$.

IMPORTANT: You MUST respond ONLY with the rephrased question. Do not include any other text or explanation."""

    user_prompt = f"""Reformat this question following the above rules exactly:
ORIGINAL QUESTION: {item['prompt']}"""

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return chat_messages

def create_solution_prompt(item):
    system_prompt = """Given a math question and original solution, you are a precise mathematical solution reformatter that rephrases step-by-step solutions to match the given question without multiple choices.

CRITICAL REFORMATTING RULES:
1. Solution should match the original step-by-step explanation without given choices, just to solve the problem step-by-step without caring what option is correct
2. Keep all mathematical notation and rigor intact
3. The final answer MUST be wrapped in LaTeX boxed format: $\\boxed{}$

Example 1:

Input:
QUESTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, then the number of sides of this regular polygon is ( )\n\nA: $12$\n\nB: $10$\n\nC: $8$\n\nD: $6$
ORIGINAL SOLUTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, we can find the number of sides of this polygon by following these steps:\n\n1. Calculate the exterior angle of the polygon. The sum of an interior angle and its corresponding exterior angle is $180^{\\circ}$. Therefore, the exterior angle is $180^{\\circ} - 144^{\\circ} = 36^{\\circ}$.\n\n2. The sum of all exterior angles of any polygon is always $360^{\\circ}$. To find the number of sides (n) of the polygon, we use the formula for the sum of exterior angles: $n \\times \\text{exterior angle} = 360^{\\circ}$. Substituting the exterior angle we found:\n\n\\[n = \\frac{360^{\\circ}}{36^{\\circ}} = 10\\]\n\nTherefore, the number of sides of this regular polygon is $\\boxed{10}$, which corresponds to choice $\\boxed{B}$.

Output: 
REPHRASED SOLUTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, we can find the number of sides of this polygon by following these steps:\n\n1. Calculate the exterior angle of the polygon. The sum of an interior angle and its corresponding exterior angle is $180^{\\circ}$. Therefore, the exterior angle is $180^{\\circ} - 144^{\\circ} = 36^{\\circ}$.\n\n2. The sum of all exterior angles of any polygon is always $360^{\\circ}$. To find the number of sides (n) of the polygon, we use the formula for the sum of exterior angles: $n \\times \\text{exterior angle} = 360^{\\circ}$. Substituting the exterior angle we found:\n\n\\[n = \\frac{360^{\\circ}}{36^{\\circ}} = 10\\]\n\nTherefore, the number of sides of this regular polygon is $\\boxed{10}$.

IMPORTANT: You MUST respond ONLY with the rephrased solution. Do not include any other text or explanation. Always ensure the final answer is wrapped in LaTeX boxed format $\\boxed{}$"""

    user_prompt = f"""Reformat this solution to match the rephrased question:
QUESTION: {item['prompt']}
ORIGINAL SOLUTION: {item['solution']}"""

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return chat_messages

def create_filter_prompt(item):
    system_prompt = """You are an expert at analyzing mathematical questions. Your task is to determine if multiple choice questions can be solved by direct calculation or derivation, without needing to compare specific given options.

Key classification rules:
1. If the question asks for a specific mathematical result (like calculate, solve, find, derive), it's REMOVABLE
2. If the question requires choosing between specific statements or properties that aren't derivable without seeing them, it's NOT REMOVABLE
3. If the question has a definite mathematical answer that can be calculated, it's REMOVABLE, even if it seems complex

You should strictly respond in this exact format:
REMOVABLE: [Yes/No]
REASON: [1-2 sentences explaining why]
NEW_QUESTION: [restate the original question only if the choices are removable, otherwise leave blank]

Examples:

Example 1:

Input:
QUESTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, then the number of sides of this regular polygon is ( )
A: $12$
B: $10$
C: $8$
D: $6$

Output:
REMOVABLE: Yes
REASON: The answer can be directly calculated using the formula for interior angles of regular polygons.
NEW_QUESTION: Given that an interior angle of a regular polygon is $144^{\\circ}$, calculate the number of sides of this regular polygon.

Example 2:

Input:
QUESTION: Given f(x)=x²+1, which statement is true? (A) f is decreasing (B) f has minimum at x=0 (C) f(-1)=f(1) (D) f has no real roots

Output:
REMOVABLE: No
REASON: Question requires evaluation of specific mathematical statements, meaningless without the given options.
NEW_QUESTION:

Example 3:

Input:
QUESTION: Given f(x)=2a-sin x, then f''(x)= ?
(A) cos x  (B) -cos x  (C) 2+cos x  (D) 2-cos x

Output:
REMOVABLE: Yes
REASON: This is a direct calculus problem asking to find the second derivative, which can be calculated through standard differentiation.
NEW_QUESTION: Given f(x)=2a-sin x, find f''(x).

Example 4:

Input:
QUESTION: Given the function f(x)=-x⁵-3x³-5x+3, if f(a)+f(a-2) > 6, then what is the range of the real number a?
(A) (-∞,3)  (B) (3,+∞)  (C) (1,+∞)  (D) (-∞,1)

Output:
REMOVABLE: Yes
REASON: This is a mathematical problem that can be solved by analyzing the inequality and finding the range directly.
NEW_QUESTION: Given the function f(x)=-x⁵-3x³-5x+3, if f(a)+f(a-2) > 6, find the range of the real number a.

You should strictly respond in this exact format, and keep all mathematical notation and rigor intact:
REMOVABLE: [Yes/No]
REASON: [1-2 sentences explaining why]
NEW_QUESTION: [restate the original question only if the choices are removable, otherwise leave blank]"""

    user_prompt = f"""QUESTION: {item['prompt']}"""

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return chat_messages

def filter_questions(data, llm, tokenizer, sampling_params):
    new_data = []
    
    retry_items = data.copy()
    retry_count = 0
    max_retries = 10
    
    while retry_items and retry_count < max_retries:
        length = len(retry_items)
        print(f"\nAttempt {retry_count + 1}, processing {length} items")
        
        prompts = [tokenizer.apply_chat_template(
                create_filter_prompt(item),
                tokenize=False,
                add_generation_prompt=True
            ) for item in retry_items]
        outputs = llm.generate(prompts, sampling_params)

        next_retry_items = []
        current_results = []
        
        for i in tqdm(range(length)):
            item = data[i]
            output = outputs[i].outputs[0].text.strip()
            try:
                l1, r1 = output.split("NEW_QUESTION:")
                new_question = r1.strip()
                
                l2, r2 = l1.split("REASON:")
                reason = r2.strip()
                
                removable = l2.split("REMOVABLE:")[-1].strip().lower()
                assert removable in ['yes', 'no']
                
                item['removable'] = removable == 'yes'
                item['reason'] = reason
                item['reformatted_question'] = new_question if removable else ""
                
                current_results.append(item)
            except:
                next_retry_items.append(item)

        new_data.extend(current_results)

        retry_items = next_retry_items
        retry_count += 1
        
        if retry_items:
            print(f"Success rate: {(len(data) - len(retry_items))/len(data):.2%}")
        
    return new_data


def main():
    # TODO: CUDA_VISIBLE_DEVICES=4,5,6,7 python stage2_format_choice.py
    date = "20241212" # TODO
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=f"{date}/stage1_filtered.jsonl")
    parser.add_argument("--output_dir", type=str, default=f"{date}/formatted")
    parser.add_argument("--model_name_or_path", type=str, default="path/to/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize LLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8
    )

    # Load data
    _data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            _data.append(json.loads(line))

    # _data = _data[:10]
    all_outputs = []
    data = [item for item in _data if item['type'] == "choice" and "following" not in item['prompt'].lower() and "statement" not in item['prompt'].lower()]
    print(f"Totally {len(data)}...")
    
    # TODO
    turn = 0
    max_samples = 40000
    initial_length = min((turn + 1) * max_samples, len(data)) - turn * max_samples

    print(f"Total number of data to be formatted: from {turn * max_samples} to {min((turn + 1) * max_samples, len(data))}")
    
    data = data[turn * max_samples: min((turn + 1) * max_samples, len(data))]

    # filter question should with choice
    data = filter_questions(data, llm, tokenizer, sampling_params)
    data = [item for item in data if item['removable']]
    
    print(f"After filtering the question which should with the options, remaining {len(data)}, rate: {round(len(data) / initial_length, 4)}...")
    
    retry_items = data.copy()
    retry_count = 0
    max_retries = 10
    
    while retry_items and retry_count < max_retries:
        length = len(retry_items)
        print(f"\nAttempt {retry_count + 1}, processing {length} items")

        prompts = []
        prompts.extend([tokenizer.apply_chat_template(
            create_question_prompt(item),
            tokenize=False,
            add_generation_prompt=True
        ) for item in retry_items])
        prompts.extend([tokenizer.apply_chat_template(
            create_solution_prompt(item),
            tokenize=False,
            add_generation_prompt=True
        ) for item in retry_items])
        
        outputs = llm.generate(prompts, sampling_params)

        next_retry_items = []
        current_results = []
        
        for i in tqdm(range(length)):
            item = retry_items[i]
            output_q = outputs[i].outputs[0].text.strip()
            output_s = outputs[length + i].outputs[0].text.strip()

            try:
                question = output_q
                solution = output_s.split("REPHRASED SOLUTION:")[-1].strip()

                result = {
                    "id": item["id"],
                    "source": item["source"],
                    "remove": item['remove'],
                    "prompt": question,
                    "solution": solution,
                    "reference": item['reference'],
                    "type": "formatted_qa"
                }
                current_results.append(result)
            except Exception as e:
                next_retry_items.append(item)

        all_outputs.extend(current_results)

        retry_items = next_retry_items
        retry_count += 1
        
        if retry_items:
            print(f"Success rate: {(len(data) - len(retry_items))/len(data):.2%}")

    all_outputs = sorted(all_outputs, key=lambda x: int(x['id']))
    all_outputs = [item for item in all_outputs if 'option' not in item['solution'].lower()]
    # Save results
    with open(os.path.join(args.output_dir, f"{turn}.jsonl"), 'w', encoding='utf-8') as f:
        for output in all_outputs:
            f.write(json.dumps(output) + '\n')

if __name__ == "__main__":
    main()