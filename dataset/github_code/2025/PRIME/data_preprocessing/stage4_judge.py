import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import re, os
import random
from collections import Counter
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from IPython import embed
from math_util import *
from openai import OpenAI
import torch

import time
import concurrent.futures
from threading import Lock

class RateLimiter:
    def __init__(self, max_per_second):
        self.lock = Lock()
        self.last_call = 0
        self.min_interval = 1.0 / max_per_second

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


class MathProblemValidator:
    def __init__(self, model_path: str, tp_size: int, try_num: int, use_fp8):
        if use_fp8:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=0.95, quantization="fp8")
        else:
            if 'gemini' in model_path:
                self.client = OpenAI(
                    api_key="<api_key>",
                    base_url="<base_url>",
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.llm = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=0.95) # TODO
        self.try_num = try_num
        self.sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=16384, # TODO
            top_p=0.95
        )

    def create_cnk12_prompt(self, problem: str) -> List[Dict[str, str]]:
        aspect = """Solve this problem step by step in Latex format. If you encounter any issues so that you can't solve the problem and find it no solution, stop and explain why. If the problem has solution, you should show complete reasoning trace.

Your response MUST follow this exact format:
[If SOLVABLE: Show full solution steps and final answer in $\\boxed{{}}$. If UNSOLVABLE: Clearly state which condition makes it impossible to solve and why, at last output $\\boxed{{No solution}}$.]"""

        system_prompt = """You are a mathematical validator specializing in high school and college-level mathematics. Focus on the specific validation aspect requested."""

        user_prompt = f"""Problem: {problem}

Analysis Required:
{aspect}"""
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return chat_messages

    def create_aops_forum_prompt(self, problem: str) -> List[Dict[str, str]]:
        system_prompt = """You are a mathematical validator specializing in competition mathematics from basic contest level through olympiad level. Focus on providing rigorous solutions with clear mathematical reasoning."""

        aspect = """Solve this problem step by step in Latex format. If you encounter any issues so that you can't solve the problem and find it no solution, stop and explain why. If the problem has solution, you should show complete reasoning trace.

Your response MUST follow this exact format:
[If SOLVABLE: Show full solution steps and final answer in $\\boxed{{}}$. If UNSOLVABLE: Clearly state which condition makes it impossible to solve and why, at last output $\\boxed{{No solution}}$.]"""

        user_prompt = f"""Problem: {problem}

Analysis Required:
{aspect}"""
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return chat_messages

    def create_olympiads_prompt(self, problem: str) -> List[Dict[str, str]]:
        system_prompt = """You are a mathematical validator specializing in olympiad-level mathematics covering algebra, number theory, geometry, and combinatorics. Focus on providing rigorous solutions with clear mathematical reasoning."""

        aspect = """Solve this problem step by step in Latex format. If you encounter any issues so that you can't solve the problem and find it no solution, stop and explain why. If the problem has solution, you should show complete reasoning trace.

Your response MUST follow this exact format:
[If SOLVABLE: Show full solution steps and final answer in $\\boxed{{}}$. If UNSOLVABLE: Clearly state which condition makes it impossible to solve and why, at last output $\\boxed{{No solution}}$.]"""

        user_prompt = f"""Problem: {problem}

Analysis Required:
{aspect}"""
        
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return chat_messages

    def create_prompt(self, source, problem: str) -> List[Dict[str, str]]:
        
        if source == "cn_k12" or source == "synthetic_amc" or source == "amc_aime" or source == "synthetic_math":
            return self.create_cnk12_prompt(problem)
        elif source == "aops_forum":
            return self.create_aops_forum_prompt(problem)
        elif source == "olympiads":
            return self.create_olympiads_prompt(problem)

    def chat_complete(self, messages):
        self.rate_limiter.wait()
        response = self.client.chat.completions.create(
            model="gemini-2.0-flash-thinking-exp-1219",
            messages=messages,
            temperature=0.8,
            max_completion_tokens=16384,
            n=1,
        )
        content = response.choices[0].message.content
        return content.strip() 

    def generate_gemini_solutions(self, source, problem_list: List[str]) -> List[str]:
        # Create prompts
        prompts = []
        for problem in problem_list:
            chat_messages = self.create_prompt(source, problem)
            prompts.extend([chat_messages] * self.try_num)
        
        max_workers = min(32, len(prompts))
        results = [None] * len(prompts)
        
        self.rate_limiter = RateLimiter(max_per_second=50) # TODO
        
        def process(prompt, idx):
            try:
                solution = self.chat_complete(prompt)
                return idx, solution
            except Exception as e:
                return idx, f"Error: {str(e)}"

        retry_count = 0
        max_retries = 10
        failed_indices = list(range(len(prompts)))

        while failed_indices and retry_count < max_retries:
            before_failed = len(failed_indices)
            print(f"\nAttempt {retry_count + 1}, processing {len(failed_indices)} requests...")
            time.sleep(5)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(process, prompts[idx], idx): idx 
                    for idx in failed_indices
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(failed_indices)):
                    idx, result = future.result()
                    results[idx] = result
            
            failed_indices = [
                idx for idx, result in enumerate(results) if "Error" in result
            ]
            
            retry_count += 1
            success_rate = 1 - len(failed_indices) / before_failed
            print(f"Success rate: {success_rate:.4f}")
            
            # Adjust rate limit based on success rate
            if success_rate < 0.5:
                # Reduce rate if success rate is low
                current_rate = 1.0 / self.rate_limiter.min_interval
                new_rate = max(1, current_rate * 0.8)  # Reduce by 20% but not below 1
                self.rate_limiter = RateLimiter(max_per_second=new_rate)
                print(f"Adjusting rate limit to {new_rate:.2f} requests per second")

        return results

    def generate_solutions(self, source, problem_list: List[str]) -> List[str]:
        prompts = []
        for problem in problem_list:
            chat_messages = self.create_prompt(source, problem)
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.extend([formatted_prompt] * self.try_num)
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    @staticmethod
    def analyze_solutions(attempts: List[str], ground_truth) -> Dict[str, Any]:
        def extract_status_and_steps(response: str) -> Dict:
            return {
                "solution": response
            }

        # Process all attempts
        parsed_attempts = [extract_status_and_steps(attempt) for attempt in attempts]

        
        # TODO: 对于两小问的问题，先统计数量然后去除掉
        is_matched_2, ground_truth = match_answer(ground_truth)
        for attempt in parsed_attempts:
            is_matched_1, given_answer = match_answer(attempt["solution"])

            attempt['status'] = "undefined"
            attempt["solution_matched"] = given_answer
            attempt["ground_truth_matched"] = ground_truth

            if "No solution" in given_answer:
                attempt['status'] = "unsolvable"
            else:
                attempt['status'] = "solvable"

            if not is_matched_1:
                attempt['match_result'] = "given_answer not match"
                continue
            if not is_matched_2:
                attempt['match_result'] = "ground_truth not match"
                continue

            if grade_answer(given_answer, ground_truth):
                attempt['match_result'] = "answer matched"
            else:
                attempt['match_result'] = "answer not matched"
        
        status_counts = Counter(attempt["status"] for attempt in parsed_attempts)
        majority_status = status_counts.most_common(1)[0][0]
        status_consistency = status_counts[majority_status] / len(attempts)

        match_counts = Counter(attempt["match_result"] for attempt in parsed_attempts)
        majority_match = match_counts.most_common(1)[0][0]
        answer_consistency = match_counts[majority_match] / len(attempts)
        
        return {
            "attempts": parsed_attempts,
            "solution_status": {
                "majority": majority_status,
                "consistency": status_consistency
            },
            "match_results": {
                "majority": majority_match,
                "consistency": answer_consistency
            }
        }
        

class DataProcessor:
    def __init__(self, input_file: str, output_path: str):
        self.input_file = input_file
        self.output_path = output_path
    
    def load_test_data(self):
        dataset = {
            "synthetic_math": [],
            "cn_k12": [],
            "olympiads": []
        }
        with open("path/to/test_samples.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            dataset[item["source"]].append(item)

        for source in dataset.keys():
            print(f"{source}: {len(dataset[source])}...")

        return dataset


    def load_data(self) -> List[Dict]:
        dataset = {
            "synthetic_math": [],
            "cn_k12": [],
            "olympiads": [],
            "aops_forum": [],
            "synthetic_amc": [],
            "amc_aime": [],
        }
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset[data['source']].append(data)
        
        for source in dataset.keys():
            # dataset[source] = dataset[source][:10000]
            # dataset[source] = random.sample(dataset[source], 100) # TODO
            print(f"{source}: {len(dataset[source])}...")

        return dataset

    def save_results(self, data: Dict):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def sample():
    dataset = []
    with open("path/to/final_math_qa.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    sampled = random.sample(dataset, 100)
    with open("sampled.jsonl", 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=4)
    

def main():
    # TODO: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python stage4_judge.py
    date = "20241128" # TODO
    parser = argparse.ArgumentParser()
    # deepseek-math-7b-instruct
    # Qwen2.5-Math-72B-Instruct
    # QwQ-32B-Preview
    # gemini-2.0-flash-thinking-exp
    MAX_NUM = 10000 # TODO
    parser.add_argument("--turn", type=int, default=2) # TODO
    parser.add_argument("--data_subset", type=str, default="olympiads") # TODO
    parser.add_argument("--use_fp8", type=bool, default=False) # TODO
    parser.add_argument("--model_path", type=str, default="gemini-2.0-flash-thinking-exp") # TODO
    parser.add_argument("--input_file", type=str, default=f"{date}/stage3_final_math_qa.jsonl")
    parser.add_argument("--output_path", type=str, default=f"{date}/stage4_judged")
    parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--try_num", type=int, default=5)
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.data_subset)
    os.makedirs(args.output_path, exist_ok=True)
    args.output_path = os.path.join(args.output_path, f"{args.model_path.split('/')[-1]}_{args.turn}.json")

    # Initialize components
    validator = MathProblemValidator(args.model_path, args.tensor_parallel_size, args.try_num, args.use_fp8)
    processor = DataProcessor(args.input_file, args.output_path)

    if args.data_subset == "test":
        dataset = processor.load_test_data()
    else:
        dataset = processor.load_data()

    for source, data in dataset.items():
        if source != args.data_subset: continue
        print(f"Processing {source}...")
        print(f"Saving to {args.output_path}...")
        data = data[(args.turn * MAX_NUM):((args.turn + 1) * MAX_NUM)]
        print(f"Running from {args.turn * MAX_NUM} to {(args.turn + 1) * MAX_NUM}...")
        num_questions = len(data)

        if 'gemini' in args.model_path:
            attempts = validator.generate_gemini_solutions(source, [item['prompt'] for item in data])
        else:
            attempts = validator.generate_solutions(source, [item['prompt'] for item in data])
        for i in range(num_questions):
            analysis = validator.analyze_solutions(attempts[i * args.try_num: (i + 1) * args.try_num], data[i]['solution'])
            data[i].update({
                "validation_results": analysis
            })
            
        cnt_solu_status = {}
        cnt_answer_match = {}
        for i in range(num_questions):
            solution_majority = data[i]["validation_results"]["solution_status"]["majority"]
            cnt_solu_status[solution_majority] = cnt_solu_status.get(solution_majority, 0) + 1
            
            answer_majority = data[i]["validation_results"]["match_results"]["majority"]
            cnt_answer_match[answer_majority] = cnt_answer_match.get(answer_majority, 0) + 1
        
        for k, v in cnt_solu_status.items():
            v /= num_questions
        
        for k, v in cnt_answer_match.items():
            v /= num_questions

        print(f"cnt_solu_status: {cnt_solu_status}")
        print(f"cnt_answer_match: {cnt_answer_match}")
        processor.save_results(data)

if __name__ == "__main__":
    main()