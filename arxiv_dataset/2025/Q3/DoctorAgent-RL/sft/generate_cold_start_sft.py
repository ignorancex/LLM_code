import json
import openai
from openai import OpenAI
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import pandas as pd
import copy
import threading

# 创建 API 客户端
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com/v1")

# prompt 模板
def make_prompt(ori_prompt, ori_response):
    ori_prompt = list(ori_prompt)
    ori_prompt.append({"content": ori_response, "role": "assistant"})
    return (
        f"{str(ori_prompt)}\n"
        f"Help me supplement the thinking process in the content of each assistant and modify it into the format of <think> </think><answer> </answer>. "
        f"Directly modify the content of each assistant in the original content and output the content with json format (put into ```json```) without any other words."
    )

def expand_patient_info(case, max_retries=3):
    print(f"Start: {time.time()}, Thread: {threading.current_thread().name}")
    ori_prompt = case["prompt"]
    ori_response = case["response"]
    
    prompt_text = make_prompt(ori_prompt, ori_response)

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a medical assistant that helps expand doctor's thinking for better clarifying patients' symptoms."},
                    {"role": "user", "content": prompt_text}
                ],
                stream=False
            )
            expanded_info = response.choices[0].message.content.strip()
            
            # Create new JSON with all original content plus expanded info
            expanded_case = copy.deepcopy(case)
            expanded_case["prompt"] = expanded_info
            print(f"End: {time.time()}, Thread: {threading.current_thread().name}")
            return expanded_case

        except Exception as e:
            print(f"[Retry] Request failed, error: {e}")
            time.sleep(2)

    print("[Failed] Expansion failed, returning original")
    return case

def process_all_cases(input_path, output_path, start_index=0, max_workers=4):
    if ".parquet" in input_path:
        df = pd.read_parquet(input_path)
        cases = df.to_dict(orient='records')[:1000]
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

    print(f"Loaded {len(cases)} cases, starting from index {start_index}... Using {max_workers} workers")
    expanded_cases = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(expand_patient_info, cases[i]): i
            for i in range(start_index, len(cases))
        }

        completed_count = 0
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing"):
            idx = future_to_idx[future]
            try:
                expanded = future.result()
                expanded_cases.append(expanded)
                completed_count += 1

                if completed_count % 100 == 0 or completed_count + start_index == len(cases):
                    if ".parquet" in output_path:
                        df = pd.DataFrame(expanded_cases)
                        df.to_parquet(output_path, index=False)
                    else:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(expanded_cases, f, ensure_ascii=False, indent=2)
                    print(f"{completed_count} cases processed, saved to: {output_path}")
            except Exception as e:
                print(f"[Error] Case {idx+1} failed: {e}")
    
    if ".parquet" in output_path:
        df = pd.DataFrame(expanded_cases)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(expanded_cases, f, ensure_ascii=False, indent=2)
    print(f"All done! Final results saved to: {output_path}")


if __name__ == "__main__":
    process_all_cases(
        input_path="data/train_en_8068_all_format_filter_4k_sft_diagnosis.parquet",
        output_path="data/train_en_8068_all_format_filter_4k_sft_w_thinking_1k.parquet",
        start_index=0,
        max_workers=4
    )