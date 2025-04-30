import json
from tqdm import tqdm
import os
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "fake-api-key"
openai_api_base = "http://localhost:7999/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# ✅ 只使用 GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 处理每个问题的函数
def process_problem(problem, processed_submission_ids):
    context_plain = problem["context_plain"]
    source_code = problem.get("sourceCode", "")
    languages_id = problem["languages_id"]
    submission_id = problem["submission_id"]

    # 如果该 submission_id 已经处理过，跳过
    if submission_id in processed_submission_ids:
        print(f"Skipping submission {submission_id} as it has already been processed.")
        return problem, processed_submission_ids

    # 根据编程语言生成语言的提示
    language = "python" if languages_id == 8 else "cpp"

    prompt_1 = f"""
    Your task is to carefully read the following problem description and implement a solution in {language}.
    Please first provide your reasoning in plain text, and then provide the corresponding code.
    Format your response as follows using Markdown:

    ### Reasoning

    <Please provide only your step-by-step reasoning in plain text here.>

    ### Code

    <Please provide only your code in {language} here, with no extra explanation or text.>


    Here is the problem description:

    {context_plain}

    """

    prompt_2 = f"""
    Your task is to carefully read the following problem description and revise the given code.
    The code given is AC code. ( correct and has passed the test. )
    Please first provide your reasoning in plain text, and then provide the corresponding code.
    Format your response as follows using Markdown:
    
    ### Reasoning

    <Please provide only your step-by-step reasoning in plain text here.>

    ### Code

    <Please provide only your code in {language} here, with no extra explanation or text.>

    
    Here is the problem description:

    {context_plain}

    Here is the user's AC Code:

    {source_code}

    """

    # 函数：调用Qwen模型生成代码
    def generate_code(prompt0):
        completion = client.completions.create(model="/mnt/nvme0/QwQ-32B",
                                      prompt=prompt0,max_tokens=20000)
        return completion.choices[0].text


    # 如果没有generate_code字段，生成方式一的代码
    if "generate_code" not in problem:
        generated_code = generate_code(prompt_1)
        problem["generate_code"] = generated_code

    # 如果没有generate_code_ref字段，生成方式二的代码
    if "generate_code_ref" not in problem:
        generated_code_ref = generate_code(prompt_2)
        problem["generate_code_ref"] = generated_code_ref

    # 记录已处理的submission_id
    processed_submission_ids.add(submission_id)

    return problem, processed_submission_ids

# 读取原始文件并处理
def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_submission_ids = set()

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            processed_submission_ids = {problem["submission_id"] for problem in existing_data}
    except FileNotFoundError:
        existing_data = []

    save_every = 1  # ✅ 每10个保存一次

    for idx, problem in enumerate(tqdm(data, desc="Processing Problems")):
        if problem["submission_id"] in processed_submission_ids:
            print(f"Skipping submission {problem["submission_id"]} as it has already been processed.")
        else:
            updated_problem, processed_submission_ids = process_problem(problem, processed_submission_ids)
            existing_data.append(updated_problem)

            if (idx + 1) % save_every == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # ✅ 最后统一保存一次，防止遗漏
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print("File processing complete.")

# 输入输出文件路径
language = "cpp"
input_file = f"dataset/unique_problem_{language}.json"
output_file = f"qwq_32b_{language}.json"

# 执行处理
process_file(input_file, output_file)
