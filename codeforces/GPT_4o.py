import json
from openai import AzureOpenAI
from tqdm import tqdm

# 初始化 Azure OpenAI 客户端
client = AzureOpenAI(
    api_key="",  # 替换为你的API Key
    api_version="2024-02-01",  # 替换为正确的API版本
    azure_endpoint="https://mllm-as-a-judge.openai.azure.com/openai/deployments/gpt-4o-mini-llm-influence-s3/chat/completions?api-version=2024-08-01-preview"  # 替换为你的Azure OpenAI Endpoint
)

input_file = "LLM_code/codeforces/problem_1000.jsonl"     # 输入 JSONL 文件路径
output_file = "LLM_code/codeforces/GPT_code.jsonl"     # 输出结果文件路径

def build_prompt(problem):
    return f"""
You are given a programming problem. Please write correct and efficient Python code to solve it.

Description:
{problem['description']}

Input Specification:
{problem['input_spec']}

Output Specification:
{problem['output_spec']}

Just return Python code. Do not explain anything.
"""

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Processing problems"):
        problem = json.loads(line)
        src_uid = problem.get("src_uid")
        prompt = build_prompt(problem)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-llm-influence-s3",
                messages=[{"role": "user", "content": prompt}]
            )
            code = response.choices[0].message.content
        except Exception as e:
            code = f"Error: {str(e)}"
        
        result = {
            "src_uid": src_uid,
            "code": code
        }
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
