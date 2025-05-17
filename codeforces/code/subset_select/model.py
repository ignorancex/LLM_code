from openai import OpenAI
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=""
    # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # api_key = ""

)

# 输入输出路径
input_path = "LLM_code/codeforces/subset_select/benchmark.jsonl"
output_path = "LLM_code/codeforces/subset_select/qwen_14b_cpp.jsonl"

# 读取已有输出，构建 problem -> result 映射
existing_results = {}
try:
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                existing_results[obj["problem"]] = obj
except FileNotFoundError:
    pass  # 文件不存在也没关系

# 单次生成函数
def generate_code(prompt, idx):
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>",
            },
            extra_body={"enable_thinking": False},
            model="qwen3-14b",
            messages=[{"role": "user", "content": prompt}],
        )
        code = completion.choices[0].message.content.strip()
    except Exception as e:
        code = f"[Error: {e}]"
    return idx, code

# 加载所有 benchmark 题目
with open(input_path, "r", encoding="utf-8") as fin:
    problems = [json.loads(line) for line in fin if line.strip()]

# 以覆盖模式写入（每次都写完整内容）
with open(output_path, "w", encoding="utf-8") as fout:
    for item in tqdm(problems, desc="Processing problems"):
        problem = item.get("problem", "")
        context = item.get("context_plain", "").strip()

        # 构造 prompt
        prompt = (
            "Your task is to carefully read the following problem description and implement a solution in C++. "
            "Return only the code without any explanations. Here is the problem description:\n\n"
            f"{context}"
        )

        # 判断是否需要修复
        if problem in existing_results:
            result = existing_results[problem]
            missing_or_error_indices = [
                i for i in range(1, 33)
                if f"pass@{i}" not in result or str(result[f"pass@{i}"]).startswith("[Error")
            ]
            if not missing_or_error_indices:
                # 全部正常，直接写回
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue
        else:
            result = {"problem": problem}
            missing_or_error_indices = list(range(1, 33))

        # 并发生成缺失/错误项
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(generate_code, prompt, i): i
                for i in missing_or_error_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fixing {problem}", leave=False, ncols=80):
                idx, code = future.result()
                result[f"pass@{idx}"] = code

        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()
