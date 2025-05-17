from openai import OpenAI
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 模型和语言的组合
models = ["deepseek-reasoner"]
languages = ["cpp"]

# 输入文件路径
input_path = "LLM_code/codeforces/subset_select/benchmark.jsonl"

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="https://api.deepseek.com/beta",
    api_key="",  # 请确保 key 保密
)

# 单次生成函数
def generate_code(prompt, idx, model, lang):
    try:
        # 设置前缀，如 "```python\n" 或 "```cpp\n"
        prefix = f"\n```{lang}\n"
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix, "prefix": True}
            ],
            stop=["```"],
        )
        code = completion.choices[0].message.content.strip()
    except Exception as e:
        code = f"[Error: {e}]"
    return idx, code

# 加载 benchmark 数据
with open(input_path, "r", encoding="utf-8") as fin:
    problems = [json.loads(line) for line in fin if line.strip()]

# 遍历所有模型和语言组合
for model in models:
    for lang in languages:
        output_path = f"LLM_code/codeforces/subset_select/deepseek_{model.split('-')[-1]}_{lang}.jsonl"

        # 读取已有结果
        existing_results = {}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        existing_results[obj["problem"]] = obj

        with open(output_path, "w", encoding="utf-8") as fout:
            for item in tqdm(problems, desc=f"Processing {model} / {lang}"):
                problem = item.get("problem", "")
                context = item.get("context_plain", "").strip()

                # 构造 prompt
                prompt = (
                    f"Your task is to carefully read the following problem description and implement a solution in {lang}. "
                    "Return only the code without any explanations. Here is the problem description:\n\n"
                    f"{context}"
                )

                # 判断是否已有结果
                if problem in existing_results:
                    result = existing_results[problem]
                    missing_or_error_indices = [
                        i for i in range(1, 33)
                        if f"pass@{i}" not in result or str(result[f"pass@{i}"]).startswith("[Error")
                    ]
                    if not missing_or_error_indices:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        continue
                else:
                    result = {"problem": problem}
                    missing_or_error_indices = list(range(1, 33))

                # 并发生成缺失的代码
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(generate_code, prompt, i, model, lang): i
                        for i in missing_or_error_indices
                    }

                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"Fixing {problem}", leave=False, ncols=80):
                        idx, code = future.result()
                        result[f"pass@{idx}"] = code

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
