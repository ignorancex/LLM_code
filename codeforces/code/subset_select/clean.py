import json
import re
from pathlib import Path
from tqdm import tqdm

# ======== 配置路径 ========
input_path = "LLM_code/codeforces/subset_select/qwen_coder_cpp.jsonl"  # 原始结果文件
output_path = "LLM_code/codeforces/models_code/qwen_coder_cpp.jsonl"  # 清洗后的输出文件

# ======== 判断语言 ========
def detect_language_by_filename(filename):
    if "_py" in filename:
        return "python"
    elif "_cpp" in filename:
        return "cpp"
    else:
        return ""

# ======== 提取代码块（不含语言标识）=======
def extract_code_block(text, language):
    if not isinstance(text, str):
        return "[Error: no code]"
    # 匹配 ```python\n...``` 或 ```cpp\n...```
    pattern = rf"```{language}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else "[Error: no code]"

# ======== 主处理逻辑 ========
def clean_jsonl_file(input_path, output_path):
    language = detect_language_by_filename(input_path)
    cleaned_data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Cleaning"):
            if not line.strip():
                continue
            obj = json.loads(line)
            problem = obj.get("problem", "")
            cleaned_obj = {"problem": problem}

            # 提取并排序所有 pass@i 的项
            pass_entries = [
                (int(k.split("@")[1]), k) for k in obj if k.startswith("pass@")
            ]
            pass_entries.sort()

            for idx, key in pass_entries:
                content = obj.get(key, "")
                code = extract_code_block(content, language)
                cleaned_obj[f"pass@{idx}"] = code

            cleaned_data.append(cleaned_obj)

    # 写入清洗后的 jsonl 文件
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in cleaned_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Cleaning completed. Output saved to {output_path}")

# ======== 运行入口 ========
if __name__ == "__main__":
    clean_jsonl_file(input_path, output_path)
