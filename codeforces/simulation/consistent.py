import os
import json

# 只保留这些字段
keep_fields = {
    "submission_id",
    "context_plain",
    "sourceCode",
    "languages_id",
    "generate_code",
    "generate_code_ref",
    "generate_reasoning",
    "generate_code_block",
    "generate_ref_reasoning",
    "generate_ref_code_block",
}

input_dir = "LLM_code/codeforces/simulation/output"
output_dir = "LLM_code/codeforces/simulation/output_cleaned"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        path = os.path.join(input_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 保留字段
        cleaned = [
            {k: item[k] for k in keep_fields if k in item}
            for item in data
        ]

        # 保存清理后的 JSON
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        print(f"✅ Cleaned: {filename} → {out_path}")
