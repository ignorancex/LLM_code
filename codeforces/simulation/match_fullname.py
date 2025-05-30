import os
import json

def enrich_with_fullname(model_dir, mapping_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(model_dir):
        if not file.endswith(".json"):
            continue

        # 识别语言
        parts = file.split("_")
        lang = parts[-1].replace(".json", "")

        # 加载映射表
        mapping_path = os.path.join(mapping_dir, f"unique_problem_{lang}.json")
        if not os.path.exists(mapping_path):
            print(f"⚠️ 跳过 {file}：找不到 {mapping_path}")
            continue

        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        id2fullname = {item["submission_id"]: item["fullname"] for item in mapping_data}

        # 加载模型数据
        input_path = os.path.join(model_dir, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 匹配并添加 fullname
        for item in data:
            sid = item.get("submission_id")
            item["fullname"] = id2fullname.get(sid, "")

        # 保存到输出目录
        out_path = os.path.join(output_dir, file)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ enriched: {file} → {out_path}")

# === 执行脚本 ===
enrich_with_fullname(
    model_dir="LLM_code/codeforces/simulation/output_cleaned",
    mapping_dir="LLM_code/codeforces/simulation",
    output_dir="LLM_code/codeforces/simulation/models"
)
