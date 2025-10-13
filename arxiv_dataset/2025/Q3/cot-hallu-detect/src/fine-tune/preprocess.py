
import json

# see https://github.com/sylinrl/TruthfulQA/tree/main/data 
# for finetune_truth.jsonl and finetune_info.jsonl

def main(
    data_path: str,
    # finetune_truth.jsonl
    # finetune_info.jsonl
):

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    data_t = [
        {
            "messages": [
                {"role": "user", "content": d["prompt"]},
                {"role": "assistant", "content": d["completion"]},
            ]
        } for d in dataset if d
    ]

    out = data_path + "-template.json"
    with open(out, "w") as f:
        f.write(json.dumps(data_t))
