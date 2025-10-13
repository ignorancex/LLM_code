import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.moderation import load_llamaguard, moderate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="wildchat_eval.csv")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")
    dataset = load_dataset("allenai/WildChat", split=f"train[:{args.n}]", token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto").eval()
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    def generate(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
    rows = []
    for row in tqdm(dataset, total=len(dataset), desc="Generating"):
        convo = row["conversation"]
        prompt = ""
        for turn in convo:
            if turn["role"] == "user":
                prompt = turn["content"]
                break
        if not prompt.strip():
            continue
        response = generate(prompt)
        rows.append({"prompt": prompt, "response": response})
    df = pd.DataFrame(rows)
    # Moderation
    tokenizer_m, model_m = load_llamaguard()
    safe_count = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        flag = moderate(row["prompt"], row["response"], tokenizer_m, model_m)
        df.at[i, "response_flag"] = flag
        if flag == "safe":
            safe_count += 1
    df.to_csv(args.output, index=False)
    print(f"Safe@1: {(safe_count / len(df)) * 100:.2f}%")
    print(f"Saved flagged output to {args.output}")

if __name__ == "__main__":
    main()
