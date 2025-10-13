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
    parser.add_argument("--output", type=str, default="wildjailbreak_eval.csv")
    parser.add_argument("--n", type=int, default=250)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")

    full = load_dataset("allenai/wildjailbreak", "train", keep_default_na=False, token=hf_token)["train"]
    small = full.select(range(args.n))
    prompts = [ex["vanilla"] or ex["adversarial"] for ex in small]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    model.to(device).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        top_p=1.00,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    headers = ["prompt", "response"]
    df_init = pd.DataFrame(columns=headers)
    df_init.to_csv(args.output, index=False)

    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n=== Prompt {idx}/{len(prompts)} ===")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)
        texts = tokenizer.batch_decode(outs, skip_special_tokens=True)
        responses = [t[len(prompt):].strip() for t in texts]
        row = [prompt] + responses
        pd.DataFrame([row], columns=headers).to_csv(
            args.output, mode="a", index=False, header=False
        )
        del inputs, outs, texts, responses, row
        torch.cuda.empty_cache()

    # Moderation
    df = pd.read_csv(args.output)
    tokenizer_m, model_m = load_llamaguard()
    safe1_count = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        flag = moderate(row["prompt"], row["response"], tokenizer_m, model_m)
        df.at[i, "response_flag"] = flag
        if flag == "safe":
            safe1_count += 1
    df.to_csv(args.output, index=False)
    print(f"Safe@1: {(safe1_count / len(df)) * 100:.2f}%")
    print(f"Saved flagged output to {args.output}")

if __name__ == "__main__":
    main()
