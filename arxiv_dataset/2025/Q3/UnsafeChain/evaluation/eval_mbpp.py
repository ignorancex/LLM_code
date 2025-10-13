import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="mbpp_eval.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    df = pd.read_csv(args.input_csv)
    responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating with model"):
        prompt = row["text"].strip() + "\n\nLet's think step by step to arrive at the correct answer.\n\nReasoning:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        out_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated = out_text[len(prompt):].strip()
        responses.append(generated)
    df["response"] = responses
    df.to_csv(args.output, index=False)
    print(f"✅ Saved → {args.output}")

if __name__ == "__main__":
    main()
