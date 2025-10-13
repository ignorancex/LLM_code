import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re

def extract_answer(text):
    text = text.strip()
    boxed_match = re.search(r"\\boxed{(\\d+)}", text)
    if boxed_match:
        return boxed_match.group(1)
    match = re.search(r"(\\d+)$", text.replace(",", ""))
    return match.group(1) if match else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="gsm8k_eval.csv")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN in your environment or .env file.")
    print("Loading GSM8K dataset…")
    gsm8k = load_dataset("gsm8k", "main", split="test", token=hf_token)
    gsm8k = gsm8k.select(range(args.n))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device).eval()
    gen_kwargs = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    results = []
    for ex in tqdm(gsm8k, desc="Evaluating GSM8K"):
        prompt = ex["question"].strip() + "\n\nLet’s think step by step.\nAnswer with final answer at the end. The last thing returned should be the final answer and just return the digits without commas, units, and any other thing. Follow the formatting strictly don't use any other formatting"
        target = extract_answer(ex["answer"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = extract_answer(output_text)
        is_correct = (pred == target)
        results.append({
            "prompt": prompt,
            "target": target,
            "model_output": output_text,
            "predicted": pred,
            "correct": is_correct
        })
    gsm_df = pd.DataFrame(results)
    gsm_df.to_csv(args.output, index=False)
    accuracy = gsm_df["correct"].mean() * 100
    print(f"GSM8K Pass@1 Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
