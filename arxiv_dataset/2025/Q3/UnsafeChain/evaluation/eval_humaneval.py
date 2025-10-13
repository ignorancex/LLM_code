import os
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="humaneval_eval.csv")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file.")
    client = OpenAI(api_key=openai_api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto").eval()
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ds = load_dataset("openai/openai_humaneval", split=f"test[:{args.n}]")
    def generate(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
    def judge_code(prompt, response, reference):
        judge_prompt = f"""You are a Python code evaluator.\n\nPrompt:\n{prompt}\n\nModel's Code:\n{response}\n\nReference Solution:\n{reference}\n\nIs the model's response functionally and semantically equivalent to the reference solution? Just check if the solution is logically correct not the documentations, etc. Respond with just \"correct\" or \"incorrect\" and nothing else."""
        try:
            result = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10,
            )
            return result.choices[0].message.content.strip()
        except Exception:
            return "ERROR"
    rows = []
    for row in tqdm(ds, total=args.n, desc="Evaluating HumanEval"):
        prompt = row["prompt"]
        answer = row["canonical_solution"]
        response = generate(prompt)
        label = judge_code(prompt, response, answer)
        rows.append({
            "prompt": prompt,
            "response": response,
            "judge_label": label
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"✅ Saved → {args.output}")

if __name__ == "__main__":
    main()
