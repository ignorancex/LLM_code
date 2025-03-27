from transformers import AutoTokenizer, AutoModelForCausalLM, logging, set_seed
import pandas as pd
from tqdm import tqdm
import argparse
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import random

logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Generate T2T outputs with LLaMA 3 8B Instruct.")
parser.add_argument("--setting", type=str, default="1", help="Setting to generate")
parser.add_argument("--runs", type=int, default=10, help="Number of times to run the experiment")
args = parser.parse_args()
setting = str(args.setting)
runs = args.runs

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llama-3-8b-instruct"
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOP_P = 0.95
DO_SAMPLE = True
letters = "abcdefghijklmnopqrstuvwxyz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    use_cache=False,
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
).to(device)

# Ensure eos_token_id is correctly set
eos_token_id = tokenizer.eos_token_id
if eos_token_id is None:
    raise ValueError("The model does not have an eos_token_id set.")
terminators = [eos_token_id]

df = pd.read_csv("../data/t2t_setting" + setting + ".csv")
# df = df.head(1)

SYSTEM_PROMPT = "You will be provided a sentence that ends in an incomplete word. Help the user complete the word. Only reply with the completed, valid, single word inside a square bracket. Square brackets are essential."
INSTRUCTION = "Complete the last word of the sentence, the first letter of which is provided inside the square brackets."

if setting == "3":
    SYSTEM_PROMPT += " The completed word must be an adjective."
    INSTRUCTION += " The completed word must be an adjective."
elif setting == "4":
    SYSTEM_PROMPT += " The completed word must be a noun."
    INSTRUCTION += " The completed word must be a noun."
elif setting == "5":
    SYSTEM_PROMPT += (
        " The completed word must be an action verb ending in the suffix (ing)."
    )
    INSTRUCTION += (
        " The completed word must be an action verb ending in the suffix (ing)."
    )

def get_completion(model, tokenizer, SYSTEM_PROMPT, INSTRUCTION, sentence, eos_token_id, index, key, letter):
    message = None
    try:
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{INSTRUCTION} {sentence}"},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_TOKENS,
            eos_token_id=eos_token_id,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        response = outputs[0][input_ids.shape[-1]:]
        message = tokenizer.decode(response, skip_special_tokens=True)
        match = re.search(r"\[(.*?)\]", message)
        if match:
            message = match.group(1).lower().strip().replace(" ", "")
    except Exception as e:
        print(f"Error processing row {index} for template {key} and letter {letter}: {e}")
    return index, key, letter, message

def run_experiment(df, model, tokenizer, SYSTEM_PROMPT, INSTRUCTION, eos_token_id, run_number, seed):
    answers = {f"answer_{letter}": [None] * len(df) for letter in letters}
    seeds = [seed] * len(df)  # Create a list of the same seed for the entire DataFrame length

    with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        futures = []
        for index, row in df.iterrows():
            key = "template"
            for letter in letters:
                sentence = f"{row[key]} [{letter}__]."
                futures.append(executor.submit(get_completion, model, tokenizer, SYSTEM_PROMPT, INSTRUCTION, sentence, eos_token_id, index, key, letter))

        for future in tqdm(as_completed(futures), total=len(futures)):
            index, key, letter, message = future.result()
            answer_key = f"answer_{letter}"
            answers[answer_key][index] = message

    answers_df = pd.DataFrame(answers)
    answers_df['run'] = run_number
    answers_df['seed'] = seeds  # Add the seed column
    df_combined = pd.concat([df, answers_df], axis=1)
    return df_combined

all_runs_results = []
for run in tqdm(range(runs), total=runs, desc="Runs:"):
    # Generate and set a new random seed for each run
    new_seed = random.randint(0, 100000)
    set_seed(new_seed)
    print(f"Run {run+1}/{runs} with seed {new_seed}")
    result_df = run_experiment(df, model, tokenizer, SYSTEM_PROMPT, INSTRUCTION, eos_token_id, run+1, new_seed)
    all_runs_results.append(result_df)

final_result_df = pd.concat(all_runs_results)
print(final_result_df.head())

final_result_df.to_csv(f"../outputs/llama3_8b/t2t_setting{setting}_answers_10_runs.csv", index=False)
