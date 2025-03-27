import pandas as pd
import os
import json
import math
import re
from openai import AzureOpenAI
from tqdm import tqdm
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Generate text using OpenAI.")
parser.add_argument(
    "--setting", type=str, default="1", help="Setting of text to generate"
)
parser.add_argument(
    "--runs", type=int, default=10, help="Number of times to run the experiment"
)
args = parser.parse_args()

setting = str(args.setting)
runs = args.runs

# Load secrets for API access
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# Initialize OpenAI client
client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

# Configuration for the model and completions
MODEL = "gpt-4o"  # Change to a valid model name
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOP_P = 0.95
letters = "abcdefghijklmnopqrstuvwxyz"

# Read template pairs from CSV
df = pd.read_csv("../data/t2t_setting" + setting + ".csv")
# df = df.head(1)

# System and user instruction
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

def get_completion(client, model, message_text, index, key, letter):
    message = None
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=message_text,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )

        # Extract answer and compute confidence
        message_content = completion.choices[0].message.content
        if message_content:
            match = re.search(r"\[(.*?)\]", message_content)
            if match:
                message = match.group(1).lower().strip().replace(" ", "")
    except Exception as e:
        print(
            f"Error processing row {index} for template {key} and letter {letter}: {e}"
        )
    return index, key, letter, message

def run_experiment(df, client, model, run_number):
    answers = {f"answer_{letter}": [None] * len(df) for letter in letters}

    with ThreadPoolExecutor() as executor:
        futures = []
        for index, row in df.iterrows():
            key = "template"
            for letter in letters:
                sentence = f"{row[key]} [{letter}__]."
                message_text = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{INSTRUCTION} {sentence} Answer : "},
                ]
                futures.append(executor.submit(get_completion, client, model, message_text, index, key, letter))

        for future in tqdm(as_completed(futures), total=len(futures)):
            index, key, letter, message = future.result()
            answer_key = f"answer_{letter}"
            answers[answer_key][index] = message

    answers_df = pd.DataFrame(answers)
    answers_df['run'] = run_number
    df_combined = pd.concat([df, answers_df], axis=1)
    return df_combined

all_runs_results = []
for run in tqdm(range(runs), total=runs, desc="Runs:"):
    print(f"Run {run+1}/{runs}")
    result_df = run_experiment(df, client, MODEL, run+1)
    all_runs_results.append(result_df)

final_result_df = pd.concat(all_runs_results)
print(final_result_df.head())

final_result_df.to_csv(f"../outputs/gpt4/t2t_setting{setting}_answers_10_runs.csv", index=False)
