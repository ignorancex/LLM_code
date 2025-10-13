import json
import re
from tqdm import tqdm
from openai import OpenAI


def build_prompt(source, output, reference):
    return f"""
You are a linguistic expert evaluating machine-translated dialectal text. Rate the translation on the following dimensions:

1. Fluency (1–5): Is the output grammatically correct and natural in the Hutsul dialect?
2. Adequacy (1–5): Does the output preserve the meaning of the original source?
3. Dialectal Quality (1–5): Does the output reflect the expected phonological, lexical, and grammatical properties of the Hutsul dialect?

Return your answer in this exact JSON format:

{{"fluency": x, "adequacy": y, "dialect": z}}

Do not explain your ratings.

Source (Standard Ukrainian): {source}

Model Output (Hutsul): {output}

Reference (Hutsul): {reference}
""".strip()


# Safe JSON extraction
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

client = OpenAI(api_key="")

input_path_l = ["translated_gpt4o_results", "translated_results_llama_original", "translated_results_llama",
                "translated_results_mistral_original_cleaned", "translated_results_mistral_cleaned"]

for input_path in input_path_l:
    output_path = input_path + "_llm_judged_results.jsonl"

    with open(input_path+".jsonl", "r", encoding="utf-8") as f:
        data =  [json.loads(line) for _, line in zip(range(400), f)]

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Scoring with GPT-4o"):
            source = item["Source (Standard Ukrainian)"]
            output = item["Model Output (Hutsul)"]
            reference = item["Reference (Hutsul)"]

            prompt = build_prompt(source, output, reference)

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150
                )

                raw_response = response.choices[0].message.content.strip()
                scores = extract_json(raw_response)

                if not scores:
                    print(f"Could not parse LLM output:\n{raw_response}\n")
                    continue

                item["LLM Evaluation"] = scores
                json.dump(item, out_f, ensure_ascii=False)
                out_f.write("\n")

            except Exception as e:
                print(f"Error on input: {source[:40]}... → {e}")
                continue

    print(f"LLM scoring complete. Results saved to {output_path}")

import json
files = ["translated_gpt4o_results_llm_judged_results.jsonl", 
         "translated_results_llama_llm_judged_results.jsonl",
        "translated_results_llama_original_llm_judged_results.jsonl",
                "translated_results_mistral_cleaned_llm_judged_results.jsonl", 
                "translated_results_mistral_original_cleaned_llm_judged_results.jsonl"]

def average_scores(file_path):
    fluency, adequacy, dialect = [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            scores = item.get("LLM Evaluation", {})
            if {"fluency", "adequacy", "dialect"}.issubset(scores):
                fluency.append(scores["fluency"])
                adequacy.append(scores["adequacy"])
                dialect.append(scores["dialect"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "fluency": avg(fluency),
        "adequacy": avg(adequacy),
        "dialect": avg(dialect),
        "count": len(fluency)
    }

print(f"{'File':<35} {'Fluency':>8} {'Adequacy':>9} {'Dialect':>9} {'Count':>7}")
print("-" * 70)

for file in files:
    scores = average_scores(file)
    print(f"{file:<35} {scores['fluency']:>8.2f} {scores['adequacy']:>9.2f} {scores['dialect']:>9.2f} {scores['count']:>7}")
