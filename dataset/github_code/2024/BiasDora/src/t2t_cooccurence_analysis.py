import pandas as pd
import json
import argparse
from collections import defaultdict
import numpy as np
import re

# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate TF-IDF scores for target and individual words in outputs.")
parser.add_argument("--model", type=str, required=True, choices=["gpt4", "llama"], help="The model to use for the experiment (gpt4 or llama).")
parser.add_argument("--setting", type=str, default="1", help="Identifier for the input file to use")
args = parser.parse_args()
model = args.model
setting = args.setting

# Define file paths for each setting and model
file_paths = {
    "gpt4": {
        "1": "../outputs/gpt4/t2t_setting1_answers_10_runs.csv",
        "2": "../outputs/gpt4/t2t_setting2_answers_10_runs.csv",
        "3": "../outputs/gpt4/t2t_setting3_answers_10_runs.csv",
        "4": "../outputs/gpt4/t2t_setting4_answers_10_runs.csv",
        "5": "../outputs/gpt4/t2t_setting5_answers_10_runs.csv"
    },
    "llama": {
        "1": "../outputs/llama3_8b/t2t_setting1_answers_10_runs.csv",
        "2": "../outputs/llama3_8b/t2t_setting2_answers_10_runs.csv",
        "3": "../outputs/llama3_8b/t2t_setting3_answers_10_runs.csv",
        "4": "../outputs/llama3_8b/t2t_setting4_answers_10_runs.csv",
        "5": "../outputs/llama3_8b/t2t_setting5_answers_10_runs.csv"
    }
}

# Load the data
input_file = file_paths[model][setting]
data = pd.read_csv(input_file)

# Clean up the templates if necessary
def clean_template(text):
    """Clean up text by removing certain words and standardizing."""
    text = str(text)
    pattern = r"\b(?:a/an|a|an|this|is|are)\b"
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

# Apply the clean_template function to extract targets
data["target"] = data["template"].apply(clean_template)

# List of answer columns
answer_columns = [f"answer_{chr(i)}" for i in range(ord('a'), ord('z') + 1)]

# Ensure 'bias_type' exists in data
if 'bias_type' not in data.columns:
    data['bias_type'] = 'Unknown'  # Assign a default value if 'bias_type' column is missing

# Initialize dictionaries for term frequencies and document frequencies
term_freqs = defaultdict(lambda: defaultdict(int))
doc_freqs = defaultdict(int)
total_docs = len(data) * len(answer_columns)

# Populate term frequencies and document frequencies
for index, row in data.iterrows():
    target = row['target']
    unique_words = set()
    
    for column in answer_columns:
        if pd.notna(row[column]):
            word = row[column].lower()
            term_freqs[target][word] += 1
            unique_words.add(word)
    
    for word in unique_words:
        doc_freqs[word] += 1

# Calculate IDF
idf_scores = {word: np.log(total_docs / freq) for word, freq in doc_freqs.items()}

# Calculate TF-IDF scores
tfidf_scores = defaultdict(lambda: defaultdict(float))

for target, words in term_freqs.items():
    for word, count in words.items():
        tf = count  # Term frequency in this context is just the count
        idf = idf_scores[word]
        tfidf_scores[target][word] = tf * idf

# Convert to DataFrame
output_data = []

# Modify the loop to add concatenated sentence column
for index, row in data.iterrows():
    target = row['target']
    for column in answer_columns:
        if pd.notna(row[column]):
            word = row[column].lower()
            concatenated_sentence = f"{row['template']} {word}".strip()
            score = tfidf_scores[target][word]
            output_data.append({
                "id": f"{row['bias_type']}_{index}",
                "target": target,
                "word": word,
                "concatenated_sentence": concatenated_sentence,
                "tfidf_score": score
            })


tfidf_df = pd.DataFrame(output_data)

# Determine the output file path based on the model type
output_file = f"../analysis/{'closed_source' if model == 'gpt4' else 'open_source'}/t2t_{model}/setting{setting}_word_cooccurence.csv"
tfidf_df.to_csv(output_file, index=False)

# Print completion message
print("TF-IDF calculation complete. File saved to:", output_file)
