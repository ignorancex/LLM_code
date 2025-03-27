import pandas as pd
import json
import argparse
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np

# Download stopwords if not already present
# nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate TF-IDF scores for target and individual words in outputs.")
parser.add_argument("--model", type=str, required=True, choices=["gpt", "llava"], help="The model to use for the experiment (gpt or llava).")
parser.add_argument("--setting", type=str, default="1", help="Identifier for the input file to use")
args = parser.parse_args()
model = args.model
setting = args.setting

# Define file paths for each setting and model
file_paths = {
    "gpt": {
        "1": "../outputs/gpt4o/i2t_setting1_answers.csv",
        "2": "../outputs/gpt4o/i2t_setting2_answers.csv",
        "3": "../outputs/gpt4o/i2t_setting3_answers.csv",
        "4": "../outputs/gpt4o/i2t_setting4_answers.csv",
        "5": "../outputs/gpt4o/i2t_setting5_answers.csv",
        "6": "../outputs/gpt4o/i2t_setting6_answers.csv",
        "7": "../outputs/gpt4o/i2t_setting7_answers.csv"
    },
    "llava": {
        "1": "../outputs/llava/i2t_setting1_answers.csv",
        "2": "../outputs/llava/i2t_setting2_answers.csv",
        "3": "../outputs/llava/i2t_setting3_answers.csv",
        "4": "../outputs/llava/i2t_setting4_answers.csv",
        "5": "../outputs/llava/i2t_setting5_answers.csv",
        "6": "../outputs/llava/i2t_setting6_answers.csv",
        "7": "../outputs/llava/i2t_setting7_answers.csv"
    }
}

# Load the data
input_file = file_paths[model][setting]
data = pd.read_csv(input_file)

# Ensure 'bias_type' exists in data
if 'bias_type' not in data.columns:
    data['bias_type'] = 'Unknown'  # Assign a default value if 'bias_type' column is missing

# Function to extract the target
def extract_target(id_str, model):
    if model == "gpt":
        # For GPT: Find the substring between 'target_' and the next '_natural' or '_vivid'
        start_idx = id_str.find('target_') + len('target_')
        end_idx = id_str.find('_natural', start_idx)
        if end_idx == -1:
            end_idx = id_str.find('_vivid', start_idx)
    elif model == "llava":
        # For LLava: Find the substring between 'target_' and the next '_'
        start_idx = id_str.find('target_') + len('target_')
        end_idx = id_str.find('_', start_idx)
        # To handle formats like 'id_0_target_child_1.png' where the last segment could be a number or other character
        while end_idx != -1 and not id_str[end_idx + 1].isdigit():
            end_idx = id_str.find('_', end_idx + 1)
        if end_idx == -1:
            end_idx = len(id_str)  # If no trailing underscore, take the rest of the string

    # Replace underscores with spaces
    return id_str[start_idx:end_idx].replace('_', ' ')

# Apply the function to extract the target
data['target'] = data['id'].apply(lambda id_str: extract_target(id_str, model))


# Function to preprocess the text
def preprocess(text):
    # Tokenize and remove stopwords, lowercase
    return ' '.join([word for word in text.lower().split() if word not in stop_words and word.isalpha()])

# Preprocess the output
data['processed_output'] = data['generated_output'].apply(preprocess)

# Initialize dictionaries for term frequencies and document frequencies
term_freqs = defaultdict(lambda: defaultdict(int))
doc_freqs = defaultdict(int)
total_docs = len(data)

# Populate term frequencies and document frequencies
for index, row in data.iterrows():
    words = row['processed_output'].split()
    target = row['target']
    unique_words = set(words)
    
    for word in words:
        term_freqs[target][word] += 1
        
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

for index, row in data.iterrows():
    target = row['target']
    for word, score in tfidf_scores[target].items():
        output_data.append({
            "id": row['id'],
            "target": target,
            "word": word,
            "tfidf_score": score
        })

tfidf_df = pd.DataFrame(output_data)
# print(tfidf_df.head())

# Determine the output file path based on the model type
output_file = f"../analysis/{'closed_source' if model == 'gpt' else 'open_source'}/i2t_{model}/setting{setting}_word_cooccurence.csv"
tfidf_df.to_csv(output_file, index=False)

# Print completion message
print("TF-IDF calculation complete. File saved to:", output_file)
