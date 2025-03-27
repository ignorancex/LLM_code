import pandas as pd
import re
import numpy as np
import argparse
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Function to get BERT embedding for a given text."""
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state.mean(dim=1).squeeze()

# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate cosine similarity using BERT embeddings.")
parser.add_argument("--setting", type=str, default="1", help="Identifier for the input file to use")
args = parser.parse_args()
setting = args.setting

# Define file paths and template column names for each setting
file_paths = {
    "1": "../outputs/gpt4/t2t_setting1_answers.csv",
    "2": "../outputs/gpt4/t2t_setting2_answers.csv",
    "3": "../outputs/gpt4/t2t_setting3_answers.csv",
    "4": "../outputs/gpt4/t2t_setting4_answers.csv",
    "5": "../outputs/gpt4/t2t_setting5_answers.csv",
}

template_columns = {
    "1": ("template1_singular", "template2_singular"),
    "2": ("template1_plural", "template2_plural"),
    "3": ("template1", "template2"),
    "4": ("template1", "template2"),
    "5": ("template1", "template2"),
}

# Load the data
input_file = file_paths[setting]
data = pd.read_csv(input_file)

def clean_template(text):
    """Clean up text by removing certain words and standardizing."""
    text = str(text)
    pattern = r"\b(?:a/an|a|an|this|is|are)\b"
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

template1, template2 = template_columns[setting]
data["target1"] = data[template1].apply(clean_template)
data["target2"] = data[template2].apply(clean_template)

# Calculate cosine similarity for each target-answer pair
for i in tqdm(range(1, 27), total=26):  # Assuming answers from a to z
    answer_prefix1 = f"answer1_{chr(96+i)}"
    answer_prefix2 = f"answer2_{chr(96+i)}"
    
    # Directly compute cosine similarity without storing embeddings
    data[f"cosine_similarity1_{chr(96+i)}"] = data.apply(
        lambda row: cosine_similarity(
            get_bert_embedding(str(row["target1"])).reshape(1, -1),
            get_bert_embedding(str(row[answer_prefix1])).reshape(1, -1)
        )[0][0], axis=1)
    
    data[f"cosine_similarity2_{chr(96+i)}"] = data.apply(
        lambda row: cosine_similarity(
            get_bert_embedding(str(row["target2"])).reshape(1, -1),
            get_bert_embedding(str(row[answer_prefix2])).reshape(1, -1)
        )[0][0], axis=1)

# Save the results to a new CSV file
output_file = f"../analysis/closed_source/t2t_gpt4/setting{setting}_word_similarity.csv"
data.to_csv(output_file, index=False)

print("Cosine similarity calculation complete using BERT. File saved to:", output_file)
