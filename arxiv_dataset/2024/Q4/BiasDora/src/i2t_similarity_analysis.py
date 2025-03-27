import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import argparse
from tqdm import tqdm
tqdm.pandas()

stop_words = set(stopwords.words('english'))
# Setup the argument parser
parser = argparse.ArgumentParser(description="Calculate cosine similarities using BERT embeddings.")
parser.add_argument("--setting", type=str, default="1", help="Identifier for the input file to use")
args = parser.parse_args()
setting = args.setting

# Define file paths for each setting
file_paths = {
    "1": "../outputs/gpt4o/i2t_setting1_answers.csv",
    "2": "../outputs/gpt4o/i2t_setting2_answers.csv",
    "3": "../outputs/gpt4o/i2t_setting3_answers.csv",
    "4": "../outputs/gpt4o/i2t_setting4_answers.csv",
    "5": "../outputs/gpt4o/i2t_setting5_answers.csv",
    "6": "../outputs/gpt4o/i2t_setting6_answers.csv",
    "7": "../outputs/gpt4o/i2t_setting7_answers.csv"
}

# Load the data
input_file = file_paths[setting]
data = pd.read_csv(input_file)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_embed(text):
    # Encode the text, and compute BERT embeddings
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Take the mean of all token embeddings to get the sentence embedding
    return model_output.last_hidden_state.mean(dim=1)

# Function to extract and process target
def extract_target(id_str):
    start_idx = id_str.find('target_') + len('target_')
    end_idx = id_str.find('_natural', start_idx)
    if end_idx == -1:
        end_idx = id_str.find('_vivid', start_idx)
    return id_str[start_idx:end_idx].replace('_', ' ')

data['target'] = data['id'].progress_apply(extract_target)

# Preprocessing function for the text
def preprocess(text):
    # Tokenize, remove stopwords, lowercase, and filter non-alphabetical characters
    return ' '.join([word for word in text.lower().split() if word not in stop_words and word.isalpha()])

data['processed_output'] = data['generated_output'].progress_apply(preprocess)

# Calculate embeddings for each target and text
data['target_embedding'] = data['target'].progress_apply(bert_embed)
data['text_embedding'] = data['processed_output'].progress_apply(bert_embed)

# Calculate cosine similarities
def calculate_cosine_similarity(row):
    similarities = {}
    for word in row['processed_output'].split():
        word_embedding = bert_embed(word).reshape(1, -1)
        similarity = cosine_similarity(row['target_embedding'], word_embedding)[0][0]
        similarities[f"{row['target']} || {word}"] = similarity
    return similarities

data['cosine_similarity'] = data.progress_apply(calculate_cosine_similarity, axis=1)

# Save results to CSV
output_file = f"../analysis/closed_source/i2t_gpt4o/setting{setting}_word_similarity.csv"
# data.to_csv(output_file, index=False)
data.drop(columns=['target_embedding', 'text_embedding']).to_csv(output_file, index=False)


print("Cosine similarity calculation complete. File saved to:", output_file)
