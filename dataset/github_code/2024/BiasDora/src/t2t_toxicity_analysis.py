import pandas as pd
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import os

# Setup the argument parser
parser = argparse.ArgumentParser(
    description="Run toxicity analysis on specified setting and model."
)
parser.add_argument(
    "--modelname", type=str, required=True, choices=["gpt4", "llama"], help="The model to use for the experiment (gpt4 or llama)."
)
parser.add_argument(
    "--setting", type=str, required=True, choices=["1", "2", "3", "4", "5"], help="Identifier for the input file to use"
)
args = parser.parse_args()
modelname = args.modelname
setting = args.setting

# Define file paths for each model and setting
file_map = {
    "gpt4": [
        f"../results/t2t_gpt4/setting{setting}_pvalue_significant_tfidf.csv",
        f"../results/t2t_gpt4/setting{setting}_significant_tfidf.csv"
    ],
    "llama": [
        f"../results/t2t_llama/setting{setting}_pvalue_significant_tfidf.csv",
        f"../results/t2t_llama/setting{setting}_significant_tfidf.csv"
    ]
}

# Ensure the output directory exists
output_directory = f"../analysis/{'closed_source' if modelname == 'gpt4' else 'open_source'}/t2t_{modelname}"
os.makedirs(output_directory, exist_ok=True)

# Initialize the text classification pipeline
pipe = pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")

def predict_toxicity(text):
    """Predict toxicity and return both label and score."""
    result = pipe(text)[0]
    label = result['label']
    score = result['score']
    return label, score

# Process each file path in the file map
for file_path in file_map[modelname]:
    if os.path.exists(file_path):
        # Load data
        data = pd.read_csv(file_path)
        # data = data.head(100)  # Limit to 10 rows for this example

        # Initialize new columns for toxicity analysis
        data['toxicity_label'] = None
        data['toxicity_score'] = None

        # Apply the predict_toxicity function and store the results in the new columns
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            if pd.notna(row['concatenated_sentence']):
                label, score = predict_toxicity(row['concatenated_sentence'])
                data.at[index, 'toxicity_label'] = label
                data.at[index, 'toxicity_score'] = score

        # Extract the base file name for the output
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Determine the appropriate output directory based on the model type
        output_directory = f"../analysis/{'closed_source/t2t_gpt4' if modelname == 'gpt4' else 'open_source/t2t_llama'}"
        os.makedirs(output_directory, exist_ok=True)

        # Save the data with toxicity scores to a new file
        output_file_path = os.path.join(output_directory, f"{base_file_name}_toxicity.csv")
        data.to_csv(output_file_path, index=False)

        # Print completion message for each file
        print(f"Toxicity analysis complete for {file_path}. File saved to: {output_file_path}")
    else:
        print(f"File not found: {file_path}")
