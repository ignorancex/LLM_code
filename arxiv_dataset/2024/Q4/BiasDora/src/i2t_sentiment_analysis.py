import pandas as pd
from transformers import pipeline
import os
import argparse

# Setup the argument parser
parser = argparse.ArgumentParser(
    description="Run sentiment analysis on specified setting and model."
)
parser.add_argument(
    "--model", type=str, required=True, choices=["gpt4o", "llava"], help="The model to use for the experiment (gpt4o or llama)."
)
parser.add_argument(
    "--setting", type=str, required=True, choices=["1", "2", "3", "4", "5", "6", "7"], help="Identifier for the input file to use"
)
args = parser.parse_args()
model = args.model
setting = args.setting

# Define file paths for each model and setting
file_map = {
    "gpt4o": [
        f"../results/i2t_gpt4o/setting{setting}_pvalue_significant_tfidf.csv",
        f"../results/i2t_gpt4o/setting{setting}_significant_tfidf.csv"
    ],
    "llava": [
        f"../results/i2t_llava/setting{setting}_pvalue_significant_tfidf.csv",
        f"../results/i2t_llava/setting{setting}_significant_tfidf.csv"
    ]
}

# Ensure the output directory exists
output_directory = f"../analysis/{'closed_source' if model == 'gpt4o' else 'open_source'}/i2t_{model}"
os.makedirs(output_directory, exist_ok=True)

# Initialize the Hugging Face pipeline for sentiment analysis
sentiment_model = pipeline("sentiment-analysis")

# Loop through each file path in the file map and process them individually
for file_path in file_map[model]:
    if os.path.exists(file_path):
        # Load data
        data = pd.read_csv(file_path)
        # data = data.head(10)

        # Add a new column for sentiment analysis
        data['word_sentiment'] = data['word'].apply(
            lambda x: sentiment_model(x)[0]["label"] if pd.notna(x) else None
        )

        # Extract the base file name (without directory and extension) for the output
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Determine the appropriate output directory based on the model type
        output_directory = f"../analysis/{'closed_source/i2t_gpt4o' if model == 'gpt4o' else 'open_source/i2t_llava'}"
        os.makedirs(output_directory, exist_ok=True)

        # Save the data with sentiments to a new file in the appropriate directory
        output_file_path = os.path.join(
            output_directory, f"{base_file_name}_sentiments.csv"
        )
        data.to_csv(output_file_path, index=False)

        # Print completion message for each file
        print(f"Sentiment analysis complete for {file_path}. File saved to: {output_file_path}")
    else:
        print(f"File not found: {file_path}")
