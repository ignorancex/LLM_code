import pandas as pd
import os
from datasets import Dataset, DatasetDict, Features, Value, Image
from tqdm import tqdm

# Load the CSV file
file_path = 'results/dalle_images.csv'
data = pd.read_csv(file_path)

# Identify unique concepts
unique_concepts = data['concept'].unique()

# Create a dictionary to hold the splits
splits = {concept: data[data['concept'] == concept] for concept in unique_concepts}

# Display the number of items in each split
split_summary = {concept: len(splits[concept]) for concept in splits}
print(split_summary)

# Create a directory to save the splits
output_dir = 'dallestreet'
os.makedirs(output_dir, exist_ok=True)

# Save each split to a CSV file
for concept, df in splits.items():
    split_path = os.path.join(output_dir, f'{concept}.csv')
    df.to_csv(split_path, index=False)

# List of the files created for uploading
split_files = os.listdir(output_dir)
print(split_files)

# Create a DatasetDict to hold the splits
dataset_dict = DatasetDict()

# Define the features including the image column
features = Features({
    'image': Image(),  # Image feature to handle image paths
    'country': Value('string'),
    'type': Value('string')
})

# Load each split into the DatasetDict
for split_file in tqdm(split_files):
    split_path = os.path.join(output_dir, split_file)
    concept = split_file.replace('.csv', '')
    df = pd.read_csv(split_path)

    # Rename the 'image_path' column to 'image' to match the expected feature name
    df = df.rename(columns={'image_path': 'image'})

    # Drop the 'concept' column as it's not needed
    df = df.drop(columns=['concept'])

    # Create the dataset
    dataset = Dataset.from_pandas(df, features=features)
    dataset_dict[concept] = dataset

# Push the dataset to the Hub
dataset_dict.push_to_hub("iamshnoo/dallestreet")
