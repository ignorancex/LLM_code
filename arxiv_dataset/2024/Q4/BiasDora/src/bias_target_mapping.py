import pandas as pd
import re
import json

# Load the CSV file
input_path = "../data/t2t_setting1.csv"
output_path = "../data/bias_target_mapping.json"

# Load the data
data = pd.read_csv(input_path)

# Define the clean_template function
def clean_template(text):
    """Clean up text by removing certain words and standardizing."""
    text = str(text)
    pattern = r"\b(?:a/an|a|an|this|is|are)\b"
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text

# Apply the clean_template function to extract targets
data["target"] = data["template"].apply(clean_template)

# Create a dictionary to store the bias type to target mapping
bias_target_mapping = {}

# Iterate over the rows and populate the dictionary
for index, row in data.iterrows():
    bias_type = row['bias_type']
    target = row['target']
    
    if bias_type not in bias_target_mapping:
        bias_target_mapping[bias_type] = []
    
    # Add the target to the bias type if it's not already present
    if target not in bias_target_mapping[bias_type]:
        bias_target_mapping[bias_type].append(target)

# Save the mapping to a JSON file
with open(output_path, "w") as json_file:
    json.dump(bias_target_mapping, json_file, indent=4)

print(f"Bias to target mapping saved to {output_path}")
