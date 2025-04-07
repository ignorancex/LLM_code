import openai
import pandas as pd
import json

# Load the CSV file
file_path = "FILE_PATH"
df = pd.read_csv(file_path)

# Prepare the list to hold all formatted data
formatted_data = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    dna_sequence = row["DNA Sequence"].upper()
    drug_class = row["Drug Class"]

    # Create the formatted message as required
    formatted_message = {
        "messages": [
            {"role": "system", "content": "You are a DNA and antimicrobial resistance expert."},
            {"role": "user", "content": f"Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence ({dna_sequence})?"},
            {"role": "assistant", "content": f"answer: {drug_class}"}
        ]
    }

    # Append to the list
    formatted_data.append(formatted_message)

# Save the formatted data to a JSON file --> not JSON, should be JSONL
# output_file_path = "/content/drive/MyDrive/applicationsML/nlp_project/datasets/formatted_train_data.json"
# with open(output_file_path, 'w') as json_file:
#     json.dump(formatted_data, json_file, indent=4)

# print(f"Formatted data saved to {output_file_path}")


output_file_path = "FILE_PATH"
with open(output_file_path, 'w') as jsonl_file:
    for entry in formatted_data:
        jsonl_file.write(json.dumps(entry) + "\n")

print(f"Formatted data saved to {output_file_path}")



client.files.create(
  file=open("TRAIN_DATASET_FILE_PATH", "rb"),
  purpose="fine-tune"
)



client.fine_tuning.jobs.create(
  training_file="FILE_ID",  # should use file id which provided after uploading
  model="gpt-4o-mini-2024-07-18"# not like this: "gpt-4o-mini" , should be more specific like "gpt-4o-mini-2024-07-18"
)

