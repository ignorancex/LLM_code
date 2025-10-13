import os
import json

# Set the data directory
data_dir = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/checklist/trained_reference_based"

# Read the base data file
base_file = os.path.join(data_dir, "results_base_data.json")
with open(base_file, 'r') as f:
    base_data = json.load(f)

# Extract the base value for "test_avg_bleurt"
base_bleurt = base_data.get("test_avg_bleurt")

# Initialize an empty dictionary to store the relative results
relative_results = {}

# Iterate over all files in the data directory
for filename in os.listdir(data_dir):
    # Check if file starts with "_results" and is a JSON file
    if filename.startswith("results") and filename.endswith(".json"):
        file_path = os.path.join(data_dir, filename)

        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract "test_avg_bleurt" from the file
        test_bleurt = data.get("test_avg_bleurt")

        if test_bleurt is not None:
            # Calculate the relative difference
            relative_difference = (test_bleurt - base_bleurt) / base_bleurt

            # Save the relative difference to the dictionary with the file name as the key
            relative_results[filename] = relative_difference

# Save the resulting scores in a new JSON file
output_file = os.path.join(data_dir, "results_relative.json")
with open(output_file, 'w') as f:
    json.dump(relative_results, f, indent=4)

print("Relative results saved to 'results_relative.json'")
