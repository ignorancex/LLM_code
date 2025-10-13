import os
import json

def concatenate_json_files(input_directory, output_file):
    all_json_objects = []

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(input_directory, filename)
            
            # Read JSON data from each file
            with open(filepath, 'r') as file:
                json_objects = json.load(file)
                all_json_objects.extend(json_objects)
    
    # Write the concatenated JSON objects to the output file
    with open(output_file, 'w') as file:
        json.dump(all_json_objects, file, indent=4)

input_directory = '/home/sajib/Documents/LLM_Embedding_Attack/outputs/outputs_RR_harmful_behaviors_Llama2'
output_file = '/home/sajib/Documents/LLM_Embedding_Attack/outputs/output_loss_001_llam2_RR.json'
concatenate_json_files(input_directory, output_file)
# Example usage
# input_directory = '/home/sajib/Documents/LLM_Embedding_Attack/outputs/output_gcg_harmful_behaviors'
# output_file = '/home/sajib/Documents/LLM_Embedding_Attack/outputs/output_loss_001_llam2_gcg.json'